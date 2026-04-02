"""Core service for multi-mode investment analysis."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from aegean.core.agent import Agent, AgentRegistry
from aegean.core.coordinator import ConsensusCoordinator
from aegean.core.decision_engine import WeightedDecisionEngine
from aegean.core.models import ConsensusConfig, Solution
from aegean.investment.models import (
    AgentOutput,
    AssetType,
    ConsensusResultView,
    InvestmentAnalysisRequest,
    InvestmentAnalysisResponse,
    InvestmentMetadata,
    InvestmentMode,
    InvestmentRecommendation,
    InvestmentSummary,
    MarketCode,
    RecommendationAction,
    RiskGateResult,
)
from aegean.risk.models import RiskContext, RiskRequest, RiskSubject
from aegean.risk.risk_consensus import RiskConsensusCoordinator


_SIGNAL_MAP = {
    "buy": "bullish",
    "overweight": "bullish",
    "bullish": "bullish",
    "sell": "bearish",
    "underweight": "bearish",
    "bearish": "bearish",
    "hold": "neutral",
    "watch": "neutral",
    "neutral": "neutral",
}

_ACTION_BY_SIGNAL = {
    "bullish": RecommendationAction.BUY,
    "bearish": RecommendationAction.SELL,
    "neutral": RecommendationAction.HOLD,
}

_V3_ALLOWED_ASSET_TYPES = {
    AssetType.EQUITY,
    AssetType.ETF,
    AssetType.INDEX,
    AssetType.FUND,
    AssetType.CONVERTIBLE_BOND,
    AssetType.FUTURES,
    AssetType.OPTIONS,
    AssetType.CRYPTO,
}
_V3_ALLOWED_MARKETS = {MarketCode.CN, MarketCode.HK, MarketCode.US}
_ALLOWED_RISK_PROFILES = {"conservative", "balanced", "aggressive"}
_ALLOWED_OBJECTIVES = {"defensive", "income", "balanced", "alpha"}

_ASSET_TASK_TYPE = {
    AssetType.EQUITY: "equity_analysis",
    AssetType.ETF: "etf_analysis",
    AssetType.INDEX: "index_analysis",
    AssetType.FUND: "fund_analysis",
    AssetType.CONVERTIBLE_BOND: "convertible_bond_analysis",
    AssetType.FUTURES: "futures_analysis",
    AssetType.OPTIONS: "options_analysis",
    AssetType.CRYPTO: "crypto_analysis",
}

_ASSET_SKILLS = {
    AssetType.EQUITY: ["fundamental_analysis", "equity_valuation"],
    AssetType.ETF: ["index_tracking", "allocation_analysis"],
    AssetType.INDEX: ["macro_regime", "index_structure"],
    AssetType.FUND: ["fund_selection", "manager_style"],
    AssetType.CONVERTIBLE_BOND: ["credit_analysis", "equity_optional_value"],
    AssetType.FUTURES: ["futures_basis", "margin_risk"],
    AssetType.OPTIONS: ["options_greeks", "vol_surface"],
    AssetType.CRYPTO: ["onchain_signal", "liquidity_microstructure"],
}

_ASSET_DATA_SOURCES = {
    AssetType.EQUITY: ["public_market_data", "company_fundamentals"],
    AssetType.ETF: ["public_market_data", "etf_holdings"],
    AssetType.INDEX: ["public_market_data", "macro_indicators"],
    AssetType.FUND: ["public_market_data", "fund_nav_feed"],
    AssetType.CONVERTIBLE_BOND: ["public_market_data", "bond_termsheet_feed"],
    AssetType.FUTURES: ["public_market_data", "futures_curve_feed", "margin_rules"],
    AssetType.OPTIONS: ["public_market_data", "option_chain_feed", "vol_surface_feed"],
    AssetType.CRYPTO: ["public_market_data", "exchange_depth_feed", "onchain_metrics"],
}


def _profile_exposure_cap(risk_profile: str) -> float:
    profile = (risk_profile or "balanced").lower()
    caps = {
        "conservative": 0.05,
        "balanced": 0.15,
        "aggressive": 0.30,
    }
    return caps.get(profile, 0.15)


def _objective_exposure_cap(objective: str) -> float:
    goal = (objective or "balanced").lower()
    caps = {
        "defensive": 0.08,
        "income": 0.10,
        "balanced": 0.15,
        "alpha": 0.30,
    }
    return caps.get(goal, 0.15)


def _compute_constrained_recommendation(
    action: RecommendationAction,
    position_suggestion: Dict[str, float],
    constraints: Dict[str, Any],
    risk_profile: str,
    objective: str,
    asset_symbol: str,
    asset_type: AssetType,
    portfolio_context: Optional[Dict[str, Any]],
) -> Tuple[RecommendationAction, Dict[str, float], Dict[str, Any]]:
    """Pure function for final action/exposure computation (V3 constraints-aware)."""
    next_action = action
    next_position = dict(position_suggestion)
    triggered_rules: List[str] = []

    allowed_actions = constraints.get("allowed_actions")
    if isinstance(allowed_actions, list) and allowed_actions:
        allowed = {str(a).lower() for a in allowed_actions}
        if next_action.value not in allowed:
            next_action = RecommendationAction.HOLD
            triggered_rules.append("allowed_actions")

    if bool(constraints.get("no_short")) and next_action == RecommendationAction.SELL:
        next_action = RecommendationAction.HOLD
        triggered_rules.append("no_short")

    disallowed_symbols = {
        str(s).upper() for s in (constraints.get("disallowed_symbols") or []) if str(s).strip()
    }
    disallowed_asset_types = {
        str(t).lower() for t in (constraints.get("disallowed_asset_types") or []) if str(t).strip()
    }

    portfolio = portfolio_context or {}
    disallowed_symbols.update(
        str(s).upper()
        for s in (portfolio.get("disallowed_symbols") or [])
        if str(s).strip()
    )
    disallowed_asset_types.update(
        str(t).lower()
        for t in (portfolio.get("disallowed_asset_types") or [])
        if str(t).strip()
    )

    if asset_symbol.upper() in disallowed_symbols:
        next_action = RecommendationAction.HOLD
        triggered_rules.append("disallowed_symbols")

    if asset_type.value.lower() in disallowed_asset_types:
        next_action = RecommendationAction.HOLD
        triggered_rules.append("disallowed_asset_types")

    target = float(next_position.get("target_exposure_pct", 0.0) or 0.0)

    caps = {
        "model_output": target,
        "risk_profile": _profile_exposure_cap(risk_profile),
        "objective": _objective_exposure_cap(objective),
    }

    max_exposure = constraints.get("max_exposure_pct")
    if max_exposure is not None:
        caps["max_exposure_pct"] = max(float(max_exposure), 0.0)

    max_single_from_constraints = constraints.get("max_single_position_pct")
    if max_single_from_constraints is not None:
        caps["max_single_position_pct"] = max(float(max_single_from_constraints), 0.0)

    max_single_from_portfolio = portfolio.get("max_single_position_pct")
    if max_single_from_portfolio is not None:
        caps["portfolio_max_single_position_pct"] = max(float(max_single_from_portfolio), 0.0)

    current_total_exposure_pct = float(portfolio.get("current_total_exposure_pct", 0.0) or 0.0)
    max_total_exposure_pct = portfolio.get("max_total_exposure_pct")
    if max_total_exposure_pct is not None:
        remaining_budget = max(float(max_total_exposure_pct) - current_total_exposure_pct, 0.0)
        caps["portfolio_remaining_budget_pct"] = remaining_budget

    target = min(caps.values()) if caps else 0.0

    if next_action == RecommendationAction.SELL:
        target = 0.0
        triggered_rules.append("sell_forces_zero")
    if next_action == RecommendationAction.HOLD:
        hold_cap = 0.05
        if target > hold_cap:
            triggered_rules.append("hold_cap")
        target = min(target, hold_cap)

    position_before = float(position_suggestion.get("target_exposure_pct", 0.0) or 0.0)
    next_position["target_exposure_pct"] = max(target, 0.0)

    max_dd = constraints.get("max_drawdown_guard_pct")
    if max_dd is not None:
        next_position["max_drawdown_guard_pct"] = max(float(max_dd), 0.0)
        triggered_rules.append("max_drawdown_guard_pct")

    effective_caps = {k: float(v) for k, v in caps.items()}
    min_cap = min(effective_caps.values()) if effective_caps else 0.0
    cap_owner = next((k for k, v in effective_caps.items() if v == min_cap), "none")

    summary = {
        "input_action": action.value,
        "output_action": next_action.value,
        "input_target_exposure_pct": position_before,
        "output_target_exposure_pct": next_position["target_exposure_pct"],
        "effective_caps": effective_caps,
        "binding_cap": cap_owner,
        "triggered_rules": sorted(set(triggered_rules)),
        "asset_symbol": asset_symbol,
        "asset_type": asset_type.value,
    }

    return next_action, next_position, summary


EventSink = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


class InvestmentAnalysisService:
    """Investment analysis orchestrator across fast/auto/collaborate/roundtable modes."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        memory_system: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        risk_coordinator: Optional[RiskConsensusCoordinator] = None,
    ):
        self.agent_registry = agent_registry
        self.memory_system = memory_system
        self.llm_client = llm_client
        self.risk_coordinator = risk_coordinator

    async def analyze(
        self,
        request: InvestmentAnalysisRequest,
        event_sink: EventSink = None,
    ) -> InvestmentAnalysisResponse:
        started = time.time()
        request_id = f"inv-{uuid.uuid4().hex[:12]}"
        task_type = self._task_type_for_asset(request.asset.asset_type)
        selected_skills = self._selected_skills_for_asset(request.asset.asset_type)

        await self._emit(event_sink, "analysis_started", request_id=request_id, mode=request.mode.value)
        self._validate_request(request)
        await self._emit(event_sink, "request_validated", request_id=request_id)

        agents = self._select_agents(request.mode, task_type)
        await self._emit(
            event_sink,
            "agents_selected",
            request_id=request_id,
            agent_ids=[a.agent_id for a in agents],
            task_type=task_type,
            selected_skills=selected_skills,
        )

        task = self._build_task_prompt(request, task_type, selected_skills)
        solutions = await self._collect_solutions(agents, task, event_sink, request_id)
        agent_outputs = [self._solution_to_output(sol) for sol in solutions]

        recommendation, summary = self._merge_outputs(agent_outputs)

        consensus_view = ConsensusResultView(enabled=False)
        if request.mode == InvestmentMode.ROUNDTABLE and agents:
            await self._emit(event_sink, "roundtable_started", request_id=request_id)
            consensus_view, recommendation = await self._run_roundtable(task, agents, recommendation)
            await self._emit(
                event_sink,
                "roundtable_finished",
                request_id=request_id,
                rounds_used=consensus_view.rounds_used,
                consensus_reached=consensus_view.consensus_reached,
            )

        recommendation, constraints_summary = self._apply_constraints(request, recommendation)
        await self._emit(
            event_sink,
            "constraints_applied",
            request_id=request_id,
            action=recommendation.action.value,
            target_exposure_pct=recommendation.position_suggestion.get("target_exposure_pct", 0.0),
            constraints_applied_summary=constraints_summary,
        )
        await self._emit(
            event_sink,
            "recommendation_ready",
            request_id=request_id,
            action=recommendation.action.value,
            confidence=recommendation.confidence,
        )

        risk_gate = await self._evaluate_risk(request, recommendation)
        await self._emit(
            event_sink,
            "risk_gate_finished",
            request_id=request_id,
            status=risk_gate.status,
            risk_level=risk_gate.risk_level,
        )

        report_md = self._build_report_markdown(
            request,
            recommendation,
            summary,
            agent_outputs,
            consensus_view,
            risk_gate,
        )
        metadata = InvestmentMetadata(
            token_usage=self._aggregate_tokens(solutions),
            latency_ms=int((time.time() - started) * 1000),
            data_sources=self._asset_data_sources_for(request.asset.asset_type),
            selected_skills=selected_skills,
            task_type=task_type,
            constraints_applied_summary=constraints_summary,
        )

        response = InvestmentAnalysisResponse(
            request_id=request_id,
            mode=request.mode,
            asset=request.asset,
            timeframe=request.timeframe,
            recommendation=recommendation,
            summary=summary,
            agent_outputs=agent_outputs,
            risk_gate=risk_gate,
            consensus=consensus_view,
            report_markdown=report_md,
            metadata=metadata,
        )
        await self._emit(event_sink, "analysis_completed", request_id=request_id)
        return response

    def _validate_request(self, request: InvestmentAnalysisRequest) -> None:
        asset_type = request.asset.asset_type
        market = request.asset.market

        risk_profile = (request.risk_profile or "balanced").lower()
        if risk_profile not in _ALLOWED_RISK_PROFILES:
            raise ValueError(
                "risk_profile must be one of: conservative, balanced, aggressive"
            )

        objective = (request.objective or "balanced").lower()
        if objective not in _ALLOWED_OBJECTIVES:
            raise ValueError(
                "objective must be one of: defensive, income, balanced, alpha"
            )

        if market not in _V3_ALLOWED_MARKETS:
            raise ValueError(
                f"Investment API supports markets CN/HK/US, got {market.value}."
            )
        if asset_type not in _V3_ALLOWED_ASSET_TYPES:
            raise ValueError(
                "Unsupported asset_type. Current API supports "
                "equity/etf/index/fund/convertible_bond/futures/options/crypto."
            )

    def _select_agents(self, mode: InvestmentMode, task_type: str) -> List[Agent]:
        all_agents = self.agent_registry.get_all_agents()
        if not all_agents:
            return []

        ranked = sorted(
            all_agents,
            key=lambda a: a.get_weight_for_task(task_type),
            reverse=True,
        )

        if mode == InvestmentMode.FAST:
            return ranked[:1]
        if mode == InvestmentMode.AUTO:
            return ranked[:2]
        if mode == InvestmentMode.COLLABORATE:
            return ranked[: min(4, len(ranked))]
        return ranked[: min(5, len(ranked))]

    async def _collect_solutions(
        self,
        agents: List[Agent],
        task: str,
        event_sink: EventSink,
        request_id: str,
    ) -> List[Solution]:
        if not agents:
            return []

        async def _run(agent: Agent) -> Tuple[Agent, Any]:
            try:
                result = await agent.generate_solution(task)
                return agent, result
            except Exception as exc:
                return agent, exc

        tasks = [asyncio.create_task(_run(agent)) for agent in agents]
        solutions: List[Solution] = []

        for done_task in asyncio.as_completed(tasks):
            agent, result = await done_task
            if isinstance(result, Exception):
                solution = Solution(
                    agent_id=agent.agent_id,
                    answer=f"Agent failed: {result}",
                    confidence=0.2,
                )
            else:
                solution = result
            solutions.append(solution)
            await self._emit(
                event_sink,
                "agent_completed",
                request_id=request_id,
                agent_id=solution.agent_id,
                confidence=solution.confidence,
            )

        return solutions

    def _build_task_prompt(
        self,
        request: InvestmentAnalysisRequest,
        task_type: str,
        selected_skills: List[str],
    ) -> str:
        facts = "\n".join(f"- {f}" for f in request.public_facts[:10])
        facts = facts or "- No extra public facts provided"
        skill_lines = "\n".join(f"- {s}" for s in selected_skills) or "- generic_analysis"

        return (
            f"You are an investment analyst. Analyze {request.asset.symbol} "
            f"({request.asset.market.value}, {request.asset.asset_type.value}) with horizon {request.timeframe.horizon}.\n"
            f"Task type: {task_type}.\n"
            f"Risk profile: {request.risk_profile}. Objective: {request.objective}.\n"
            f"Required skills:\n{skill_lines}\n"
            f"Market snapshot: {request.market_snapshot or 'N/A'}\n"
            f"Facts:\n{facts}\n"
            "Output concise recommendation with one of BUY/HOLD/SELL/WATCH and key reasons."
        )

    @staticmethod
    def _task_type_for_asset(asset_type: AssetType) -> str:
        return _ASSET_TASK_TYPE.get(asset_type, "investment_analysis")

    @staticmethod
    def _selected_skills_for_asset(asset_type: AssetType) -> List[str]:
        return _ASSET_SKILLS.get(asset_type, ["general_investment_analysis"])

    @staticmethod
    def _asset_data_sources_for(asset_type: AssetType) -> List[str]:
        return _ASSET_DATA_SOURCES.get(asset_type, ["public_market_data"])

    def _solution_to_output(self, solution: Solution) -> AgentOutput:
        signal = self._extract_signal(solution.answer)
        evidence = [ln.strip() for ln in solution.answer.split("\n") if ln.strip()][:3]
        return AgentOutput(
            agent_id=solution.agent_id,
            signal=signal,
            confidence=max(0.0, min(solution.confidence, 1.0)),
            evidence=evidence,
        )

    def _extract_signal(self, text: str) -> str:
        lower = text.lower()
        for key, mapped in _SIGNAL_MAP.items():
            if key in lower:
                return mapped
        return "neutral"

    def _merge_outputs(self, outputs: List[AgentOutput]) -> Tuple[InvestmentRecommendation, InvestmentSummary]:
        if not outputs:
            rec = InvestmentRecommendation(
                action=RecommendationAction.WATCH,
                confidence=0.3,
                position_suggestion={"target_exposure_pct": 0.0, "max_drawdown_guard_pct": 0.08},
            )
            summary = InvestmentSummary(
                thesis="Insufficient agent outputs, fallback to watch.",
                key_drivers=[],
                key_risks=["insufficient_agent_signal"],
            )
            return rec, summary

        weighted = Counter()
        for out in outputs:
            weighted[out.signal] += out.confidence

        top_signal = weighted.most_common(1)[0][0]
        total = sum(weighted.values()) or 1.0
        confidence = min(weighted[top_signal] / total, 1.0)

        action = _ACTION_BY_SIGNAL.get(top_signal, RecommendationAction.HOLD)
        exposure = (
            0.15
            if action == RecommendationAction.BUY
            else 0.05
            if action == RecommendationAction.HOLD
            else 0.0
        )

        rec = InvestmentRecommendation(
            action=action,
            confidence=round(confidence, 4),
            position_suggestion={
                "target_exposure_pct": exposure,
                "max_drawdown_guard_pct": 0.08,
            },
        )
        summary = InvestmentSummary(
            thesis=f"Overall agent sentiment is {top_signal}.",
            key_drivers=[f"{o.agent_id}:{o.signal}" for o in outputs[:4]],
            key_risks=["market_regime_shift", "event_risk"],
        )
        return rec, summary

    def _apply_constraints(
        self,
        request: InvestmentAnalysisRequest,
        recommendation: InvestmentRecommendation,
    ) -> Tuple[InvestmentRecommendation, Dict[str, Any]]:
        action, position, summary = _compute_constrained_recommendation(
            action=recommendation.action,
            position_suggestion=recommendation.position_suggestion,
            constraints=request.constraints or {},
            risk_profile=request.risk_profile,
            objective=request.objective,
            asset_symbol=request.asset.symbol,
            asset_type=request.asset.asset_type,
            portfolio_context=request.portfolio_context,
        )

        return (
            InvestmentRecommendation(
                action=action,
                confidence=recommendation.confidence,
                position_suggestion=position,
            ),
            summary,
        )

    async def _run_roundtable(
        self,
        task: str,
        agents: List[Agent],
        current_recommendation: InvestmentRecommendation,
    ) -> Tuple[ConsensusResultView, InvestmentRecommendation]:
        registry = AgentRegistry()
        for agent in agents:
            registry.register_agent(agent)

        engine = WeightedDecisionEngine(
            quorum_threshold=0.5,
            stability_horizon=2,
            agent_registry=registry,
        )
        coordinator = ConsensusCoordinator(
            agent_registry=registry,
            config=ConsensusConfig(max_rounds=3, stability_horizon=2),
            decision_engine=engine,
        )
        result = await coordinator.run_consensus(task)

        if result.final_solution:
            signal = self._extract_signal(result.final_solution.answer)
            action = _ACTION_BY_SIGNAL.get(signal, current_recommendation.action)
            recommendation = InvestmentRecommendation(
                action=action,
                confidence=max(
                    current_recommendation.confidence,
                    result.final_solution.confidence,
                ),
                position_suggestion=current_recommendation.position_suggestion,
            )
        else:
            recommendation = current_recommendation

        view = ConsensusResultView(
            enabled=True,
            rounds_used=result.rounds_used,
            consensus_reached=result.consensus_reached,
            weighted_votes={},
        )
        return view, recommendation

    async def _evaluate_risk(
        self,
        request: InvestmentAnalysisRequest,
        recommendation: InvestmentRecommendation,
    ) -> RiskGateResult:
        if not self.risk_coordinator:
            return RiskGateResult(status="pass", risk_level="low", risk_indicators=[])

        exposure = recommendation.position_suggestion.get("target_exposure_pct", 0.0)
        amount = float((request.constraints or {}).get("notional_amount", 0.0))
        if amount <= 0:
            amount = 10_000 * exposure

        rr = RiskRequest(
            subject=RiskSubject(
                subject_id=request.user_id,
                subject_type="user",
                trust_score=1.0,
            ),
            context=RiskContext(
                action_type="investment_recommendation",
                description=(
                    f"{request.asset.symbol} recommendation={recommendation.action.value} "
                    f"mode={request.mode.value}"
                ),
                amount=amount,
                currency=(request.constraints or {}).get("currency", "USD"),
                trace_context=request.custom_question,
            ),
            priority="normal",
            metadata={"investment_mode": request.mode.value},
        )

        decision = await self.risk_coordinator.evaluate(rr)
        status = "pass"
        if decision.decision.value == "challenge":
            status = "challenge"
        elif decision.decision.value == "reject":
            status = "reject"

        return RiskGateResult(
            status=status,
            risk_level=decision.risk_level.value,
            risk_indicators=decision.risk_indicators[:10],
        )

    def _aggregate_tokens(self, solutions: List[Solution]) -> Dict[str, int]:
        prompt = sum(max(sol.tokens_prompt, 0) for sol in solutions)
        completion = sum(max(sol.tokens_completion, 0) for sol in solutions)
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def _build_report_markdown(
        self,
        request: InvestmentAnalysisRequest,
        recommendation: InvestmentRecommendation,
        summary: InvestmentSummary,
        outputs: List[AgentOutput],
        consensus: ConsensusResultView,
        risk_gate: RiskGateResult,
    ) -> str:
        lines = [
            f"# Investment Analysis: {request.asset.symbol}",
            "",
            f"- Mode: {request.mode.value}",
            f"- Market: {request.asset.market.value}",
            f"- Asset Type: {request.asset.asset_type.value}",
            f"- Horizon: {request.timeframe.horizon}",
            "",
            "## Recommendation",
            f"- Action: **{recommendation.action.value.upper()}**",
            f"- Confidence: {recommendation.confidence:.2f}",
            f"- Target Exposure: {recommendation.position_suggestion.get('target_exposure_pct', 0.0):.2%}",
            "",
            "## Thesis",
            summary.thesis,
            "",
            "## Key Drivers",
        ]
        lines.extend([f"- {d}" for d in summary.key_drivers] or ["- N/A"])
        lines.extend(["", "## Key Risks"])
        lines.extend([f"- {r}" for r in summary.key_risks] or ["- N/A"])

        lines.extend(["", "## Agent Outputs"])
        if outputs:
            for out in outputs:
                lines.append(f"- {out.agent_id}: {out.signal} (conf={out.confidence:.2f})")
        else:
            lines.append("- No agent outputs")

        lines.extend(["", "## Risk Gate", f"- Status: {risk_gate.status}", f"- Level: {risk_gate.risk_level}"])
        if risk_gate.risk_indicators:
            lines.extend([f"- Indicator: {x}" for x in risk_gate.risk_indicators])

        lines.extend(["", "## Roundtable Consensus"])
        lines.append(f"- Enabled: {consensus.enabled}")
        if consensus.enabled:
            lines.append(f"- Rounds Used: {consensus.rounds_used}")
            lines.append(f"- Reached: {consensus.consensus_reached}")

        return "\n".join(lines)

    async def _emit(self, event_sink: EventSink, event_type: str, **payload: Any) -> None:
        if event_sink is None:
            return
        await event_sink({"type": event_type, "payload": payload})
