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

_V1_ALLOWED_ASSET_TYPES = {AssetType.EQUITY, AssetType.ETF, AssetType.INDEX}
_V1_ALLOWED_MARKETS = {MarketCode.CN, MarketCode.HK, MarketCode.US}


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

        await self._emit(event_sink, "analysis_started", request_id=request_id, mode=request.mode.value)
        self._validate_request(request)
        await self._emit(event_sink, "request_validated", request_id=request_id)

        agents = self._select_agents(request.mode)
        await self._emit(
            event_sink,
            "agents_selected",
            request_id=request_id,
            agent_ids=[a.agent_id for a in agents],
        )

        task = self._build_task_prompt(request)
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

        recommendation = self._apply_constraints(request, recommendation)
        await self._emit(
            event_sink,
            "constraints_applied",
            request_id=request_id,
            action=recommendation.action.value,
            target_exposure_pct=recommendation.position_suggestion.get("target_exposure_pct", 0.0),
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
            data_sources=["public_market_data"],
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

        if market not in _V1_ALLOWED_MARKETS:
            raise ValueError(
                f"V1 only supports markets CN/HK/US, got {market.value}."
            )
        if asset_type not in _V1_ALLOWED_ASSET_TYPES:
            if asset_type in {AssetType.FUND, AssetType.CONVERTIBLE_BOND}:
                raise ValueError(
                    f"{asset_type.value} is planned for V2. "
                    "Current V1 supports only equity/etf/index."
                )
            raise ValueError(
                f"{asset_type.value} is planned for V3. "
                "Current V1 supports only equity/etf/index."
            )

    def _select_agents(self, mode: InvestmentMode) -> List[Agent]:
        all_agents = self.agent_registry.get_all_agents()
        if not all_agents:
            return []

        if mode == InvestmentMode.FAST:
            return all_agents[:1]
        if mode == InvestmentMode.AUTO:
            return all_agents[:2]
        if mode == InvestmentMode.COLLABORATE:
            return all_agents[: min(4, len(all_agents))]
        return all_agents[: min(5, len(all_agents))]

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

    def _build_task_prompt(self, request: InvestmentAnalysisRequest) -> str:
        facts = "\n".join(f"- {f}" for f in request.public_facts[:10])
        facts = facts or "- No extra public facts provided"

        return (
            f"You are an investment analyst. Analyze {request.asset.symbol} "
            f"({request.asset.market.value}, {request.asset.asset_type.value}) with horizon {request.timeframe.horizon}.\n"
            f"Risk profile: {request.risk_profile}. Objective: {request.objective}.\n"
            f"Market snapshot: {request.market_snapshot or 'N/A'}\n"
            f"Facts:\n{facts}\n"
            "Output concise recommendation with one of BUY/HOLD/SELL/WATCH and key reasons."
        )

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
    ) -> InvestmentRecommendation:
        constraints = request.constraints or {}
        action = recommendation.action
        position = dict(recommendation.position_suggestion)

        allowed_actions = constraints.get("allowed_actions")
        if isinstance(allowed_actions, list) and allowed_actions:
            allowed = {str(a).lower() for a in allowed_actions}
            if action.value not in allowed:
                action = RecommendationAction.HOLD

        if bool(constraints.get("no_short")) and action == RecommendationAction.SELL:
            action = RecommendationAction.HOLD

        target = float(position.get("target_exposure_pct", 0.0) or 0.0)

        profile_cap = self._profile_exposure_cap(request.risk_profile)
        objective_cap = self._objective_exposure_cap(request.objective)
        target = min(target, profile_cap, objective_cap)

        max_exposure = constraints.get("max_exposure_pct")
        if max_exposure is not None:
            cap = float(max_exposure)
            if cap < 0:
                cap = 0.0
            target = min(target, cap)

        if action == RecommendationAction.SELL:
            target = 0.0
        if action == RecommendationAction.HOLD:
            target = min(target, 0.05)

        position["target_exposure_pct"] = max(target, 0.0)

        max_dd = constraints.get("max_drawdown_guard_pct")
        if max_dd is not None:
            position["max_drawdown_guard_pct"] = max(float(max_dd), 0.0)

        return InvestmentRecommendation(
            action=action,
            confidence=recommendation.confidence,
            position_suggestion=position,
        )

    @staticmethod
    def _profile_exposure_cap(risk_profile: str) -> float:
        profile = (risk_profile or "balanced").lower()
        caps = {
            "conservative": 0.05,
            "balanced": 0.15,
            "aggressive": 0.30,
        }
        return caps.get(profile, 0.15)

    @staticmethod
    def _objective_exposure_cap(objective: str) -> float:
        goal = (objective or "balanced").lower()
        caps = {
            "defensive": 0.08,
            "income": 0.10,
            "balanced": 0.15,
            "alpha": 0.30,
        }
        return caps.get(goal, 0.15)

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
