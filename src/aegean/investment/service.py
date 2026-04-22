"""Core service for multi-mode investment analysis."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from aegean.core.agent import Agent, AgentRegistry
from aegean.core.coordinator import ConsensusCoordinator
from aegean.core.decision_engine import WeightedDecisionEngine
from aegean.core.models import ConsensusConfig, Solution, GroupKnowledgeInjection
from aegean.investment.models import (
    AgentOutput,
    AnalysisFramework,
    AssetType,
    CatalystItem,
    ConsensusResultView,
    ConsensusTrace,
    DisagreementSummary,
    DiscussionAgentEntry,
    DiscussionRound,
    ExternalEvidenceNode,
    ExternalNewsItem,
    InvestmentAnalysisRequest,
    InvestmentAnalysisResponse,
    InvestmentMetadata,
    InvestmentMode,
    InvestmentRecommendation,
    InvestmentSummary,
    MarketCode,
    PolicyOverrides,
    RecommendationAction,
    RiskGateResult,
    ScenarioItem,
)
from aegean.risk.models import RiskContext, RiskRequest, RiskSubject
from aegean.risk.risk_consensus import RiskConsensusCoordinator
from aegean.services.group_chat_service import GroupChatService
from aegean.investment.debate import (
    DebateContext,
    build_chair_prompt,
    build_research_manager_prompt,
    build_researcher_prompt,
    build_risk_debate_prompt,
    parse_confidence,
    parse_target_exposure_pct,
)
from aegean.investment.masters import (
    available_personas as _master_available_personas,
    build_master_prompt,
)
from aegean.investment.memory import RoleMemoryRegistry
from aegean.investment.providers import CoinGeckoProvider, ExaProvider, FMPProvider, FinnhubProvider, SerpAPIProvider, TavilyProvider, TushareProvider, YFinanceProvider
from aegean.investment.providers.gateway import InvestmentDataGateway
from aegean.investment.risk import PortfolioRiskEngine, PortfolioRiskResult
from aegean.investment.sentiment import (
    SentimentPipeline,
    finnhub_insider_to_trades,
    news_items_to_articles,
    tushare_insider_to_trades,
)


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

_SIGNAL_TO_STANCE = {
    "bullish": "support",
    "bearish": "oppose",
    "neutral": "review",
}

_ROLE_BY_TASK_TYPE = {
    "equity_analysis": "fundamental_specialist",
    "etf_analysis": "allocation_specialist",
    "index_analysis": "macro_specialist",
    "fund_analysis": "fund_specialist",
    "convertible_bond_analysis": "convertible_bond_specialist",
    "futures_analysis": "futures_specialist",
    "options_analysis": "options_specialist",
    "crypto_analysis": "crypto_specialist",
}

_EQUITY_PANEL_ROLES = [
    "fundamental_specialist",
    "valuation_specialist",
    "macro_specialist",
    "risk_specialist",
]

_ROLE_DATA_FOCUS = {
    "fundamental_specialist": ["fundamentals", "market"],
    "valuation_specialist": ["fundamentals", "market"],
    "allocation_specialist": ["market"],
    "macro_specialist": ["market", "news"],
    "fund_specialist": ["fundamentals", "market"],
    "convertible_bond_specialist": ["fundamentals", "market"],
    "futures_specialist": ["market", "news"],
    "options_specialist": ["market"],
    "crypto_specialist": ["market", "news"],
    "risk_specialist": ["news", "market"],
    "investment_specialist": ["market", "fundamentals", "news"],
}

_ROLE_SKILL_FOCUS = {
    "fundamental_specialist": ["fundamental_analysis"],
    "valuation_specialist": ["equity_valuation"],
    "macro_specialist": ["macro_regime"],
    "risk_specialist": ["general_investment_analysis"],
}

_TITLE_BY_ROLE = {
    "fundamental_specialist": "Fundamental Analysis Agent",
    "valuation_specialist": "Equity Valuation Agent",
    "allocation_specialist": "Allocation Analysis Agent",
    "macro_specialist": "Macro Analysis Agent",
    "fund_specialist": "Fund Selection Agent",
    "convertible_bond_specialist": "Convertible Bond Agent",
    "futures_specialist": "Futures Analysis Agent",
    "options_specialist": "Options Analysis Agent",
    "crypto_specialist": "Crypto Analysis Agent",
    "risk_specialist": "Risk Review Agent",
    "bull_researcher": "Bull Researcher",
    "bear_researcher": "Bear Researcher",
    "research_manager": "Research Manager",
    "risk_aggressive": "Aggressive Risk Analyst",
    "risk_neutral": "Neutral Risk Analyst",
    "risk_conservative": "Conservative Risk Analyst",
    "chair": "Chair / Portfolio Manager",
}

_DEBATE_STANCES = ("bull", "bear")
_RISK_DEBATE_STANCES = ("aggressive", "neutral", "conservative")
_DEFAULT_MASTER_PERSONAS = ("buffett", "burry", "wood")

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

_SKILL_PROFILES = {
    "fundamental_analysis": {
        "name": "Fundamental Analysis",
        "description": "Review business quality, earnings power, and operating momentum.",
        "categories": ["investment_methodology", "equity_research"],
        "required_data_sources": ["public_market_data", "company_fundamentals"],
    },
    "equity_valuation": {
        "name": "Equity Valuation",
        "description": "Assess valuation ranges, multiple compression risk, and upside/downside asymmetry.",
        "categories": ["investment_methodology", "valuation"],
        "required_data_sources": ["public_market_data", "company_fundamentals"],
    },
    "index_tracking": {
        "name": "Index Tracking",
        "description": "Evaluate benchmark composition, tracking behavior, and replication quality.",
        "categories": ["etf_research", "portfolio_construction"],
        "required_data_sources": ["public_market_data", "etf_holdings"],
    },
    "allocation_analysis": {
        "name": "Allocation Analysis",
        "description": "Judge fit with portfolio objectives, diversification impact, and exposure budgets.",
        "categories": ["portfolio_construction"],
        "required_data_sources": ["public_market_data", "etf_holdings"],
    },
    "macro_regime": {
        "name": "Macro Regime",
        "description": "Analyze macro cycle, policy backdrop, and index-level regime sensitivity.",
        "categories": ["macro", "index_research"],
        "required_data_sources": ["public_market_data", "macro_indicators"],
    },
    "index_structure": {
        "name": "Index Structure",
        "description": "Review concentration, factor tilt, and structural exposures of the index.",
        "categories": ["index_research"],
        "required_data_sources": ["public_market_data", "macro_indicators"],
    },
    "fund_selection": {
        "name": "Fund Selection",
        "description": "Evaluate strategy persistence, fit-for-purpose, and product quality.",
        "categories": ["fund_research"],
        "required_data_sources": ["public_market_data", "fund_nav_feed"],
    },
    "manager_style": {
        "name": "Manager Style",
        "description": "Assess manager behavior, style drift, and repeatability of returns.",
        "categories": ["fund_research"],
        "required_data_sources": ["fund_nav_feed"],
    },
    "credit_analysis": {
        "name": "Credit Analysis",
        "description": "Evaluate balance-sheet risk, default sensitivity, and credit spread resilience.",
        "categories": ["credit", "bond_research"],
        "required_data_sources": ["public_market_data", "bond_termsheet_feed"],
    },
    "equity_optional_value": {
        "name": "Equity Optional Value",
        "description": "Analyze embedded equity optionality in convertible instruments.",
        "categories": ["convertible_bond", "valuation"],
        "required_data_sources": ["public_market_data", "bond_termsheet_feed"],
    },
    "futures_basis": {
        "name": "Futures Basis",
        "description": "Review basis structure, carry signal, and term curve behavior.",
        "categories": ["derivatives", "futures"],
        "required_data_sources": ["public_market_data", "futures_curve_feed"],
    },
    "margin_risk": {
        "name": "Margin Risk",
        "description": "Assess leverage, margin sensitivity, and liquidation risk.",
        "categories": ["derivatives", "risk_controls"],
        "required_data_sources": ["margin_rules", "futures_curve_feed"],
    },
    "options_greeks": {
        "name": "Options Greeks",
        "description": "Analyze delta, gamma, theta, and vega exposure around the trade idea.",
        "categories": ["derivatives", "options"],
        "required_data_sources": ["public_market_data", "option_chain_feed"],
    },
    "vol_surface": {
        "name": "Vol Surface",
        "description": "Assess implied volatility term structure and skew for option positioning.",
        "categories": ["derivatives", "options"],
        "required_data_sources": ["vol_surface_feed", "option_chain_feed"],
    },
    "onchain_signal": {
        "name": "On-chain Signal",
        "description": "Evaluate wallet activity, network usage, and structural on-chain demand signals.",
        "categories": ["crypto", "onchain_research"],
        "required_data_sources": ["public_market_data", "onchain_metrics"],
    },
    "liquidity_microstructure": {
        "name": "Liquidity Microstructure",
        "description": "Review exchange depth, spread quality, and execution fragility.",
        "categories": ["crypto", "market_microstructure"],
        "required_data_sources": ["exchange_depth_feed", "public_market_data"],
    },
    "general_investment_analysis": {
        "name": "General Investment Analysis",
        "description": "Fallback generalist investment reasoning skill.",
        "categories": ["investment_methodology"],
        "required_data_sources": ["public_market_data"],
    },
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
        group_service: Optional[GroupChatService] = None,
        role_memory: Optional[RoleMemoryRegistry] = None,
        portfolio_risk_engine: Optional[PortfolioRiskEngine] = None,
        sentiment_pipeline: Optional[SentimentPipeline] = None,
    ):
        self.agent_registry = agent_registry
        self.memory_system = memory_system
        self.llm_client = llm_client
        self.risk_coordinator = risk_coordinator
        self.group_service = group_service
        self.role_memory = role_memory or RoleMemoryRegistry()
        self.portfolio_risk_engine = portfolio_risk_engine or PortfolioRiskEngine()
        self.sentiment_pipeline = sentiment_pipeline or SentimentPipeline()
        self.market_data_provider = YFinanceProvider()
        self.cn_market_data_provider = TushareProvider()
        self.crypto_market_data_provider = CoinGeckoProvider()
        self.fundamental_data_provider = FMPProvider()
        self.cn_fundamental_data_provider = TushareProvider()
        self.news_data_provider = FinnhubProvider()
        self.search_data_provider = TavilyProvider()
        self.search_fallback_provider = ExaProvider()
        self.data_gateway = InvestmentDataGateway(
            {
                "market": self.market_data_provider,
                "cn_market": self.cn_market_data_provider,
                "crypto_market": self.crypto_market_data_provider,
                "fundamentals": self.fundamental_data_provider,
                "cn_fundamentals": self.cn_fundamental_data_provider,
                "news": self.news_data_provider,
                "search": self.search_data_provider,
                "search_fallback": self.search_fallback_provider,
                "search_fallback_2": SerpAPIProvider(),
            }
        )
        self.analysis_runs: Dict[str, Dict[str, Any]] = {}

    async def analyze(
        self,
        request: InvestmentAnalysisRequest,
        event_sink: EventSink = None,
        request_id: Optional[str] = None,
    ) -> InvestmentAnalysisResponse:
        started = time.time()
        created_at = _utc_now()
        event_count = 0
        timeline: List[Dict[str, Any]] = []
        agents_panel: List[Dict[str, Any]] = []
        request_id = request_id or f"inv-{uuid.uuid4().hex[:12]}"
        self.analysis_runs[request_id] = {
            "status": "running",
            "request": request.model_dump(mode="json"),
            "timeline": timeline,
            "agents": agents_panel,
            "result": None,
            "discussion": {"enabled": False, "final_summary": "", "rounds": []},
            "policy_overrides": {},
            "risk_gate": {},
        }

        async def emit(event_type: str, **payload: Any) -> None:
            nonlocal event_count
            event_count += 1
            event = {
                "type": event_type,
                "timestamp": _utc_now().isoformat(),
                "request_id": request_id,
                "payload": payload,
            }
            timeline.append(event)
            self._update_analysis_run(event, agents_panel)
            await self._emit(
                event_sink,
                event_type,
                request_id=request_id,
                event_timestamp=event["timestamp"],
                **payload,
            )

        task_type = self._task_type_for_asset(request.asset.asset_type)
        selected_skills = self._selected_skills_for_asset(request.asset.asset_type)
        resolved_skill_profiles = self._resolve_skill_profiles(selected_skills)
        data_sources = self._asset_data_sources_for(request.asset.asset_type)
        normalized_market_data = await self._fetch_normalized_asset_data(request)
        group_injection = await self._build_group_knowledge_injection(request, task_type)

        await emit(
            "analysis_started",
            mode=request.mode.value,
            group_id=(group_injection.group_id if group_injection else None),
        )
        self._validate_request(request)
        await emit("request_validated")

        agents = self._select_agents(request.mode, task_type)
        panel_roles = self._panel_roles_for_task(task_type)
        panel_role_skills = {
            role: self._skills_for_role(role, selected_skills)
            for role in panel_roles
        }
        panel_role_data_focus = {
            role: list(_ROLE_DATA_FOCUS.get(role, ["market", "fundamentals", "news"]))
            for role in panel_roles
        }
        analysis_framework = self._build_analysis_framework(
            request.mode,
            task_type,
            selected_skills,
            data_sources,
            resolved_skill_profiles,
            group_injection,
            panel_roles,
            panel_role_skills,
            panel_role_data_focus,
        )
        await emit(
            "agents_selected",
            agent_ids=[a.agent_id for a in agents],
            task_type=task_type,
            selected_skills=selected_skills,
            resolved_skill_profiles=resolved_skill_profiles,
            group_memory_attached=bool(group_injection and group_injection.memory_context),
        )

        await emit(
            "normalized_data_ready",
            providers=normalized_market_data.get("providers", {}),
            provider_status=normalized_market_data.get("provider_status", {}),
            provider_signals=normalized_market_data.get("provider_signals", []),
            has_news=bool(normalized_market_data.get("news")),
        )

        task = self._build_task_prompt(request, task_type, selected_skills, normalized_market_data, group_injection)
        solutions = await self._collect_solutions(
            agents,
            request,
            task_type,
            selected_skills,
            normalized_market_data,
            group_injection,
            emit,
        )
        panel_roles = self._panel_roles_for_task(task_type)
        agent_outputs = [
            self._solution_to_output(
                sol,
                task_type,
                role=(panel_roles[index] if index < len(panel_roles) else None),
            )
            for index, sol in enumerate(solutions)
        ]

        recommendation, summary = self._merge_outputs(agent_outputs)
        external_evidence = self._build_external_evidence(normalized_market_data)
        bull_case, bear_case = self._build_bull_bear_cases(summary, agent_outputs, external_evidence)
        catalysts = self._build_catalysts(request, external_evidence)
        scenarios = self._build_scenarios(recommendation)
        disagreement_summary = self._build_disagreement_summary(summary, agent_outputs)

        consensus_view = ConsensusResultView(enabled=False)
        consensus_trace = ConsensusTrace(discussion_enabled=False)
        if request.mode == InvestmentMode.ROUNDTABLE and agents:
            await emit("roundtable_started")
            consensus_view, recommendation, consensus_trace = await self._run_roundtable(
                task,
                agents,
                recommendation,
                agent_outputs,
                emit,
            )
            await emit(
                "roundtable_finished",
                rounds_used=consensus_view.rounds_used,
                consensus_reached=consensus_view.consensus_reached,
            )
        elif request.mode == InvestmentMode.DEBATE and agents:
            await emit("debate_started")
            consensus_view, recommendation, consensus_trace = await self._run_debate(
                request,
                task_type,
                agent_outputs,
                recommendation,
                normalized_market_data,
                group_injection,
                emit,
            )
            await emit(
                "debate_finished",
                rounds_used=consensus_view.rounds_used,
                consensus_reached=consensus_view.consensus_reached,
            )

        recommendation, constraints_summary = self._apply_constraints(request, recommendation)
        recommendation, constraints_summary, portfolio_risk_payload = self._apply_portfolio_risk(
            request,
            normalized_market_data,
            recommendation,
            constraints_summary,
        )
        if portfolio_risk_payload:
            await emit("portfolio_risk_applied", **portfolio_risk_payload)
        policy_overrides = self._build_policy_overrides(constraints_summary)
        await emit(
            "constraints_applied",
            action=recommendation.action.value,
            target_exposure_pct=recommendation.position_suggestion.get("target_exposure_pct", 0.0),
            constraints_applied_summary=constraints_summary,
        )
        await emit(
            "recommendation_ready",
            action=recommendation.action.value,
            confidence=recommendation.confidence,
        )

        await emit("risk_gate_started")
        risk_gate = await self._evaluate_risk(request, recommendation, group_injection)
        await emit(
            "risk_gate_finished",
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
            external_evidence,
        )
        completed_at = _utc_now()
        metadata = InvestmentMetadata(
            token_usage=self._aggregate_tokens(solutions),
            latency_ms=int((time.time() - started) * 1000),
            data_sources=data_sources,
            selected_skills=selected_skills,
            resolved_skill_profiles=resolved_skill_profiles,
            panel_roles=panel_roles,
            panel_role_skills=panel_role_skills,
            panel_role_data_focus=panel_role_data_focus,
            task_type=task_type,
            constraints_applied_summary=constraints_summary,
            provider_status=normalized_market_data.get("provider_status", {}),
            provider_signals=normalized_market_data.get("provider_signals", []),
            created_at=created_at,
            completed_at=completed_at,
            event_count=event_count,
            schema_version="investment_analysis.v2",
            debug_flags={
                "used_roundtable": consensus_view.enabled,
                "used_constraints": bool(request.constraints or request.portfolio_context),
                "used_risk_gate": True,
                "used_external_news": bool(normalized_market_data.get("news")),
                "used_group_knowledge": bool(group_injection),
            },
        )

        response = InvestmentAnalysisResponse(
            request_id=request_id,
            status="completed",
            mode=request.mode,
            asset=request.asset,
            timeframe=request.timeframe,
            analysis_framework=analysis_framework,
            recommendation=recommendation,
            summary=summary,
            bull_case=bull_case,
            bear_case=bear_case,
            catalysts=catalysts,
            scenario_analysis=scenarios,
            agent_outputs=agent_outputs,
            disagreement_summary=disagreement_summary,
            risk_gate=risk_gate,
            consensus=consensus_view,
            consensus_trace=consensus_trace,
            policy_overrides=policy_overrides,
            report_markdown=report_md,
            metadata=metadata,
        )
        self.analysis_runs[request_id]["status"] = "completed"
        self.analysis_runs[request_id]["result"] = response.model_dump(mode="json")
        self.analysis_runs[request_id]["discussion"] = consensus_trace.model_dump(mode="json")
        self.analysis_runs[request_id]["policy_overrides"] = policy_overrides.model_dump(mode="json")
        self.analysis_runs[request_id]["risk_gate"] = risk_gate.model_dump(mode="json")
        await emit("analysis_completed")
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

        panel_roles = self._panel_roles_for_task(task_type)
        role_specific_agents: List[Agent] = []
        used_agent_ids = set()

        for role in panel_roles:
            ranked_for_role = sorted(
                all_agents,
                key=lambda a: a.get_weight_for_task(role),
                reverse=True,
            )
            for candidate in ranked_for_role:
                if candidate.agent_id in used_agent_ids:
                    continue
                role_specific_agents.append(candidate)
                used_agent_ids.add(candidate.agent_id)
                break

        ranked = sorted(
            all_agents,
            key=lambda a: a.get_weight_for_task(task_type),
            reverse=True,
        )
        ordered_agents = role_specific_agents + [
            agent for agent in ranked if agent.agent_id not in used_agent_ids
        ]

        if mode == InvestmentMode.FAST:
            return ordered_agents[:1]
        if mode == InvestmentMode.AUTO:
            return ordered_agents[: min(max(2, len(panel_roles)), len(ordered_agents))]
        if mode == InvestmentMode.COLLABORATE:
            return ordered_agents[: min(max(4, len(panel_roles)), len(ordered_agents))]
        return ordered_agents[: min(max(5, len(panel_roles)), len(ordered_agents))]

    async def _collect_solutions(
        self,
        agents: List[Agent],
        request: InvestmentAnalysisRequest,
        task_type: str,
        selected_skills: List[str],
        normalized_market_data: Dict[str, Any],
        group_injection: Optional[GroupKnowledgeInjection],
        emit: Callable[..., Awaitable[None]],
    ) -> List[Solution]:
        if not agents:
            return []

        async def _run(agent: Agent, agent_task: str) -> Tuple[Agent, Any]:
            try:
                result = await agent.generate_solution(agent_task)
                return agent, result
            except Exception as exc:
                return agent, exc

        panel_roles = self._panel_roles_for_task(task_type)
        group_id = group_injection.group_id if group_injection else None
        situation_text = self._build_situation_text(request, normalized_market_data)
        agent_context: Dict[str, Dict[str, str]] = {}
        tasks = []
        for index, agent in enumerate(agents):
            agent_role = panel_roles[index] if index < len(panel_roles) else self._role_for_task_type(task_type)
            role_skills = self._skills_for_role(agent_role, selected_skills)
            recalled = self.role_memory.recall(
                agent_role, situation_text, group_id=group_id, n_matches=2
            )
            agent_context[agent.agent_id] = {"role": agent_role}
            tasks.append(
                asyncio.create_task(
                    _run(
                        agent,
                        self._build_task_prompt(
                            request,
                            task_type,
                            role_skills,
                            normalized_market_data,
                            group_injection,
                            agent_role=agent_role,
                            agent_id=agent.agent_id,
                            role_memory_hits=recalled,
                        ),
                    )
                )
            )

        solutions: List[Solution] = []

        for index, agent in enumerate(agents):
            agent_role = panel_roles[index] if index < len(panel_roles) else self._role_for_task_type(task_type)
            await emit(
                "agent_started",
                agent_id=agent.agent_id,
                role=agent_role,
            )

        for done_task in asyncio.as_completed(tasks):
            agent, result = await done_task
            if isinstance(result, Exception):
                solution = Solution(
                    agent_id=agent.agent_id,
                    answer=f"Agent failed: {result}",
                    confidence=0.2,
                )
                await emit(
                    "agent_failed",
                    agent_id=agent.agent_id,
                    message=str(result),
                )
            else:
                solution = result
            solutions.append(solution)
            signal = self._extract_signal(solution.answer)
            agent_role = agent_context.get(agent.agent_id, {}).get("role") or self._role_for_task_type(task_type)
            if not isinstance(result, Exception):
                self._record_role_experience(
                    role=agent_role,
                    situation=situation_text,
                    solution=solution,
                    group_id=group_id,
                    request=request,
                    signal=signal,
                )
            await emit(
                "agent_completed",
                agent_id=solution.agent_id,
                confidence=solution.confidence,
                signal=signal,
                summary=self._summary_from_answer(solution.answer),
            )

        return solutions

    def _build_situation_text(
        self,
        request: InvestmentAnalysisRequest,
        normalized_market_data: Dict[str, Any],
    ) -> str:
        parts: List[str] = [
            f"symbol={request.asset.symbol}",
            f"market={request.asset.market.value}",
            f"asset_type={request.asset.asset_type.value}",
            f"horizon={request.timeframe.horizon}",
            f"risk_profile={request.risk_profile}",
            f"objective={request.objective}",
        ]
        if request.market_snapshot:
            parts.append(f"snapshot={request.market_snapshot}")
        market = normalized_market_data.get("market") or {}
        for key in ("price", "change_pct", "pe_ratio", "pb_ratio", "market_cap", "week52_high", "week52_low"):
            if key in market and market[key] is not None:
                parts.append(f"{key}={market[key]}")
        fundamentals = normalized_market_data.get("fundamentals") or {}
        for key in ("revenue_growth", "net_margin", "roe", "debt_to_equity"):
            if key in fundamentals and fundamentals[key] is not None:
                parts.append(f"{key}={fundamentals[key]}")
        news = normalized_market_data.get("news") or []
        polarities = [
            item.get("polarity") for item in news[:5]
            if isinstance(item, dict) and item.get("polarity")
        ]
        if polarities:
            parts.append("news_polarity=" + ",".join(polarities))
        return " ".join(str(p) for p in parts)

    def _record_role_experience(
        self,
        role: str,
        situation: str,
        solution: Solution,
        group_id: Optional[str],
        request: InvestmentAnalysisRequest,
        signal: str,
    ) -> None:
        answer = (solution.answer or "").strip()
        if not answer:
            return
        metadata = {
            "agent_id": solution.agent_id,
            "confidence": solution.confidence,
            "signal": signal,
            "symbol": request.asset.symbol,
            "market": request.asset.market.value,
            "asset_type": request.asset.asset_type.value,
            "horizon": request.timeframe.horizon,
        }
        try:
            self.role_memory.record(
                role=role,
                situation=situation,
                recommendation=answer,
                group_id=group_id,
                metadata=metadata,
                created_at=_utc_now().isoformat(),
            )
        except Exception:
            # Memory recording should never break analysis.
            pass

    def _build_task_prompt(
        self,
        request: InvestmentAnalysisRequest,
        task_type: str,
        selected_skills: List[str],
        normalized_market_data: Dict[str, Any],
        group_injection: Optional[GroupKnowledgeInjection] = None,
        agent_role: Optional[str] = None,
        agent_id: Optional[str] = None,
        role_memory_hits: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        facts = "\n".join(f"- {f}" for f in request.public_facts[:10])
        facts = facts or "- No extra public facts provided"
        skill_lines = "\n".join(f"- {s}" for s in selected_skills) or "- generic_analysis"
        effective_role = agent_role or self._role_for_task_type(task_type)
        role_title = _TITLE_BY_ROLE.get(effective_role, "Investment Analysis Agent")
        normalized_context = self._format_normalized_data_for_prompt(
            normalized_market_data,
            effective_role,
        )

        group_context = ""
        if group_injection:
            extras = []
            if group_injection.memory_context:
                extras.append(group_injection.memory_context)
            if group_injection.skill_descriptions:
                extras.append("Group skill registry:\n" + "\n".join(f"- {item}" for item in group_injection.skill_descriptions[:8]))
            if group_injection.document_summaries:
                extras.append("Group shared documents:\n" + "\n".join(f"- {item}" for item in group_injection.document_summaries[:6]))
            if group_injection.historical_case_ids:
                extras.append("Group historical cases:\n" + "\n".join(f"- {item}" for item in group_injection.historical_case_ids[:6]))
            if group_injection.graph_ids:
                extras.append("Relevant knowledge graphs:\n" + "\n".join(f"- {item}" for item in group_injection.graph_ids[:5]))
            if extras:
                group_context = "\n\nGroup shared knowledge context:\n" + "\n\n".join(extras)

        provider_status = normalized_market_data.get("provider_status") or {}
        provider_signals = self._collect_provider_signals(provider_status)

        role_memory_block = self._format_role_memory_for_prompt(role_memory_hits)
        document_block = self._format_document_context_for_prompt(
            request.metadata.get("document_chunks") if request.metadata else None
        )
        sentiment_block = self._format_sentiment_for_prompt(
            normalized_market_data.get("sentiment") if isinstance(normalized_market_data, dict) else None
        )

        return (
            f"You are {role_title}. Agent ID: {agent_id or 'shared_agent'}. Analyze {request.asset.symbol} "
            f"({request.asset.market.value}, {request.asset.asset_type.value}) with horizon {request.timeframe.horizon}.\n"
            f"Task type: {task_type}.\n"
            f"Agent focus: {effective_role}.\n"
            f"Risk profile: {request.risk_profile}. Objective: {request.objective}.\n"
            f"Required skills:\n{skill_lines}\n"
            f"Market snapshot: {request.market_snapshot or 'N/A'}\n"
            f"Facts:\n{facts}\n"
            f"Provider status summary:\n{self._format_provider_status_for_prompt(provider_status)}\n"
            f"Provider signals: {', '.join(provider_signals) if provider_signals else 'none'}\n"
            f"Normalized external data:\n{normalized_context}"
            f"{sentiment_block}"
            f"{document_block}"
            f"{group_context}"
            f"{role_memory_block}\n"
            "Prioritize the data most relevant to your role. If key provider data is unavailable or degraded, explicitly mention the data gap and lower confidence accordingly. Output concise recommendation with one of BUY/HOLD/SELL/WATCH and key reasons."
        )

    @staticmethod
    def _format_document_context_for_prompt(
        chunks: Optional[List[Dict[str, Any]]],
        max_chunks: int = 6,
        max_chars_per_chunk: int = 600,
    ) -> str:
        """Render pre-ingested document chunks (10-K, PDFs, memos) into prompt text.

        ``chunks`` is the caller-provided list under
        ``request.metadata['document_chunks']`` — each dict should have
        at least ``text`` and optionally ``section`` / ``source``.
        """
        if not chunks:
            return ""
        lines: List[str] = []
        for idx, chunk in enumerate(chunks[:max_chunks], start=1):
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "").strip().replace("\r", "")
            if not text:
                continue
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk].rstrip() + "..."
            section = str(chunk.get("section") or "").strip()
            source = str(chunk.get("source") or "").strip()
            header_bits = [f"#{idx}"]
            if section:
                header_bits.append(f"section={section}")
            if source:
                header_bits.append(f"source={source}")
            lines.append(f"[{' '.join(header_bits)}]\n{text}")
        if not lines:
            return ""
        return (
            "\n\nIngested document excerpts (10-K / filings / uploaded memos):\n"
            + "\n\n".join(lines)
        )

    @staticmethod
    def _format_sentiment_for_prompt(sentiment: Optional[Dict[str, Any]]) -> str:
        if not sentiment:
            return ""
        insider = sentiment.get("insider_trading", {}) or {}
        news = sentiment.get("news_sentiment", {}) or {}
        insider_metrics = insider.get("metrics", {}) or {}
        news_metrics = news.get("metrics", {}) or {}
        return (
            "\n\nSentiment (insider weight=0.3, news weight=0.7):\n"
            f"- overall: {sentiment.get('signal', 'neutral')} "
            f"(confidence={sentiment.get('confidence', 0.0)})\n"
            f"- insider: {insider.get('signal', 'neutral')}, "
            f"bullish={insider_metrics.get('bullish', 0)}, bearish={insider_metrics.get('bearish', 0)}, "
            f"total={insider_metrics.get('total', 0)}\n"
            f"- news: {news.get('signal', 'neutral')}, "
            f"bullish={news_metrics.get('bullish', 0)}, bearish={news_metrics.get('bearish', 0)}, "
            f"total={news_metrics.get('total', 0)}"
        )

    @staticmethod
    def _format_role_memory_for_prompt(
        hits: Optional[List[Dict[str, Any]]],
    ) -> str:
        if not hits:
            return ""
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            rec = (hit.get("recommendation") or "").strip().replace("\n", " ")
            if len(rec) > 400:
                rec = rec[:400].rstrip() + "..."
            score = hit.get("similarity_score") or 0.0
            outcome = hit.get("outcome") or {}
            outcome_note = ""
            if outcome:
                action = outcome.get("action") or outcome.get("actual_signal")
                hit_flag = outcome.get("hit")
                return_pct = outcome.get("return_pct")
                parts: List[str] = []
                if action:
                    parts.append(f"actual={action}")
                if hit_flag is not None:
                    parts.append(f"hit={hit_flag}")
                if return_pct is not None:
                    parts.append(f"return_pct={return_pct}")
                if parts:
                    outcome_note = f" [outcome: {', '.join(parts)}]"
            lines.append(f"- case #{idx} (sim={score:.2f}){outcome_note}: {rec}")
        return (
            "\n\nPast role experiences (same role, similar situations). "
            "Treat as precedent, not as ground truth — cite only if the current setup truly matches:\n"
            + "\n".join(lines)
        )

    async def _fetch_normalized_asset_data(
        self,
        request: InvestmentAnalysisRequest,
    ) -> Dict[str, Any]:
        normalized = await self.data_gateway.fetch_all(
            request.asset.symbol,
            request.asset.market.value,
            request.asset.asset_type.value,
        )
        await self._augment_with_sentiment(request, normalized)
        return normalized

    async def _augment_with_sentiment(
        self,
        request: InvestmentAnalysisRequest,
        normalized: Dict[str, Any],
    ) -> None:
        """Run the three-tier sentiment pipeline and attach results.

        Writes into ``normalized['sentiment']`` and pushes a synthetic
        provider-status entry under key ``sentiment`` so the existing
        provider-status / provider-signals flow picks it up automatically.
        Any failure is swallowed — sentiment is additive.
        """
        try:
            news_items = normalized.get("news") or []
            articles = news_items_to_articles(news_items)
            insider_trades = []
            insider_status: Dict[str, Any] = {"status": "skipped", "signals": []}
            if request.asset.market == MarketCode.US and isinstance(
                self.news_data_provider, FinnhubProvider
            ):
                insider_payload = await self.news_data_provider.fetch_insider_transactions(
                    request.asset.symbol
                )
                insider_status = {
                    "status": insider_payload.get("status", "unknown"),
                    "signals": list(insider_payload.get("signals") or []),
                    "message": insider_payload.get("message", ""),
                }
                insider_trades = finnhub_insider_to_trades(insider_payload.get("data") or [])
            elif request.asset.market == MarketCode.CN and isinstance(
                self.cn_market_data_provider, TushareProvider
            ):
                insider_payload = await self.cn_market_data_provider.fetch_insider_transactions(
                    request.asset.symbol
                )
                insider_status = {
                    "status": insider_payload.get("status", "unknown"),
                    "signals": list(insider_payload.get("signals") or []),
                    "message": insider_payload.get("message", ""),
                }
                insider_trades = tushare_insider_to_trades(insider_payload.get("data") or [])
            if not articles and not insider_trades:
                return
            result = self.sentiment_pipeline.assess(insider_trades, articles)
            normalized["sentiment"] = result.to_dict()
            signal_token = f"SENTIMENT_{result.signal.upper()}"
            provider_status = normalized.setdefault("provider_status", {})
            provider_status["sentiment"] = {
                "status": "ok",
                "signals": [signal_token],
                "message": (
                    f"sentiment={result.signal} conf={result.confidence:.2f} "
                    f"insider={result.insider.total} news={result.news.total}"
                ),
                "insider_status": insider_status,
            }
            signals = list(normalized.get("provider_signals") or [])
            if signal_token not in signals:
                signals.append(signal_token)
            normalized["provider_signals"] = signals
        except Exception:
            # Never let sentiment break the main pipeline.
            return

    @staticmethod
    def _format_provider_status_for_prompt(provider_status: Dict[str, Any]) -> str:
        if not provider_status:
            return "- No provider status available"
        lines = []
        for provider, status in provider_status.items():
            if not isinstance(status, dict):
                lines.append(f"- {provider}: unknown")
                continue
            lines.append(
                f"- {provider}: status={status.get('status', 'unknown')}; "
                f"signals={', '.join(status.get('signals', [])) or 'none'}; "
                f"message={status.get('message', '') or 'ok'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _collect_provider_signals(provider_status: Dict[str, Any]) -> List[str]:
        signals: List[str] = []
        for status in provider_status.values():
            if isinstance(status, dict):
                signals.extend(str(item) for item in status.get("signals", []) if str(item).strip())
        return sorted(set(signals))

    @staticmethod
    def _format_normalized_data_for_prompt(
        normalized_market_data: Dict[str, Any],
        role: Optional[str] = None,
    ) -> str:
        sections = []
        market = normalized_market_data.get("market") or {}
        fundamentals = normalized_market_data.get("fundamentals") or {}
        news = normalized_market_data.get("news") or []
        focus = _ROLE_DATA_FOCUS.get(role or "investment_specialist", ["market", "fundamentals", "news"])

        if "market" in focus and market:
            sections.append("Market data:\n" + "\n".join(f"- {key}: {value}" for key, value in market.items()))
        if "fundamentals" in focus and fundamentals:
            sections.append(
                "Fundamental data:\n" + "\n".join(f"- {key}: {value}" for key, value in fundamentals.items())
            )
        if "news" in focus and news:
            sections.append(
                "News:\n"
                + "\n".join(
                    f"- {item.get('title')} | source={item.get('source', '')} | polarity={item.get('polarity', 'neutral')}"
                    if isinstance(item, dict)
                    else f"- {item}"
                    for item in news[:5]
                )
            )
        if not sections:
            return "- No normalized external data available"
        return "\n\n".join(sections)

    async def _build_group_knowledge_injection(
        self,
        request: InvestmentAnalysisRequest,
        task_type: str,
    ) -> Optional[GroupKnowledgeInjection]:
        if not self.group_service:
            return None

        group_id = (
            request.metadata.get("group_id")
            or request.metadata.get("knowledge_group_id")
        )
        if not group_id:
            return None

        query = request.custom_question or (
            f"{request.asset.symbol} {request.asset.asset_type.value} {task_type} {request.timeframe.horizon}"
        )
        categories = sorted(
            {
                category
                for profile in self._resolve_skill_profiles(self._selected_skills_for_asset(request.asset.asset_type))
                for category in profile.get("categories", [])
            }
        )
        return await self.group_service.build_group_knowledge_injection(
            group_id=group_id,
            query=query,
            categories=categories,
        )

    @staticmethod
    def _task_type_for_asset(asset_type: AssetType) -> str:
        return _ASSET_TASK_TYPE.get(asset_type, "investment_analysis")

    @staticmethod
    def _selected_skills_for_asset(asset_type: AssetType) -> List[str]:
        return _ASSET_SKILLS.get(asset_type, ["general_investment_analysis"])

    @staticmethod
    def _resolve_skill_profiles(selected_skills: List[str]) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []
        for skill_id in selected_skills:
            profile = _SKILL_PROFILES.get(skill_id, {})
            profiles.append(
                {
                    "skill_id": skill_id,
                    "name": profile.get("name", skill_id),
                    "description": profile.get("description", ""),
                    "categories": profile.get("categories", []),
                    "required_data_sources": profile.get("required_data_sources", []),
                }
            )
        return profiles

    @staticmethod
    def _asset_data_sources_for(asset_type: AssetType) -> List[str]:
        return _ASSET_DATA_SOURCES.get(asset_type, ["public_market_data"])

    @staticmethod
    def _role_for_task_type(task_type: str) -> str:
        return _ROLE_BY_TASK_TYPE.get(task_type, "investment_specialist")

    @staticmethod
    def _panel_roles_for_task(task_type: str) -> List[str]:
        if task_type == "equity_analysis":
            return list(_EQUITY_PANEL_ROLES)
        return [InvestmentAnalysisService._role_for_task_type(task_type)]

    @staticmethod
    def _skills_for_role(role: str, selected_skills: List[str]) -> List[str]:
        focused = _ROLE_SKILL_FOCUS.get(role)
        if not focused:
            return selected_skills
        matched = [skill for skill in selected_skills if skill in focused]
        return matched or selected_skills

    def _solution_to_output(
        self,
        solution: Solution,
        task_type: str,
        role: Optional[str] = None,
    ) -> AgentOutput:
        signal = self._extract_signal(solution.answer)
        evidence = [ln.strip() for ln in solution.answer.split("\n") if ln.strip()][:3]
        effective_role = role or self._role_for_task_type(task_type)
        return AgentOutput(
            agent_id=solution.agent_id,
            role=effective_role,
            title=_TITLE_BY_ROLE.get(effective_role, "Investment Analysis Agent"),
            signal=signal,
            stance=_SIGNAL_TO_STANCE.get(signal, "review"),
            confidence=max(0.0, min(solution.confidence, 1.0)),
            summary=self._summary_from_answer(solution.answer),
            evidence=evidence,
            risks_flagged=self._extract_risks(solution.answer),
        )

    def _extract_signal(self, text: str) -> str:
        lower = text.lower()
        for key, mapped in _SIGNAL_MAP.items():
            if key in lower:
                return mapped
        return "neutral"

    @staticmethod
    def _summary_from_answer(text: str) -> str:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return lines[0] if lines else "No summary provided."

    @staticmethod
    def _extract_risks(text: str) -> List[str]:
        lower = text.lower()
        risks = []
        for key in ["valuation", "volatility", "liquidity", "macro", "event"]:
            if key in lower:
                risks.append(f"{key}_risk")
        return risks[:3]

    def _merge_outputs(self, outputs: List[AgentOutput]) -> Tuple[InvestmentRecommendation, InvestmentSummary]:
        if not outputs:
            rec = InvestmentRecommendation(
                action=RecommendationAction.WATCH,
                confidence=0.3,
                position_suggestion={"target_exposure_pct": 0.0, "max_drawdown_guard_pct": 0.08},
                decision_rationale="Insufficient agent outputs, fallback to watch.",
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

        rationale = self._build_decision_rationale(outputs, top_signal)
        rec = InvestmentRecommendation(
            action=action,
            confidence=round(confidence, 4),
            position_suggestion={
                "target_exposure_pct": exposure,
                "max_drawdown_guard_pct": 0.08,
            },
            decision_rationale=rationale,
        )
        summary = InvestmentSummary(
            thesis=f"Overall agent sentiment is {top_signal}.",
            key_drivers=[o.summary for o in outputs[:3]],
            key_risks=[risk for o in outputs for risk in o.risks_flagged][:3] or ["market_regime_shift", "event_risk"],
        )
        return rec, summary

    @staticmethod
    def _build_decision_rationale(outputs: List[AgentOutput], top_signal: str) -> str:
        lead_ids = ", ".join(o.agent_id for o in outputs[:2])
        return f"Top signal is {top_signal}, supported by {lead_ids}."

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
                decision_rationale=recommendation.decision_rationale,
            ),
            summary,
        )

    def _apply_portfolio_risk(
        self,
        request: InvestmentAnalysisRequest,
        normalized_market_data: Dict[str, Any],
        recommendation: InvestmentRecommendation,
        constraints_summary: Dict[str, Any],
    ) -> Tuple[InvestmentRecommendation, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Cap target exposure by volatility- and correlation-adjusted limits.

        Runs only when historical price series are available. If the data is
        missing the recommendation is returned unchanged and the summary gains
        a ``portfolio_risk_adjustment.skipped`` note.
        """
        target_history = self._extract_price_history(
            normalized_market_data, request.asset.symbol
        )
        updated_summary = dict(constraints_summary or {})

        if not target_history or len(target_history) < 5:
            updated_summary["portfolio_risk_adjustment"] = {
                "applied": False,
                "reason": "insufficient_price_history",
            }
            return recommendation, updated_summary, None

        price_series = {request.asset.symbol: list(target_history)}
        portfolio_ctx = dict(request.portfolio_context or {})
        history_map = portfolio_ctx.get("price_history_by_symbol") or {}
        if isinstance(history_map, dict):
            for sym, series in history_map.items():
                if sym == request.asset.symbol or not series:
                    continue
                cleaned = [float(p) for p in series if isinstance(p, (int, float)) and p > 0]
                if len(cleaned) >= 5:
                    price_series[sym] = cleaned

        try:
            assessment = self.portfolio_risk_engine.assess(
                target_symbol=request.asset.symbol,
                price_series_by_symbol=price_series,
                portfolio=portfolio_ctx,
            )
        except Exception as exc:
            updated_summary["portfolio_risk_adjustment"] = {
                "applied": False,
                "reason": f"engine_error: {exc}",
            }
            return recommendation, updated_summary, None

        new_recommendation, risk_summary = self._cap_recommendation_by_risk(
            recommendation, assessment
        )

        caps = dict(updated_summary.get("effective_caps") or {})
        caps["portfolio_risk_pct"] = round(assessment.combined_limit_pct, 6)
        updated_summary["effective_caps"] = caps
        triggered = list(updated_summary.get("triggered_rules") or [])
        if risk_summary["binding"]:
            triggered.append("portfolio_risk_cap")
        updated_summary["triggered_rules"] = triggered
        updated_summary["portfolio_risk_adjustment"] = {
            "applied": True,
            "binding": risk_summary["binding"],
            "position_before_pct": risk_summary["position_before_pct"],
            "position_after_pct": risk_summary["position_after_pct"],
            "engine_result": assessment.to_dict(),
        }

        payload = {
            "symbol": request.asset.symbol,
            "combined_limit_pct": round(assessment.combined_limit_pct, 6),
            "base_limit_pct": round(assessment.base_limit_pct, 6),
            "correlation_multiplier": round(assessment.correlation_multiplier, 4),
            "annualized_volatility": round(assessment.volatility.annualized_volatility, 6),
            "binding": risk_summary["binding"],
            "position_before_pct": risk_summary["position_before_pct"],
            "position_after_pct": risk_summary["position_after_pct"],
            "reasoning": assessment.reasoning,
        }
        return new_recommendation, updated_summary, payload

    @staticmethod
    def _cap_recommendation_by_risk(
        recommendation: InvestmentRecommendation,
        assessment: PortfolioRiskResult,
    ) -> Tuple[InvestmentRecommendation, Dict[str, Any]]:
        position = dict(recommendation.position_suggestion or {})
        before = float(position.get("target_exposure_pct", 0.0) or 0.0)
        cap = float(assessment.combined_limit_pct)
        binding = False
        after = before
        if before > cap:
            after = cap
            position["target_exposure_pct"] = cap
            binding = True
        next_action = recommendation.action
        if cap <= 0.0 and next_action == RecommendationAction.BUY:
            next_action = RecommendationAction.HOLD
            binding = True
        return (
            InvestmentRecommendation(
                action=next_action,
                confidence=recommendation.confidence,
                position_suggestion=position,
                decision_rationale=recommendation.decision_rationale,
            ),
            {
                "binding": binding,
                "position_before_pct": round(before, 6),
                "position_after_pct": round(after, 6),
            },
        )

    @staticmethod
    def _extract_price_history(
        normalized_market_data: Dict[str, Any],
        symbol: str,
    ) -> List[float]:
        """Pull a price series for ``symbol`` from gateway output.

        Looks in several well-known locations so providers can populate the
        field without coordinating on a single schema upfront.
        """
        candidates: List[Any] = []
        market = normalized_market_data.get("market") or {}
        if isinstance(market, dict):
            for key in ("price_history", "close_history", "history"):
                if market.get(key):
                    candidates.append(market[key])
        for key in ("price_history", "market_history", "history"):
            if normalized_market_data.get(key):
                candidates.append(normalized_market_data[key])
        for candidate in candidates:
            series = InvestmentAnalysisService._coerce_price_series(candidate, symbol)
            if series:
                return series
        return []

    @staticmethod
    def _coerce_price_series(candidate: Any, symbol: str) -> List[float]:
        """Best-effort conversion of gateway history payloads into List[float]."""
        if candidate is None:
            return []
        if isinstance(candidate, dict):
            # Either {symbol: [prices]} or {date: price}
            if symbol in candidate and isinstance(candidate[symbol], (list, tuple)):
                return InvestmentAnalysisService._coerce_price_series(candidate[symbol], symbol)
            values = list(candidate.values())
            if all(isinstance(v, (int, float)) for v in values):
                return [float(v) for v in values if v is not None and v > 0]
            if values and isinstance(values[0], (list, tuple)):
                return InvestmentAnalysisService._coerce_price_series(values[0], symbol)
        if isinstance(candidate, (list, tuple)):
            result: List[float] = []
            for item in candidate:
                if isinstance(item, (int, float)):
                    if item > 0:
                        result.append(float(item))
                elif isinstance(item, dict):
                    price = (
                        item.get("close")
                        or item.get("adj_close")
                        or item.get("price")
                    )
                    if isinstance(price, (int, float)) and price > 0:
                        result.append(float(price))
            return result
        return []

    def _select_debate_agent(self, role: str) -> Optional[Agent]:
        """Pick the best-weighted agent for a debate role.

        Falls back to any available agent if none is registered with weight
        for this specific debate role.
        """
        all_agents = self.agent_registry.get_all_agents()
        if not all_agents:
            return None
        ranked = sorted(all_agents, key=lambda a: a.get_weight_for_task(role), reverse=True)
        return ranked[0]

    async def _run_agent_turn(self, agent: Agent, prompt: str) -> Solution:
        try:
            return await agent.generate_solution(prompt)
        except Exception as exc:
            return Solution(agent_id=agent.agent_id, answer=f"Agent failed: {exc}", confidence=0.2)

    @staticmethod
    def _build_debate_context(
        request: InvestmentAnalysisRequest,
        agent_outputs: List[AgentOutput],
        provider_signals: List[str],
        group_injection: Optional[GroupKnowledgeInjection],
        current_recommendation: InvestmentRecommendation,
        portfolio_risk_reasoning: str = "",
    ) -> DebateContext:
        summaries: List[str] = []
        signals: Dict[str, str] = {}
        for out in agent_outputs:
            signals[out.role or out.agent_id] = out.signal
            line = f"{out.role or out.agent_id} [{out.signal}, conf={out.confidence:.2f}]: {out.summary}"
            summaries.append(line)
        group_context_str = ""
        if group_injection and group_injection.memory_context:
            group_context_str = f"\nGroup memory context:\n{group_injection.memory_context}\n"
        return DebateContext(
            symbol=request.asset.symbol,
            market=request.asset.market.value,
            asset_type=request.asset.asset_type.value,
            horizon=request.timeframe.horizon,
            risk_profile=request.risk_profile,
            objective=request.objective,
            analyst_summaries=summaries,
            analyst_signals=signals,
            provider_signals=provider_signals,
            group_context=group_context_str,
            portfolio_risk_reasoning=portfolio_risk_reasoning,
            candidate_action=current_recommendation.action.value,
            candidate_target_exposure_pct=float(
                current_recommendation.position_suggestion.get("target_exposure_pct", 0.0) or 0.0
            ),
        )

    async def _run_masters_panel(
        self,
        request: InvestmentAnalysisRequest,
        context: DebateContext,
        bull_history: List[str],
        bear_history: List[str],
        rounds: List[DiscussionRound],
        emit: Callable[..., Awaitable[None]],
        start_round_index: int,
    ) -> List[str]:
        """Optionally run a master-investor consultation inside a debate.

        Triggered only when ``request.metadata['panel_type'] == 'masters'``.
        Each persona in ``metadata['master_personas']`` (falling back to
        ``_DEFAULT_MASTER_PERSONAS``) runs one turn against the debate
        context and the Bull/Bear transcripts so far. Outputs are
        appended as a single ``stage='masters_panel'`` discussion round.
        Returns short summaries suitable for injection into downstream
        stages' analyst_summaries.
        """
        if not request.metadata or request.metadata.get("panel_type") != "masters":
            return []

        requested = request.metadata.get("master_personas") or list(_DEFAULT_MASTER_PERSONAS)
        known = set(_master_available_personas())
        personas = [k for k in requested if k in known]
        if not personas:
            return []

        chair_agent = self._select_debate_agent("chair")
        # Reuse chair agent as a generic LLM surface; fall back to bull if absent.
        runner = chair_agent or self._select_debate_agent("bull_researcher")
        if runner is None:
            return []

        recent_analysts = list(context.analyst_summaries)
        if bull_history:
            recent_analysts.append(f"bull_last: {bull_history[-1][:300]}")
        if bear_history:
            recent_analysts.append(f"bear_last: {bear_history[-1][:300]}")

        entries: List[DiscussionAgentEntry] = []
        summaries: List[str] = []
        for persona_key in personas:
            prompt = build_master_prompt(
                key=persona_key,
                symbol=request.asset.symbol,
                market=request.asset.market.value,
                asset_type=request.asset.asset_type.value,
                horizon=request.timeframe.horizon,
                analyst_summaries=recent_analysts,
                provider_signals=context.provider_signals,
                group_context=context.group_context,
            )
            solution = await self._run_agent_turn(runner, prompt)
            signal = self._extract_signal(solution.answer)
            summary = self._summary_from_answer(solution.answer)
            entries.append(
                DiscussionAgentEntry(
                    agent_id=runner.agent_id,
                    role=f"master_{persona_key}",
                    stance="master",
                    current_signal=signal,
                    summary=summary,
                    message=solution.answer,
                )
            )
            summaries.append(f"master/{persona_key} [{signal}]: {summary}")

        if not entries:
            return []

        masters_round_number = start_round_index + 1
        rounds.append(
            DiscussionRound(
                round_number=masters_round_number,
                stage="masters_panel",
                agents=entries,
            )
        )
        await emit(
            "debate_round_finished",
            round_number=masters_round_number,
            stage="masters_panel",
        )
        return summaries

    async def _run_debate(
        self,
        request: InvestmentAnalysisRequest,
        task_type: str,
        agent_outputs: List[AgentOutput],
        current_recommendation: InvestmentRecommendation,
        normalized_market_data: Dict[str, Any],
        group_injection: Optional[GroupKnowledgeInjection],
        emit: Callable[..., Awaitable[None]],
        bull_rounds: int = 2,
        risk_rounds: int = 1,
    ) -> Tuple[ConsensusResultView, InvestmentRecommendation, ConsensusTrace]:
        """Adversarial debate: Bull vs Bear -> Research Mgr -> Risk Debate -> Chair."""
        provider_status = normalized_market_data.get("provider_status") or {}
        provider_signals = self._collect_provider_signals(provider_status)
        context = self._build_debate_context(
            request,
            agent_outputs,
            provider_signals,
            group_injection,
            current_recommendation,
        )

        bull_agent = self._select_debate_agent("bull_researcher")
        bear_agent = self._select_debate_agent("bear_researcher")
        rm_agent = self._select_debate_agent("research_manager")
        chair_agent = self._select_debate_agent("chair")
        if not all([bull_agent, bear_agent, rm_agent, chair_agent]):
            # Not enough agents to run debate; fall back to current recommendation.
            return (
                ConsensusResultView(
                    enabled=True,
                    rounds_used=0,
                    consensus_reached=False,
                    final_action=current_recommendation.action.value,
                    weighted_votes=self._calculate_weighted_votes(agent_outputs),
                ),
                current_recommendation,
                ConsensusTrace(
                    discussion_enabled=True,
                    final_summary="Debate skipped: insufficient agents registered.",
                    rounds=[],
                ),
            )

        bull_history: List[str] = []
        bear_history: List[str] = []
        rounds: List[DiscussionRound] = []

        # Phase 1: Bull vs Bear alternating debate
        for round_idx in range(1, bull_rounds + 1):
            bull_prompt = build_researcher_prompt(
                "bull", round_idx, context, bull_history, bear_history
            )
            bull_solution = await self._run_agent_turn(bull_agent, bull_prompt)
            bull_history.append(bull_solution.answer)

            bear_prompt = build_researcher_prompt(
                "bear", round_idx, context, bear_history, bull_history
            )
            bear_solution = await self._run_agent_turn(bear_agent, bear_prompt)
            bear_history.append(bear_solution.answer)

            rounds.append(
                DiscussionRound(
                    round_number=round_idx,
                    stage="investment_debate",
                    candidate_action=current_recommendation.action.value,
                    candidate_confidence=current_recommendation.confidence,
                    agents=[
                        DiscussionAgentEntry(
                            agent_id=bull_agent.agent_id,
                            role="bull_researcher",
                            stance="bull",
                            current_signal="bullish",
                            summary=self._summary_from_answer(bull_solution.answer),
                            message=bull_solution.answer,
                        ),
                        DiscussionAgentEntry(
                            agent_id=bear_agent.agent_id,
                            role="bear_researcher",
                            stance="bear",
                            current_signal="bearish",
                            summary=self._summary_from_answer(bear_solution.answer),
                            message=bear_solution.answer,
                        ),
                    ],
                )
            )
            await emit(
                "debate_round_finished",
                round_number=round_idx,
                stage="investment_debate",
            )

        # Phase 1.5: Optional master-investor consultation (panel_type="masters")
        master_summaries = await self._run_masters_panel(
            request, context, bull_history, bear_history, rounds, emit,
            start_round_index=bull_rounds,
        )
        if master_summaries:
            # Feed master views into the context so Research Manager / Chair see them.
            context.analyst_summaries = list(context.analyst_summaries) + master_summaries

        # Phase 2: Research Manager synthesis
        rm_prompt = build_research_manager_prompt(context, bull_history, bear_history)
        rm_solution = await self._run_agent_turn(rm_agent, rm_prompt)
        plan_text = rm_solution.answer
        plan_action_signal = self._extract_signal(plan_text)
        plan_exposure = parse_target_exposure_pct(plan_text)
        plan_confidence = parse_confidence(plan_text)
        rm_round_number = len(rounds) + 1
        rounds.append(
            DiscussionRound(
                round_number=rm_round_number,
                stage="research_manager",
                candidate_action=(_ACTION_BY_SIGNAL.get(plan_action_signal, current_recommendation.action)).value,
                candidate_confidence=plan_confidence if plan_confidence is not None else current_recommendation.confidence,
                agents=[
                    DiscussionAgentEntry(
                        agent_id=rm_agent.agent_id,
                        role="research_manager",
                        stance="review",
                        current_signal=plan_action_signal,
                        summary=self._summary_from_answer(plan_text),
                        message=plan_text,
                    )
                ],
            )
        )
        await emit("debate_round_finished", round_number=rm_round_number, stage="research_manager")

        # Phase 3: Risk debate (Aggressive / Neutral / Conservative)
        peer_statements: Dict[str, str] = {}
        risk_round_index = rm_round_number
        for risk_round in range(1, risk_rounds + 1):
            risk_round_index += 1
            round_entries: List[DiscussionAgentEntry] = []
            for stance in _RISK_DEBATE_STANCES:
                role_name = f"risk_{stance}"
                risk_agent = self._select_debate_agent(role_name)
                if risk_agent is None:
                    continue
                prompt = build_risk_debate_prompt(stance, risk_round, context, plan_text, peer_statements)
                sol = await self._run_agent_turn(risk_agent, prompt)
                peer_statements[stance] = sol.answer
                round_entries.append(
                    DiscussionAgentEntry(
                        agent_id=risk_agent.agent_id,
                        role=role_name,
                        stance=stance,
                        current_signal=self._extract_signal(sol.answer),
                        summary=self._summary_from_answer(sol.answer),
                        message=sol.answer,
                    )
                )
            rounds.append(
                DiscussionRound(
                    round_number=risk_round_index,
                    stage="risk_debate",
                    candidate_action=(_ACTION_BY_SIGNAL.get(plan_action_signal, current_recommendation.action)).value,
                    candidate_confidence=plan_confidence if plan_confidence is not None else current_recommendation.confidence,
                    agents=round_entries,
                )
            )
            await emit(
                "debate_round_finished",
                round_number=risk_round_index,
                stage="risk_debate",
            )

        # Phase 4: Chair / Portfolio Manager
        chair_prompt = build_chair_prompt(context, plan_text, peer_statements)
        chair_solution = await self._run_agent_turn(chair_agent, chair_prompt)
        chair_text = chair_solution.answer
        chair_signal = self._extract_signal(chair_text)
        chair_action = _ACTION_BY_SIGNAL.get(chair_signal, current_recommendation.action)
        chair_exposure = parse_target_exposure_pct(chair_text)
        if chair_exposure is None:
            chair_exposure = plan_exposure
        chair_confidence = parse_confidence(chair_text)
        rounds.append(
            DiscussionRound(
                round_number=risk_round_index + 1,
                stage="chair",
                candidate_action=chair_action.value,
                candidate_confidence=chair_confidence if chair_confidence is not None else current_recommendation.confidence,
                agents=[
                    DiscussionAgentEntry(
                        agent_id=chair_agent.agent_id,
                        role="chair",
                        stance="final",
                        current_signal=chair_signal,
                        summary=self._summary_from_answer(chair_text),
                        message=chair_text,
                    )
                ],
            )
        )
        await emit("debate_round_finished", round_number=risk_round_index + 1, stage="chair")

        position_suggestion = dict(current_recommendation.position_suggestion or {})
        if chair_exposure is not None:
            position_suggestion["target_exposure_pct"] = chair_exposure
        final_confidence = chair_confidence if chair_confidence is not None else max(
            current_recommendation.confidence, chair_solution.confidence
        )
        new_recommendation = InvestmentRecommendation(
            action=chair_action,
            confidence=max(0.0, min(1.0, final_confidence)),
            position_suggestion=position_suggestion,
            decision_rationale=self._summary_from_answer(chair_text) or current_recommendation.decision_rationale,
        )

        weighted_votes = self._calculate_weighted_votes(agent_outputs)
        view = ConsensusResultView(
            enabled=True,
            rounds_used=len(rounds),
            consensus_reached=True,
            final_action=chair_action.value,
            weighted_votes=weighted_votes,
        )
        trace = ConsensusTrace(
            discussion_enabled=True,
            final_summary=self._summary_from_answer(chair_text),
            rounds=rounds,
        )
        return view, new_recommendation, trace

    async def _run_roundtable(
        self,
        task: str,
        agents: List[Agent],
        current_recommendation: InvestmentRecommendation,
        agent_outputs: List[AgentOutput],
        emit: Callable[..., Awaitable[None]],
    ) -> Tuple[ConsensusResultView, InvestmentRecommendation, ConsensusTrace]:
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
                decision_rationale=current_recommendation.decision_rationale,
            )
        else:
            recommendation = current_recommendation

        weighted_votes = self._calculate_weighted_votes(agent_outputs)
        final_action = recommendation.action.value
        view = ConsensusResultView(
            enabled=True,
            rounds_used=result.rounds_used,
            consensus_reached=result.consensus_reached,
            final_action=final_action,
            weighted_votes=weighted_votes,
        )
        trace = self._build_consensus_trace(agent_outputs, final_action, recommendation.confidence)
        for round_info in trace.rounds:
            await emit(
                "round_started",
                round_number=round_info.round_number,
                stage=round_info.stage,
                candidate_action=round_info.candidate_action,
            )
            for agent_entry in round_info.agents:
                await emit(
                    "agent_replied_in_round",
                    round_number=round_info.round_number,
                    stage=round_info.stage,
                    agent_id=agent_entry.agent_id,
                    stance=agent_entry.stance,
                    current_signal=agent_entry.current_signal,
                    changed_position=agent_entry.changed_position,
                )
            await emit(
                "round_completed",
                round_number=round_info.round_number,
                stage=round_info.stage,
                candidate_action=round_info.candidate_action,
                candidate_confidence=round_info.candidate_confidence,
            )
        return view, recommendation, trace

    @staticmethod
    def _calculate_weighted_votes(outputs: List[AgentOutput]) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        for output in outputs:
            action = _ACTION_BY_SIGNAL.get(output.signal, RecommendationAction.HOLD).value
            votes[action] = round(votes.get(action, 0.0) + output.confidence, 4)
        return votes

    def _build_consensus_trace(
        self,
        agent_outputs: List[AgentOutput],
        final_action: str,
        final_confidence: float,
    ) -> ConsensusTrace:
        if not agent_outputs:
            return ConsensusTrace(discussion_enabled=False)

        initial_round_agents = [
            DiscussionAgentEntry(
                agent_id=output.agent_id,
                role=output.role,
                stance=output.stance or "review",
                current_signal=output.signal,
                changed_position=False,
                summary=output.summary,
                message=f"[{output.role}] {output.summary}",
                evidence=output.evidence,
            )
            for output in agent_outputs
        ]

        challenge_round_agents = [
            DiscussionAgentEntry(
                agent_id=output.agent_id,
                role=output.role,
                stance="challenge" if output.signal != self._signal_for_action(final_action) else "defend",
                previous_signal=output.signal,
                current_signal=output.signal,
                changed_position=False,
                summary=f"{output.role} reviews the committee draft against its own lens.",
                message=(
                    f"[{output.role}] Challenge the draft recommendation on {', '.join(output.risks_flagged[:2])}."
                    if output.risks_flagged
                    else f"[{output.role}] Stress-test whether the draft {final_action} call is justified."
                ),
                evidence=output.evidence,
            )
            for output in agent_outputs
        ]

        final_signal = self._signal_for_action(final_action)
        final_round_agents = [
            DiscussionAgentEntry(
                agent_id=output.agent_id,
                role=output.role,
                stance="support" if output.signal == final_signal else "revise",
                previous_signal=output.signal,
                current_signal=final_signal,
                changed_position=output.signal != final_signal,
                summary=(
                    f"{output.role} aligns to the final {final_action} committee recommendation."
                    if output.signal != final_signal
                    else output.summary
                ),
                message=(
                    f"[{output.role}] Adjusted view to {final_action} after committee challenge round."
                    if output.signal != final_signal
                    else f"[{output.role}] Supports the final {final_action} recommendation."
                ),
                evidence=output.evidence,
            )
            for output in agent_outputs
        ]

        rounds = [
            DiscussionRound(
                round_number=1,
                stage="opening_statements",
                candidate_action=_ACTION_BY_SIGNAL.get(agent_outputs[0].signal, RecommendationAction.HOLD).value,
                candidate_confidence=max(agent_outputs[0].confidence, 0.0),
                agents=initial_round_agents,
                agreement_points=["Committee opening statements collected from each role-specific panelist."],
                disagreement_points=["Initial role-based views still point to different actions or risk emphases."],
            ),
            DiscussionRound(
                round_number=2,
                stage="cross_challenge",
                candidate_action=final_action,
                candidate_confidence=max(final_confidence * 0.9, 0.0),
                agents=challenge_round_agents,
                agreement_points=["Panelists pressure-tested the draft recommendation across valuation, macro, and risk lenses."],
                disagreement_points=["Some panelists requested revisions before committee convergence."],
            ),
            DiscussionRound(
                round_number=3,
                stage="chair_synthesis",
                candidate_action=final_action,
                candidate_confidence=final_confidence,
                agents=final_round_agents,
                agreement_points=[f"Committee chair synthesized the panel into a final {final_action} stance."],
                disagreement_points=[],
            ),
        ]
        return ConsensusTrace(
            discussion_enabled=True,
            final_summary=(
                f"Investment committee completed opening statements, cross-challenge, and chair synthesis before settling on {final_action}."
            ),
            rounds=rounds,
        )

    @staticmethod
    def _signal_for_action(action: str) -> str:
        if action == RecommendationAction.BUY.value:
            return "bullish"
        if action == RecommendationAction.SELL.value:
            return "bearish"
        return "neutral"

    def _build_analysis_framework(
        self,
        mode: InvestmentMode,
        task_type: str,
        selected_skills: List[str],
        data_sources: List[str],
        resolved_skill_profiles: List[Dict[str, Any]],
        group_injection: Optional[GroupKnowledgeInjection],
        panel_roles: List[str],
        panel_role_skills: Dict[str, List[str]],
        panel_role_data_focus: Dict[str, List[str]],
    ) -> AnalysisFramework:
        why_selected = [
            f"Detected {task_type} task.",
            f"Mode {mode.value} selected {len(selected_skills)} core skill routes.",
        ]
        if mode == InvestmentMode.ROUNDTABLE:
            why_selected.append("Roundtable mode enabled consensus synthesis.")
        if group_injection:
            why_selected.append(
                f"Attached group shared knowledge from {group_injection.group_id} with {len(group_injection.skill_descriptions)} skills and {len(group_injection.graph_ids)} graphs."
            )
        if panel_roles:
            why_selected.append(
                "Investment committee panel active: " + ", ".join(panel_roles)
            )
        return AnalysisFramework(
            style="multi_agent_investment_review",
            task_type=task_type,
            selected_skills=selected_skills,
            resolved_skill_profiles=resolved_skill_profiles,
            data_sources=data_sources,
            panel_roles=panel_roles,
            panel_role_skills=panel_role_skills,
            panel_role_data_focus=panel_role_data_focus,
            why_selected=why_selected,
        )

    @staticmethod
    def _build_external_evidence(normalized_market_data: Dict[str, Any]) -> Dict[str, List[str]]:
        news = [str(item).strip() for item in (normalized_market_data.get("news") or []) if str(item).strip()]
        provider_signals = [
            str(item).strip() for item in (normalized_market_data.get("provider_signals") or []) if str(item).strip()
        ]
        supporting = news[:3]
        negative = []
        for item in news:
            lower = item.lower()
            if any(token in lower for token in ["risk", "warn", "fall", "drop", "probe", "lawsuit", "cut", "weak"]):
                negative.append(item)
        if not negative:
            negative = provider_signals[:2]
        return {
            "supporting": supporting,
            "negative": negative[:3],
            "organizations": [item.split(" - ")[0] for item in news[:3] if " - " in item],
        }

    @staticmethod
    def _build_bull_bear_cases(
        summary: InvestmentSummary,
        outputs: List[AgentOutput],
        external_evidence: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str]]:
        bull_case = [o.summary for o in outputs if o.signal == "bullish"][:3]
        bear_case = [o.summary for o in outputs if o.signal == "bearish"][:3]
        if not bull_case:
            bull_case = summary.key_drivers[:2]
        if external_evidence.get("supporting"):
            bull_case.extend(external_evidence.get("supporting", [])[:2])
        if not bear_case:
            bear_case = summary.key_risks[:2]
        if external_evidence.get("negative"):
            bear_case.extend(external_evidence.get("negative", [])[:2])
        return bull_case[:4], bear_case[:4]

    def _build_catalysts(self, request: InvestmentAnalysisRequest, external_evidence: Dict[str, List[str]]) -> List[CatalystItem]:
        catalysts = [
            CatalystItem(
                name=f"{request.asset.symbol} next earnings or major update",
                direction="two_way",
                importance="high",
                time_horizon=request.timeframe.horizon,
            )
        ]
        for headline in external_evidence.get("supporting", [])[:2]:
            catalysts.append(
                CatalystItem(
                    name=headline,
                    direction="positive",
                    importance="medium",
                    time_horizon=request.timeframe.horizon,
                )
            )
        return catalysts[:3]

    @staticmethod
    def _build_scenarios(recommendation: InvestmentRecommendation) -> List[ScenarioItem]:
        action = recommendation.action.value
        alt = "hold" if action == "buy" else "buy"
        return [
            ScenarioItem(name="base", probability=0.6, view=action, description="Most likely path under current information."),
            ScenarioItem(name="bull", probability=0.25, view="buy", description="Positive catalysts improve upside asymmetry."),
            ScenarioItem(name="bear", probability=0.15, view=alt if alt != "buy" else "sell", description="Risk events weaken the setup."),
        ]

    @staticmethod
    def _build_disagreement_summary(
        summary: InvestmentSummary,
        outputs: List[AgentOutput],
    ) -> DisagreementSummary:
        unique_signals = sorted({output.signal for output in outputs})
        disagreement_points = []
        if len(unique_signals) > 1:
            disagreement_points.append("Agents disagree on the strength of the current setup.")
        agreement_points = summary.key_drivers[:2] or ["Agents agree the asset deserves continued monitoring."]
        return DisagreementSummary(
            main_conflict=(
                "Signal dispersion exists between supportive and cautious agents."
                if disagreement_points
                else "Agents are broadly aligned on the current setup."
            ),
            agreement_points=agreement_points,
            disagreement_points=disagreement_points,
        )

    @staticmethod
    def _build_policy_overrides(summary: Dict[str, Any]) -> PolicyOverrides:
        return PolicyOverrides(
            input_action=summary.get("input_action", ""),
            output_action=summary.get("output_action", ""),
            input_target_exposure_pct=summary.get("input_target_exposure_pct", 0.0),
            output_target_exposure_pct=summary.get("output_target_exposure_pct", 0.0),
            binding_cap=summary.get("binding_cap", "none"),
            effective_caps=summary.get("effective_caps", {}),
            triggered_rules=summary.get("triggered_rules", []),
            human_readable_explanation=(
                f"Recommendation adjusted from {summary.get('input_action', 'n/a')} to "
                f"{summary.get('output_action', 'n/a')} under active constraints."
            ),
        )

    async def _evaluate_risk(
        self,
        request: InvestmentAnalysisRequest,
        recommendation: InvestmentRecommendation,
        group_injection: Optional[GroupKnowledgeInjection] = None,
    ) -> RiskGateResult:
        if not self.risk_coordinator:
            return RiskGateResult(
                status="pass",
                risk_level="low",
                risk_indicators=[],
                review_summary="No hard risk block triggered.",
            )

        exposure = recommendation.position_suggestion.get("target_exposure_pct", 0.0)
        amount = float((request.constraints or {}).get("notional_amount", 0.0))
        if amount <= 0:
            amount = 10_000 * exposure

        trace_context = request.custom_question
        if group_injection:
            extras = []
            if group_injection.memory_context:
                extras.append(group_injection.memory_context)
            if group_injection.skill_descriptions:
                extras.append("group_skills=" + "; ".join(group_injection.skill_descriptions[:6]))
            if group_injection.graph_ids:
                extras.append("graph_ids=" + ", ".join(group_injection.graph_ids[:5]))
            if extras:
                trace_context = "\n\n".join([item for item in [trace_context, *extras] if item])

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
                trace_context=trace_context,
            ),
            priority="normal",
            metadata={
                "investment_mode": request.mode.value,
                "group_id": group_injection.group_id if group_injection else None,
                "group_graph_ids": (group_injection.graph_ids if group_injection else []),
            },
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
            review_summary="Risk coordinator reviewed the recommendation.",
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
        external_evidence: Dict[str, List[str]],
    ) -> str:
        lines = [
            f"# Investment Analysis: {request.asset.symbol}",
            "",
            f"- Mode: {request.mode.value}",
            f"- Market: {request.asset.market.value}",
            f"- Asset Type: {request.asset.asset_type.value}",
            f"- Horizon: {request.timeframe.horizon}",
            "",
            "## Investment Committee Panel",
        ]
        committee_roles = [out.role for out in outputs if out.role]
        lines.extend([f"- {role}" for role in committee_roles] or ["- N/A"])
        lines.extend([
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
        ])
        lines.extend([f"- {d}" for d in summary.key_drivers] or ["- N/A"])
        lines.extend(["", "## Key Risks"])
        lines.extend([f"- {r}" for r in summary.key_risks] or ["- N/A"])

        lines.extend(["", "## External Evidence"])
        lines.extend([f"- Supporting: {item}" for item in external_evidence.get("supporting", [])] or ["- Supporting: N/A"])
        lines.extend([f"- Negative: {item}" for item in external_evidence.get("negative", [])] or ["- Negative: N/A"])

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

    def create_analysis_run(self, request: InvestmentAnalysisRequest, request_id: Optional[str] = None) -> str:
        analysis_id = request_id or f"inv-{uuid.uuid4().hex[:12]}"
        self.analysis_runs[analysis_id] = {
            "status": "pending",
            "request": request.model_dump(mode="json"),
            "timeline": [],
            "agents": [],
            "result": None,
            "discussion": {"enabled": False, "final_summary": "", "rounds": []},
            "policy_overrides": {},
            "risk_gate": {},
        }
        return analysis_id

    def get_analysis_run(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self.analysis_runs.get(request_id)

    def _update_analysis_run(self, event: Dict[str, Any], agents_panel: List[Dict[str, Any]]) -> None:
        request_id = event.get("request_id")
        if not request_id or request_id not in self.analysis_runs:
            return

        payload = event.get("payload", {})
        run = self.analysis_runs[request_id]
        event_type = event.get("type")

        if event_type == "analysis_started":
            run["status"] = "running"
        elif event_type == "analysis_completed":
            run["status"] = "completed"
        elif event_type == "roundtable_started":
            run["status"] = "roundtable"
        elif event_type == "constraints_applied":
            run["policy_overrides"] = payload.get("constraints_applied_summary", {})
        elif event_type == "agent_started":
            agent_entry = {
                "agent_id": payload.get("agent_id"),
                "role": payload.get("role", ""),
                "status": "running",
                "signal": "",
                "confidence": 0.0,
                "summary": "",
            }
            agents_panel.append(agent_entry)
        elif event_type in {"agent_completed", "agent_failed"}:
            for item in agents_panel:
                if item.get("agent_id") == payload.get("agent_id"):
                    item["status"] = "completed" if event_type == "agent_completed" else "failed"
                    item["signal"] = payload.get("signal", item.get("signal", ""))
                    item["confidence"] = payload.get("confidence", item.get("confidence", 0.0))
                    item["summary"] = payload.get("summary", item.get("summary", ""))
                    break
        elif event_type == "agent_replied_in_round":
            run.setdefault("round_replies", []).append(payload)
        elif event_type == "risk_gate_finished":
            run["risk_gate"] = {
                "status": payload.get("status", "pass"),
                "risk_level": payload.get("risk_level", "low"),
            }

    async def _emit(self, event_sink: EventSink, event_type: str, **payload: Any) -> None:
        if event_sink is None:
            return
        event_timestamp = payload.pop("event_timestamp", _utc_now().isoformat())
        request_id = payload.pop("request_id", "unknown")
        await event_sink(
            {
                "type": event_type,
                "timestamp": event_timestamp,
                "request_id": request_id,
                "payload": payload,
            }
        )
