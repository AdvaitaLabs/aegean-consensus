"""Unit tests for investment analysis service."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from aegean.core.agent import Agent, AgentRegistry
from aegean.core.models import Solution
from aegean.investment.models import (
    AssetType,
    InvestmentAnalysisRequest,
    InvestmentAsset,
    InvestmentMode,
    InvestmentTimeframe,
    MarketCode,
    RecommendationAction,
)
from aegean.investment.service import (
    InvestmentAnalysisService,
    _compute_constrained_recommendation,
)


class DummyAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        answer: str = "BUY with momentum",
        specialization: Optional[Dict[str, float]] = None,
    ):
        super().__init__(agent_id=agent_id, specialization=specialization or {})
        self._answer = answer

    async def generate_solution(self, task: str) -> Solution:
        return Solution(
            agent_id=self.agent_id,
            answer=self._answer,
            confidence=0.9,
        )

    async def refine_solution(self, refinement_set: List[Solution]) -> Solution:
        return Solution(
            agent_id=self.agent_id,
            answer=self._answer,
            confidence=0.9,
        )


def _build_request(mode: InvestmentMode, asset_type: AssetType = AssetType.EQUITY) -> InvestmentAnalysisRequest:
    return InvestmentAnalysisRequest(
        mode=mode,
        asset=InvestmentAsset(
            symbol="AAPL",
            market=MarketCode.US,
            asset_type=asset_type,
        ),
        timeframe=InvestmentTimeframe(horizon="1m"),
        public_facts=["Revenue growth remains stable"],
        user_id="user_test",
    )


def _build_service(agent_count: int = 3) -> InvestmentAnalysisService:
    registry = AgentRegistry()
    for idx in range(agent_count):
        registry.register_agent(DummyAgent(agent_id=f"agent_{idx}"))
    return InvestmentAnalysisService(agent_registry=registry)


def _build_specialized_service() -> InvestmentAnalysisService:
    registry = AgentRegistry()
    registry.register_agent(DummyAgent(agent_id="agent_eq", specialization={"equity_analysis": 0.95}))
    registry.register_agent(DummyAgent(agent_id="agent_opt", specialization={"options_analysis": 1.0}))
    registry.register_agent(DummyAgent(agent_id="agent_crypto", specialization={"crypto_analysis": 1.0}))
    registry.register_agent(DummyAgent(agent_id="agent_generic"))
    return InvestmentAnalysisService(agent_registry=registry)


@pytest.mark.asyncio
async def test_auto_mode_does_not_enable_roundtable_consensus() -> None:
    service = _build_service(agent_count=4)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.EQUITY)

    resp = await service.analyze(req)

    assert resp.mode == InvestmentMode.AUTO
    assert resp.consensus.enabled is False
    assert len(resp.agent_outputs) == 2


@pytest.mark.asyncio
async def test_roundtable_mode_enables_consensus() -> None:
    service = _build_service(agent_count=3)
    req = _build_request(mode=InvestmentMode.ROUNDTABLE, asset_type=AssetType.ETF)

    resp = await service.analyze(req)

    assert resp.mode == InvestmentMode.ROUNDTABLE
    assert resp.consensus.enabled is True


@pytest.mark.asyncio
async def test_v3_asset_types_supported() -> None:
    service = _build_service(agent_count=2)

    futures_req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.FUTURES)
    options_req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.OPTIONS)
    crypto_req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.CRYPTO)

    futures_resp = await service.analyze(futures_req)
    options_resp = await service.analyze(options_req)
    crypto_resp = await service.analyze(crypto_req)

    assert futures_resp.asset.asset_type == AssetType.FUTURES
    assert options_resp.asset.asset_type == AssetType.OPTIONS
    assert crypto_resp.asset.asset_type == AssetType.CRYPTO


@pytest.mark.asyncio
async def test_event_sink_receives_progress_events_with_constraints_summary() -> None:
    service = _build_service(agent_count=3)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.INDEX)

    events = []

    async def sink(evt):
        events.append(evt)

    await service.analyze(req, event_sink=sink)

    event_types = [evt["type"] for evt in events]
    assert "analysis_started" in event_types
    assert "request_validated" in event_types
    assert "agents_selected" in event_types
    assert "agent_completed" in event_types
    assert "constraints_applied" in event_types
    assert "recommendation_ready" in event_types
    assert "risk_gate_finished" in event_types
    assert "analysis_completed" in event_types
    assert "roundtable_started" not in event_types

    selected_event = next(evt for evt in events if evt["type"] == "agents_selected")
    assert selected_event["payload"]["task_type"] == "index_analysis"
    assert isinstance(selected_event["payload"]["selected_skills"], list)

    constraints_event = next(evt for evt in events if evt["type"] == "constraints_applied")
    assert "constraints_applied_summary" in constraints_event["payload"]


@pytest.mark.asyncio
async def test_v3_metadata_contains_task_skills_and_data_sources() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.OPTIONS)

    resp = await service.analyze(req)

    assert resp.metadata.task_type == "options_analysis"
    assert "options_greeks" in resp.metadata.selected_skills
    assert "option_chain_feed" in resp.metadata.data_sources


@pytest.mark.asyncio
async def test_skill_routing_prefers_specialized_agent_for_fast_mode() -> None:
    service = _build_specialized_service()
    req = _build_request(mode=InvestmentMode.FAST, asset_type=AssetType.OPTIONS)

    resp = await service.analyze(req)

    assert len(resp.agent_outputs) == 1
    assert resp.agent_outputs[0].agent_id == "agent_opt"


@pytest.mark.asyncio
async def test_constraints_cap_exposure_and_block_sell() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.EQUITY)
    req.constraints = {
        "allowed_actions": ["hold", "buy"],
        "max_exposure_pct": 0.03,
        "max_drawdown_guard_pct": 0.02,
    }

    resp = await service.analyze(req)

    assert resp.recommendation.action.value in {"hold", "buy"}
    assert resp.recommendation.position_suggestion["target_exposure_pct"] <= 0.03
    assert resp.recommendation.position_suggestion["max_drawdown_guard_pct"] == 0.02


@pytest.mark.asyncio
async def test_portfolio_context_caps_by_remaining_budget() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.EQUITY)
    req.portfolio_context = {
        "current_total_exposure_pct": 0.18,
        "max_total_exposure_pct": 0.2,
    }

    resp = await service.analyze(req)

    assert resp.recommendation.position_suggestion["target_exposure_pct"] <= 0.02


@pytest.mark.asyncio
async def test_roundtable_still_respects_constraints() -> None:
    service = _build_service(agent_count=3)
    req = _build_request(mode=InvestmentMode.ROUNDTABLE, asset_type=AssetType.EQUITY)
    req.constraints = {
        "allowed_actions": ["hold"],
        "max_exposure_pct": 0.01,
    }

    resp = await service.analyze(req)

    assert resp.consensus.enabled is True
    assert resp.recommendation.action.value == "hold"
    assert resp.recommendation.position_suggestion["target_exposure_pct"] <= 0.01


@pytest.mark.asyncio
async def test_metadata_contains_constraints_applied_summary() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.EQUITY)
    req.constraints = {"max_exposure_pct": 0.03}

    resp = await service.analyze(req)

    summary = resp.metadata.constraints_applied_summary
    assert summary["output_target_exposure_pct"] <= 0.03
    assert summary["binding_cap"] in summary["effective_caps"]


@pytest.mark.asyncio
async def test_invalid_risk_profile_rejected() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.EQUITY)
    req.risk_profile = "ultra_safe"

    with pytest.raises(ValueError, match="risk_profile must be one of"):
        await service.analyze(req)


@pytest.mark.asyncio
async def test_invalid_objective_rejected() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.EQUITY)
    req.objective = "hyper_growth"

    with pytest.raises(ValueError, match="objective must be one of"):
        await service.analyze(req)


def test_compute_constrained_recommendation_caps_by_profile_and_objective() -> None:
    action, position, summary = _compute_constrained_recommendation(
        action=RecommendationAction.BUY,
        position_suggestion={"target_exposure_pct": 0.2, "max_drawdown_guard_pct": 0.08},
        constraints={},
        risk_profile="conservative",
        objective="alpha",
        asset_symbol="AAPL",
        asset_type=AssetType.EQUITY,
        portfolio_context=None,
    )

    assert action == RecommendationAction.BUY
    assert position["target_exposure_pct"] == 0.05
    assert summary["binding_cap"] == "risk_profile"


def test_compute_constrained_recommendation_enforces_allowed_actions_then_hold_cap() -> None:
    action, position, summary = _compute_constrained_recommendation(
        action=RecommendationAction.BUY,
        position_suggestion={"target_exposure_pct": 0.2},
        constraints={"allowed_actions": ["hold"]},
        risk_profile="aggressive",
        objective="alpha",
        asset_symbol="AAPL",
        asset_type=AssetType.EQUITY,
        portfolio_context=None,
    )

    assert action == RecommendationAction.HOLD
    assert position["target_exposure_pct"] == 0.05
    assert "allowed_actions" in summary["triggered_rules"]


def test_compute_constrained_recommendation_disallowed_symbol_forces_hold() -> None:
    action, position, summary = _compute_constrained_recommendation(
        action=RecommendationAction.BUY,
        position_suggestion={"target_exposure_pct": 0.12},
        constraints={"disallowed_symbols": ["AAPL"]},
        risk_profile="balanced",
        objective="balanced",
        asset_symbol="AAPL",
        asset_type=AssetType.EQUITY,
        portfolio_context=None,
    )

    assert action == RecommendationAction.HOLD
    assert "disallowed_symbols" in summary["triggered_rules"]
    assert position["target_exposure_pct"] <= 0.05


def test_compute_constrained_recommendation_applies_portfolio_single_position_cap() -> None:
    action, position, summary = _compute_constrained_recommendation(
        action=RecommendationAction.BUY,
        position_suggestion={"target_exposure_pct": 0.12},
        constraints={},
        risk_profile="aggressive",
        objective="alpha",
        asset_symbol="AAPL",
        asset_type=AssetType.EQUITY,
        portfolio_context={"max_single_position_pct": 0.04},
    )

    assert action == RecommendationAction.BUY
    assert position["target_exposure_pct"] == 0.04
    assert summary["binding_cap"] == "portfolio_max_single_position_pct"


def test_compute_constrained_recommendation_applies_max_exposure_and_drawdown_override() -> None:
    action, position, summary = _compute_constrained_recommendation(
        action=RecommendationAction.BUY,
        position_suggestion={"target_exposure_pct": 0.12, "max_drawdown_guard_pct": 0.08},
        constraints={"max_exposure_pct": 0.03, "max_drawdown_guard_pct": 0.02},
        risk_profile="aggressive",
        objective="alpha",
        asset_symbol="AAPL",
        asset_type=AssetType.EQUITY,
        portfolio_context=None,
    )

    assert action == RecommendationAction.BUY
    assert position["target_exposure_pct"] == 0.03
    assert position["max_drawdown_guard_pct"] == 0.02
    assert "max_drawdown_guard_pct" in summary["triggered_rules"]


def test_compute_constrained_recommendation_sell_forces_zero_exposure() -> None:
    action, position, summary = _compute_constrained_recommendation(
        action=RecommendationAction.SELL,
        position_suggestion={"target_exposure_pct": 0.07},
        constraints={},
        risk_profile="balanced",
        objective="balanced",
        asset_symbol="AAPL",
        asset_type=AssetType.EQUITY,
        portfolio_context=None,
    )

    assert action == RecommendationAction.SELL
    assert position["target_exposure_pct"] == 0.0
    assert "sell_forces_zero" in summary["triggered_rules"]
