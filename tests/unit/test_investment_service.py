"""Unit tests for investment analysis service."""

from __future__ import annotations

from typing import List

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
)
from aegean.investment.service import InvestmentAnalysisService


class DummyAgent(Agent):
    def __init__(self, agent_id: str, answer: str = "BUY with momentum"):
        super().__init__(agent_id=agent_id)
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
async def test_v2_asset_type_rejected_in_v1() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.FUND)

    with pytest.raises(ValueError, match="planned for V2"):
        await service.analyze(req)


@pytest.mark.asyncio
async def test_v3_asset_type_rejected_in_v1() -> None:
    service = _build_service(agent_count=2)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.CRYPTO)

    with pytest.raises(ValueError, match="planned for V3"):
        await service.analyze(req)


@pytest.mark.asyncio
async def test_event_sink_receives_progress_events() -> None:
    service = _build_service(agent_count=3)
    req = _build_request(mode=InvestmentMode.AUTO, asset_type=AssetType.INDEX)

    events = []

    async def sink(evt):
        events.append(evt["type"])

    await service.analyze(req, event_sink=sink)

    assert "analysis_started" in events
    assert "request_validated" in events
    assert "agents_selected" in events
    assert "agent_completed" in events
    assert "constraints_applied" in events
    assert "recommendation_ready" in events
    assert "risk_gate_finished" in events
    assert "analysis_completed" in events
    assert "roundtable_started" not in events


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

