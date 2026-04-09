"""Unit tests for investment API handlers and SSE behavior."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from fastapi import HTTPException

from aegean.api import investment_api
from aegean.investment.models import (
    AnalysisFramework,
    AssetType,
    ConsensusResultView,
    ConsensusTrace,
    DisagreementSummary,
    InvestmentAnalysisRequest,
    InvestmentAnalysisResponse,
    InvestmentAsset,
    InvestmentMetadata,
    InvestmentMode,
    InvestmentRecommendation,
    InvestmentSummary,
    InvestmentTimeframe,
    MarketCode,
    PolicyOverrides,
    RecommendationAction,
    RiskGateResult,
)


class FakeInvestmentService:
    def __init__(self, behavior: str = "ok"):
        self.behavior = behavior
        self.runs: Dict[str, Dict[str, Any]] = {
            "inv-test-001": {
                "status": "completed",
                "timeline": [
                    {
                        "type": "analysis_started",
                        "timestamp": "2026-04-09T10:00:00Z",
                        "request_id": "inv-test-001",
                        "payload": {"mode": "auto"},
                    },
                    {
                        "type": "constraints_applied",
                        "timestamp": "2026-04-09T10:00:02Z",
                        "request_id": "inv-test-001",
                        "payload": {
                            "constraints_applied_summary": {
                                "input_action": "buy",
                                "output_action": "hold",
                                "binding_cap": "objective",
                            }
                        },
                    },
                ],
                "agents": [
                    {
                        "agent_id": "agent_0",
                        "role": "fundamental_specialist",
                        "status": "completed",
                        "signal": "bullish",
                        "confidence": 0.9,
                        "summary": "Momentum remains constructive.",
                    }
                ],
                "discussion": {
                    "enabled": True,
                    "final_summary": "Agents converged on hold.",
                    "rounds": [
                        {
                            "round_number": 1,
                            "candidate_action": "buy",
                            "candidate_confidence": 0.7,
                            "agents": [],
                            "agreement_points": [],
                            "disagreement_points": ["valuation"],
                        }
                    ],
                },
                "policy_overrides": {"binding_cap": "objective"},
                "risk_gate": {"status": "pass", "risk_level": "low"},
                "result": None,
            }
        }

    async def analyze(self, body: InvestmentAnalysisRequest, event_sink=None) -> InvestmentAnalysisResponse:
        if self.behavior == "value_error":
            raise ValueError("invalid request")
        if self.behavior == "runtime_error":
            raise RuntimeError("unexpected")

        if event_sink is not None:
            await event_sink(
                {
                    "type": "analysis_started",
                    "timestamp": "2026-04-09T10:00:00Z",
                    "request_id": "inv-test-001",
                    "payload": {"mode": body.mode.value},
                }
            )
            await event_sink(
                {
                    "type": "agent_started",
                    "timestamp": "2026-04-09T10:00:01Z",
                    "request_id": "inv-test-001",
                    "payload": {"agent_id": "agent_0", "role": "fundamental_specialist"},
                }
            )
            await event_sink(
                {
                    "type": "constraints_applied",
                    "timestamp": "2026-04-09T10:00:02Z",
                    "request_id": "inv-test-001",
                    "payload": {
                        "constraints_applied_summary": {
                            "input_action": "buy",
                            "output_action": "hold",
                            "binding_cap": "objective",
                        }
                    },
                }
            )

        response = _build_response(body)
        self.runs["inv-test-001"]["result"] = response.model_dump(mode="json")
        return response

    def get_analysis_run(self, request_id: str) -> Dict[str, Any] | None:
        return self.runs.get(request_id)


@pytest.fixture(autouse=True)
def _reset_service() -> None:
    investment_api._service = None
    yield
    investment_api._service = None


def _build_request(asset_type: AssetType = AssetType.EQUITY) -> InvestmentAnalysisRequest:
    return InvestmentAnalysisRequest(
        mode=InvestmentMode.AUTO,
        asset=InvestmentAsset(symbol="AAPL", market=MarketCode.US, asset_type=asset_type),
        timeframe=InvestmentTimeframe(horizon="1m"),
        risk_profile="balanced",
        objective="balanced",
        user_id="u-test",
    )


def _build_response(body: InvestmentAnalysisRequest) -> InvestmentAnalysisResponse:
    if body.asset.asset_type == AssetType.OPTIONS:
        data_sources = ["public_market_data", "option_chain_feed", "vol_surface_feed"]
        selected_skills = ["options_greeks", "vol_surface"]
        task_type = "options_analysis"
    else:
        data_sources = ["public_market_data"]
        selected_skills = ["fundamental_analysis"]
        task_type = "equity_analysis"

    return InvestmentAnalysisResponse(
        request_id="inv-test-001",
        status="completed",
        mode=body.mode,
        asset=body.asset,
        timeframe=body.timeframe,
        analysis_framework=AnalysisFramework(
            task_type=task_type,
            selected_skills=selected_skills,
            data_sources=data_sources,
            why_selected=["Detected task type"],
        ),
        recommendation=InvestmentRecommendation(
            action=RecommendationAction.HOLD,
            confidence=0.8,
            position_suggestion={"target_exposure_pct": 0.05, "max_drawdown_guard_pct": 0.08},
            decision_rationale="valuation and risk are balanced",
        ),
        summary=InvestmentSummary(
            thesis="test thesis",
            key_drivers=["d1"],
            key_risks=["r1"],
        ),
        bull_case=["driver"],
        bear_case=["risk"],
        catalysts=[],
        scenario_analysis=[],
        agent_outputs=[],
        disagreement_summary=DisagreementSummary(main_conflict="valuation vs momentum"),
        risk_gate=RiskGateResult(status="pass", risk_level="low", risk_indicators=[]),
        consensus=ConsensusResultView(enabled=False),
        consensus_trace=ConsensusTrace(discussion_enabled=False),
        policy_overrides=PolicyOverrides(binding_cap="objective"),
        report_markdown="# test",
        metadata=InvestmentMetadata(
            token_usage={"prompt": 1, "completion": 1, "total": 2},
            latency_ms=10,
            data_sources=data_sources,
            selected_skills=selected_skills,
            task_type=task_type,
            constraints_applied_summary={"binding_cap": "objective"},
            schema_version="investment_analysis.v2",
        ),
    )


async def _collect_sse_events(stream_response) -> List[Dict[str, Any]]:
    chunks: List[str] = []
    async for part in stream_response.body_iterator:
        chunks.append(part.decode() if isinstance(part, bytes) else part)

    events: List[Dict[str, Any]] = []
    for chunk in chunks:
        for raw in chunk.split("\n\n"):
            raw = raw.strip()
            if not raw or not raw.startswith("data: "):
                continue
            events.append(json.loads(raw[len("data: ") :]))
    return events


@pytest.mark.asyncio
async def test_analyze_investment_success_returns_response() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.analyze_investment(_build_request())

    assert resp.request_id == "inv-test-001"
    assert resp.recommendation.action == RecommendationAction.HOLD
    assert resp.analysis_framework.task_type == "equity_analysis"


@pytest.mark.asyncio
async def test_analyze_investment_value_error_maps_to_400() -> None:
    investment_api._service = FakeInvestmentService("value_error")

    with pytest.raises(HTTPException) as exc:
        await investment_api.analyze_investment(_build_request())

    assert exc.value.status_code == 400
    assert "invalid request" in exc.value.detail


@pytest.mark.asyncio
async def test_analyze_investment_runtime_error_maps_to_500() -> None:
    investment_api._service = FakeInvestmentService("runtime_error")

    with pytest.raises(HTTPException) as exc:
        await investment_api.analyze_investment(_build_request())

    assert exc.value.status_code == 500
    assert "unexpected" in exc.value.detail


@pytest.mark.asyncio
async def test_analyze_stream_success_emits_result_and_end() -> None:
    investment_api._service = FakeInvestmentService("ok")

    stream_resp = await investment_api.analyze_investment_stream(_build_request())
    events = await _collect_sse_events(stream_resp)
    types = [evt["type"] for evt in events]

    assert "analysis_started" in types
    assert "agent_started" in types
    assert "constraints_applied" in types
    assert "result" in types
    assert types[-1] == "end"

    started_event = next(evt for evt in events if evt["type"] == "analysis_started")
    assert started_event["request_id"] == "inv-test-001"
    assert started_event["timestamp"] == "2026-04-09T10:00:00Z"

    result_event = next(evt for evt in events if evt["type"] == "result")
    summary = result_event["payload"]["metadata"]["constraints_applied_summary"]
    assert summary["binding_cap"] == "objective"


@pytest.mark.asyncio
async def test_analyze_stream_value_error_emits_error_and_end() -> None:
    investment_api._service = FakeInvestmentService("value_error")

    stream_resp = await investment_api.analyze_investment_stream(_build_request())
    events = await _collect_sse_events(stream_resp)

    assert events[-1]["type"] == "end"
    error_event = next(evt for evt in events if evt["type"] == "error")
    assert error_event["payload"]["code"] == 400
    assert error_event["request_id"] == "unknown"


@pytest.mark.asyncio
async def test_analyze_stream_runtime_error_emits_error_and_end() -> None:
    investment_api._service = FakeInvestmentService("runtime_error")

    stream_resp = await investment_api.analyze_investment_stream(_build_request())
    events = await _collect_sse_events(stream_resp)

    assert events[-1]["type"] == "end"
    error_event = next(evt for evt in events if evt["type"] == "error")
    assert error_event["payload"]["code"] == 500
    assert error_event["request_id"] == "unknown"


@pytest.mark.asyncio
async def test_get_analysis_result_returns_completed_response() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis("inv-test-001")

    assert resp.request_id == "inv-test-001"
    assert resp.status == "completed"


@pytest.mark.asyncio
async def test_get_analysis_status_returns_progress() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis_status("inv-test-001")

    assert resp["request_id"] == "inv-test-001"
    assert resp["status"] == "completed"
    assert resp["progress_pct"] == 100


@pytest.mark.asyncio
async def test_get_analysis_timeline_returns_events() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis_timeline("inv-test-001")

    assert resp["request_id"] == "inv-test-001"
    assert len(resp["timeline"]) >= 1


@pytest.mark.asyncio
async def test_get_analysis_discussion_returns_trace() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis_discussion("inv-test-001")

    assert resp["request_id"] == "inv-test-001"
    assert resp["enabled"] is True
    assert len(resp["rounds"]) == 1


@pytest.mark.asyncio
async def test_get_analysis_agents_returns_panel() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis_agents("inv-test-001")

    assert resp["request_id"] == "inv-test-001"
    assert resp["agents"][0]["agent_id"] == "agent_0"


@pytest.mark.asyncio
async def test_get_analysis_policy_overrides_returns_payload() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis_policy_overrides("inv-test-001")

    assert resp["policy_overrides"]["binding_cap"] == "objective"


@pytest.mark.asyncio
async def test_get_analysis_risk_gate_returns_payload() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.get_investment_analysis_risk_gate("inv-test-001")

    assert resp["risk_gate"]["status"] == "pass"


@pytest.mark.asyncio
async def test_get_analysis_returns_404_when_missing() -> None:
    investment_api._service = FakeInvestmentService("ok")

    with pytest.raises(HTTPException) as exc:
        await investment_api.get_investment_analysis("missing")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_analysis_returns_409_when_not_completed() -> None:
    service = FakeInvestmentService("ok")
    service.runs["inv-pending-001"] = {
        "status": "running",
        "timeline": [],
        "agents": [],
        "discussion": {"enabled": False, "final_summary": "", "rounds": []},
        "policy_overrides": {},
        "risk_gate": {},
        "result": None,
    }
    investment_api._service = service

    with pytest.raises(HTTPException) as exc:
        await investment_api.get_investment_analysis("inv-pending-001")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_analyze_investment_v3_options_returns_skill_metadata() -> None:
    investment_api._service = FakeInvestmentService("ok")

    resp = await investment_api.analyze_investment(_build_request(asset_type=AssetType.OPTIONS))

    assert resp.metadata.task_type == "options_analysis"
    assert "options_greeks" in resp.metadata.selected_skills
    assert "option_chain_feed" in resp.metadata.data_sources
