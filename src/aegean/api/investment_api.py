"""FastAPI endpoints for investment analysis."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aegean.core.agent import AgentRegistry
from aegean.investment.masters import MASTER_PERSONAS
from aegean.investment.models import InvestmentAnalysisRequest, InvestmentAnalysisResponse
from aegean.investment.service import (
    InvestmentAnalysisService,
    _EXTENDED_LENS_ROLES,
    _EQUITY_PANEL_ROLES,
    _TITLE_BY_ROLE,
    _ROLE_DATA_FOCUS,
)
from aegean.memory.global_memory import GlobalMemorySystem
from aegean.risk.risk_consensus import RiskConsensusCoordinator

router = APIRouter(prefix="/api/v1/investment", tags=["Investment Analysis"])

_service: Optional[InvestmentAnalysisService] = None


def init_investment_service(
    agent_registry: AgentRegistry,
    memory_system: Optional[GlobalMemorySystem] = None,
    llm_client: Optional[Any] = None,
    risk_coordinator: Optional[RiskConsensusCoordinator] = None,
) -> None:
    global _service
    _service = InvestmentAnalysisService(
        agent_registry=agent_registry,
        memory_system=memory_system,
        llm_client=llm_client,
        risk_coordinator=risk_coordinator,
    )


def _get_service() -> InvestmentAnalysisService:
    if _service is None:
        raise HTTPException(status_code=500, detail="Investment service not initialized")
    return _service


def _get_analysis_run_or_404(service: InvestmentAnalysisService, request_id: str) -> dict:
    run = service.get_analysis_run(request_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Investment analysis {request_id} not found")
    return run


def _to_sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"


@router.get(
    "/roles/catalog",
    summary="List base panel roles, extended lens roles, and master personas",
)
async def get_roles_catalog() -> dict:
    def _entry(role: str) -> dict:
        return {
            "id": role,
            "title": _TITLE_BY_ROLE.get(role, role),
            "data_focus": list(_ROLE_DATA_FOCUS.get(role, [])),
        }

    return {
        "base_panel_roles": [_entry(r) for r in _EQUITY_PANEL_ROLES],
        "extended_lens_roles": [_entry(r) for r in sorted(_EXTENDED_LENS_ROLES)],
        "master_personas": [
            {
                "id": persona.key,
                "alias": f"{persona.key}_style",
                "display_name": persona.display_name,
                "philosophy": persona.philosophy,
                "signature_lens": persona.signature_lens,
                "output_bias": persona.output_bias,
            }
            for persona in MASTER_PERSONAS.values()
        ],
    }


@router.post(
    "/analyze",
    response_model=InvestmentAnalysisResponse,
    summary="Run investment analysis",
)
async def analyze_investment(body: InvestmentAnalysisRequest) -> InvestmentAnalysisResponse:
    service = _get_service()
    try:
        return await service.analyze(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/analyses/{request_id}",
    response_model=InvestmentAnalysisResponse,
    summary="Get investment analysis result by request id",
)
async def get_investment_analysis(request_id: str) -> InvestmentAnalysisResponse:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    result = run.get("result")
    if not result:
        raise HTTPException(status_code=409, detail=f"Investment analysis {request_id} is not completed yet")
    return InvestmentAnalysisResponse.model_validate(result)


@router.get(
    "/analyses/{request_id}/status",
    summary="Get investment analysis status",
)
async def get_investment_analysis_status(request_id: str) -> dict:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    timeline = run.get("timeline") or []
    current_stage = timeline[-1]["type"] if timeline else "pending"
    status = run.get("status", "running")
    progress_map = {
        "pending": 0,
        "running": 25,
        "roundtable": 65,
        "completed": 100,
    }
    return {
        "request_id": request_id,
        "status": status,
        "current_stage": current_stage,
        "progress_pct": progress_map.get(status, 50),
    }


@router.get(
    "/analyses/{request_id}/timeline",
    summary="Get investment analysis timeline",
)
async def get_investment_analysis_timeline(request_id: str) -> dict:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    return {
        "request_id": request_id,
        "status": run.get("status", "running"),
        "timeline": run.get("timeline", []),
    }


@router.get(
    "/analyses/{request_id}/discussion",
    summary="Get investment analysis discussion trace",
)
async def get_investment_analysis_discussion(request_id: str) -> dict:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    discussion = run.get("discussion") or {"enabled": False, "final_summary": "", "rounds": []}
    return {
        "request_id": request_id,
        **discussion,
    }


@router.get(
    "/analyses/{request_id}/agents",
    summary="Get investment analysis agent panel",
)
async def get_investment_analysis_agents(request_id: str) -> dict:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    return {
        "request_id": request_id,
        "agents": run.get("agents", []),
    }


@router.get(
    "/analyses/{request_id}/policy-overrides",
    summary="Get investment analysis policy overrides",
)
async def get_investment_analysis_policy_overrides(request_id: str) -> dict:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    return {
        "request_id": request_id,
        "policy_overrides": run.get("policy_overrides", {}),
    }


@router.get(
    "/analyses/{request_id}/risk-gate",
    summary="Get investment analysis risk gate result",
)
async def get_investment_analysis_risk_gate(request_id: str) -> dict:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    return {
        "request_id": request_id,
        "risk_gate": run.get("risk_gate", {}),
    }


@router.post(
    "/analyses",
    summary="Create investment analysis request",
)
async def create_investment_analysis(body: InvestmentAnalysisRequest) -> dict:
    service = _get_service()
    request_id = service.create_analysis_run(body)
    return {
        "request_id": request_id,
        "status": "accepted",
    }


@router.get(
    "/analyses/{request_id}/stream",
    summary="Stream investment analysis by request id",
)
async def stream_investment_analysis(request_id: str) -> StreamingResponse:
    service = _get_service()
    run = _get_analysis_run_or_404(service, request_id)
    request_payload = run.get("request")
    if not request_payload:
        raise HTTPException(status_code=409, detail=f"Investment analysis {request_id} request payload missing")
    body = InvestmentAnalysisRequest.model_validate(request_payload)

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        async def emit(event: dict) -> None:
            await queue.put(event)

        async def run_analysis() -> None:
            try:
                result = await service.analyze(body, event_sink=emit, request_id=request_id)
                await queue.put({
                    "type": "result",
                    "timestamp": None,
                    "request_id": result.request_id,
                    "payload": result.model_dump(mode="json"),
                })
            except ValueError as exc:
                await queue.put({
                    "type": "error",
                    "timestamp": None,
                    "request_id": request_id,
                    "payload": {"code": 400, "message": str(exc)},
                })
            except Exception as exc:
                await queue.put({
                    "type": "error",
                    "timestamp": None,
                    "request_id": request_id,
                    "payload": {"code": 500, "message": str(exc)},
                })
            finally:
                done.set()

        asyncio.create_task(run_analysis())

        while True:
            if done.is_set() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
                yield _to_sse(event)
            except asyncio.TimeoutError:
                continue

        yield _to_sse({"type": "end", "timestamp": None, "request_id": request_id, "payload": {"message": "stream_closed"}})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/analyze/stream",
    summary="Run investment analysis with SSE progress events",
)
async def analyze_investment_stream(body: InvestmentAnalysisRequest) -> StreamingResponse:
    service = _get_service()

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        async def emit(event: dict) -> None:
            await queue.put(event)

        async def run_analysis() -> None:
            try:
                result = await service.analyze(body, event_sink=emit)
                await queue.put({
                    "type": "result",
                    "timestamp": None,
                    "request_id": result.request_id,
                    "payload": result.model_dump(mode="json"),
                })
            except ValueError as exc:
                await queue.put({
                    "type": "error",
                    "timestamp": None,
                    "request_id": "unknown",
                    "payload": {"code": 400, "message": str(exc)},
                })
            except Exception as exc:
                await queue.put({
                    "type": "error",
                    "timestamp": None,
                    "request_id": "unknown",
                    "payload": {"code": 500, "message": str(exc)},
                })
            finally:
                done.set()

        asyncio.create_task(run_analysis())

        while True:
            if done.is_set() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
                yield _to_sse(event)
            except asyncio.TimeoutError:
                continue

        yield _to_sse({"type": "end", "timestamp": None, "request_id": "stream", "payload": {"message": "stream_closed"}})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
