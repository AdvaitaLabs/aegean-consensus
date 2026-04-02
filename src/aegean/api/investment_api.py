"""FastAPI endpoints for investment analysis."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aegean.core.agent import AgentRegistry
from aegean.investment.models import InvestmentAnalysisRequest, InvestmentAnalysisResponse
from aegean.investment.service import InvestmentAnalysisService
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


def _to_sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"


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
                    "payload": result.model_dump(mode="json"),
                })
            except ValueError as exc:
                await queue.put({"type": "error", "payload": {"code": 400, "message": str(exc)}})
            except Exception as exc:
                await queue.put({"type": "error", "payload": {"code": 500, "message": str(exc)}})
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

        yield _to_sse({"type": "end", "payload": {"message": "stream_closed"}})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
