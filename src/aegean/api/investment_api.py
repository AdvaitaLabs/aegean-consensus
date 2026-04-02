"""FastAPI endpoints for investment analysis."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

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


@router.post(
    "/analyze",
    response_model=InvestmentAnalysisResponse,
    summary="Run investment analysis",
)
async def analyze_investment(body: InvestmentAnalysisRequest) -> InvestmentAnalysisResponse:
    service = _get_service()
    try:
        return await service.analyze(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

