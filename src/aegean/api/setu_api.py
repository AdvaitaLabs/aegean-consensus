"""FastAPI endpoints for Setu governance subnet integration."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from aegean.services.setu_service import SetuService
from aegean.setu_models import (
    EvaluateAcceptedResponse,
    SetuEvaluateRequest,
    SetuResultResponse,
)

router = APIRouter(tags=["Setu Governance Adapter"])

_service: Optional[SetuService] = None


def init_setu_service(service: SetuService) -> SetuService:
    """Initialize the Setu adapter service singleton."""
    global _service
    _service = service
    return _service


def get_service() -> SetuService:
    if _service is None:
        raise HTTPException(status_code=500, detail="SetuService not initialized")
    return _service


@router.post(
    "/evaluate",
    response_model=EvaluateAcceptedResponse,
    summary="Accept Setu governance proposal evaluation task",
)
async def evaluate(body: SetuEvaluateRequest) -> EvaluateAcceptedResponse:
    service = get_service()
    try:
        return await service.submit_evaluation(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/result/{task_id}",
    response_model=SetuResultResponse,
    summary="Poll Setu governance proposal result by task id",
)
async def get_result(task_id: str) -> SetuResultResponse:
    service = get_service()
    return service.get_result(task_id)


@router.get(
    "/setu/binding",
    summary="Inspect Setu subnet to group binding",
)
async def get_binding(subnet_id: Optional[str] = None):
    service = get_service()
    try:
        return service.get_bound_group(subnet_id=subnet_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

