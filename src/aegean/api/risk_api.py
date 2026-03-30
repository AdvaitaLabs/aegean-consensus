"""
FastAPI endpoints for the Aegean Risk Assessment module.

Provides REST API for:
- Single risk evaluation  POST /api/v1/risk/evaluate
- Re-evaluation after challenge  POST /api/v1/risk/challenge/{id}/respond
- Session queries  GET /api/v1/risk/sessions/{id}
- Knowledge base seeding  POST /api/v1/risk/seed
- Validator stats  GET /api/v1/risk/stats
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from aegean.risk.models import (
    RiskSubject,
    RiskContext,
    RiskRequest,
    RiskDecision,
    RiskLevel,
    RiskDecisionType,
    EvidenceType,
    SessionStatus,
    TokenUsage,
)
from aegean.risk.risk_consensus import RiskConsensusCoordinator
from aegean.risk.session import SessionManager
from aegean.risk.challenge import ChallengeManager
from aegean.memory.global_memory import GlobalMemorySystem

router = APIRouter(prefix="/api/v1/risk", tags=["Risk Assessment"])

# Module-level singletons (initialized via init_risk_service)
_coordinator: Optional[RiskConsensusCoordinator] = None
_session_manager: Optional[SessionManager] = None
_challenge_manager: Optional[ChallengeManager] = None
_memory_system: Optional[GlobalMemorySystem] = None


def init_risk_service(
    memory_system: Optional[GlobalMemorySystem] = None,
    llm_client: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Initialize the risk service. Call this once at app startup."""
    global _coordinator, _session_manager, _challenge_manager, _memory_system

    _memory_system = memory_system or GlobalMemorySystem()
    _session_manager = SessionManager()
    _challenge_manager = ChallengeManager()

    _coordinator = RiskConsensusCoordinator.create_default(
        memory_system=_memory_system,
        llm_client=llm_client,
        config=config,
    )
    # Share session/challenge managers with coordinator
    _coordinator.session_manager = _session_manager
    _coordinator.challenge_manager = _challenge_manager


def _get_coordinator() -> RiskConsensusCoordinator:
    if _coordinator is None:
        init_risk_service()
    return _coordinator


# ==================== Request / Response Schemas ====================

class EvaluateRequest(BaseModel):
    """API request for risk evaluation."""
    subject_id: str = Field(..., description="Unique ID of subject being evaluated")
    subject_type: str = Field("user", description="'user', 'agent', 'transaction'")
    trust_score: float = Field(1.0, ge=0.0, le=1.0)
    total_transactions: int = Field(0, ge=0)
    flagged_count: int = Field(0, ge=0)
    jurisdiction: Optional[str] = None
    registered_at: Optional[datetime] = None

    action_type: str = Field(..., description="e.g. 'payment', 'withdrawal'")
    description: str = Field(..., description="Human-readable description")
    amount: Optional[float] = None
    currency: Optional[str] = None
    counterparty_id: Optional[str] = None
    geo_location: Optional[str] = None
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    channel: Optional[str] = None
    trace_context: Optional[str] = None
    recent_transaction_count: int = Field(0, ge=0)
    recent_transaction_amount: float = Field(0.0, ge=0.0)

    session_id: Optional[str] = None
    priority: str = Field("normal")
    debug_mode: bool = False
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "subject_id": "user_12345",
                "subject_type": "user",
                "trust_score": 0.75,
                "total_transactions": 42,
                "flagged_count": 0,
                "action_type": "payment",
                "description": "Transfer $3000 to supplier account",
                "amount": 3000.0,
                "currency": "USD",
                "geo_location": "NY,US",
                "channel": "web",
                "recent_transaction_count": 2,
                "recent_transaction_amount": 500.0,
            }
        }


class RiskDecisionResponse(BaseModel):
    """API response for risk evaluation."""
    decision_id: str
    request_id: str
    session_id: str
    decision: str
    risk_level: str
    confidence: float
    ttl: int
    rationale: str
    risk_indicators: List[str]
    challenge_eligible: bool
    difficulty_level: str
    participating_validators: List[str]
    weighted_votes: Dict[str, float] = {}
    rounds_used: int = 1
    validator_results: List[Dict[str, Any]] = []
    execution_time: float
    timestamp: datetime

    # Unified token usage
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_saved: int = 0
    provider: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

    # Challenge info (only present when decision=challenge)
    challenge_id: Optional[str] = None
    challenge_instructions: Optional[str] = None
    required_evidence: Optional[List[str]] = None


class ChallengeResponseRequest(BaseModel):
    """Submit evidence to resolve a challenge."""
    evidence_type: EvidenceType
    evidence_content: str = Field(..., description="The evidence content")
    submitted_by: Optional[str] = None
    evidence_metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "evidence_type": "purpose_proof",
                "evidence_content": "Payment for invoice #INV-2024-001 to supplier XYZ Corp",
                "submitted_by": "user_12345",
            }
        }


# ==================== Endpoints ====================

@router.post(
    "/evaluate",
    response_model=RiskDecisionResponse,
    summary="Evaluate risk for an action",
    description="Submit a risk evaluation request through the VAN pipeline.",
)
async def evaluate_risk(body: EvaluateRequest) -> RiskDecisionResponse:
    """
    Run a full risk assessment through the Aegean VAN pipeline.

    The Sequencer classifies complexity, activates the appropriate
    validator committee, runs parallel analysis, and returns a
    weighted-consensus decision.
    """
    coordinator = _get_coordinator()

    # Build domain objects from flat API request
    subject = RiskSubject(
        subject_id=body.subject_id,
        subject_type=body.subject_type,
        trust_score=body.trust_score,
        total_transactions=body.total_transactions,
        flagged_count=body.flagged_count,
        jurisdiction=body.jurisdiction,
        registered_at=body.registered_at,
    )
    context = RiskContext(
        action_type=body.action_type,
        description=body.description,
        amount=body.amount,
        currency=body.currency,
        counterparty_id=body.counterparty_id,
        geo_location=body.geo_location,
        ip_address=body.ip_address,
        device_id=body.device_id,
        channel=body.channel,
        trace_context=body.trace_context,
        recent_transaction_count=body.recent_transaction_count,
        recent_transaction_amount=body.recent_transaction_amount,
        metadata=body.metadata or {},
    )
    request = RiskRequest(
        subject=subject,
        context=context,
        session_id=body.session_id,
        priority=body.priority,
        debug_mode=body.debug_mode,
    )

    try:
        decision = await coordinator.evaluate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(decision)


@router.post(
    "/challenge/{challenge_id}/respond",
    response_model=RiskDecisionResponse,
    summary="Respond to a challenge and re-evaluate",
)
async def respond_to_challenge(
    challenge_id: str,
    body: ChallengeResponseRequest,
) -> RiskDecisionResponse:
    """
    Submit evidence in response to a challenge, then trigger re-evaluation.

    On success the original request is re-evaluated with the submitted
    evidence injected into trace_context.
    """
    coordinator = _get_coordinator()
    challenge_mgr = coordinator.challenge_manager

    # Verify challenge exists
    challenge = challenge_mgr.get_challenge(challenge_id)
    if not challenge:
        raise HTTPException(status_code=404, detail=f"Challenge {challenge_id} not found")

    # Submit response
    try:
        challenge_mgr.submit_response(
            challenge_id=challenge_id,
            evidence_type=body.evidence_type,
            evidence_content=body.evidence_content,
            submitted_by=body.submitted_by,
            evidence_metadata=body.evidence_metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Retrieve original session + request to re-evaluate
    session = coordinator.session_manager.get_session(challenge.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Rebuild minimal request from session metadata for re-evaluation
    # (In production the original request would be persisted; here we
    #  reconstruct from the session's stored request_id metadata)
    original_request = _rebuild_request_from_session(session, challenge_id, coordinator)
    if not original_request:
        raise HTTPException(
            status_code=422,
            detail="Cannot reconstruct original request for re-evaluation"
        )

    try:
        new_decision = await coordinator.re_evaluate(
            original_request=original_request,
            challenge_id=challenge_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(new_decision)


@router.get(
    "/sessions/{session_id}",
    summary="Get risk session details",
)
async def get_session(session_id: str) -> Dict[str, Any]:
    """Get full details of a risk evaluation session including all decisions."""
    coordinator = _get_coordinator()
    session = coordinator.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {
        "session_id": session.session_id,
        "subject_id": session.subject_id,
        "status": session.status.value,
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "challenge_count": session.challenge_count,
        "decision_count": len(session.decisions),
        "current_decision_id": session.current_decision_id,
        "decisions": [
            {
                "decision_id": d.decision_id,
                "decision": d.decision.value,
                "risk_level": d.risk_level.value,
                "confidence": d.confidence,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in session.decisions
        ],
    }


@router.get(
    "/sessions",
    summary="List risk sessions",
)
async def list_sessions(
    subject_id: Optional[str] = None,
    status: Optional[SessionStatus] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """List risk sessions with optional filters."""
    coordinator = _get_coordinator()
    sessions = coordinator.session_manager.list_sessions(
        subject_id=subject_id,
        status=status,
        limit=limit,
    )
    return {
        "total": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "subject_id": s.subject_id,
                "status": s.status.value,
                "challenge_count": s.challenge_count,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ],
    }


@router.get(
    "/stats",
    summary="Validator and session statistics",
)
async def get_stats() -> Dict[str, Any]:
    """Return validator performance stats and session summary."""
    coordinator = _get_coordinator()
    validator_stats = [
        v.get_stats() for v in coordinator.validators.values()
    ]
    session_stats = coordinator.session_manager.get_stats()
    return {
        "validators": validator_stats,
        "sessions": session_stats,
    }


@router.post(
    "/seed",
    summary="Seed risk knowledge base",
    description="Load public-domain financial risk knowledge into the RAG knowledge base.",
)
async def seed_knowledge_base(
    background_tasks: BackgroundTasks,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Seed the knowledge base with public-domain financial risk data.
    Runs in background. Set force=true to re-seed even if data exists.
    """
    coordinator = _get_coordinator()
    if not coordinator.memory_system:
        raise HTTPException(
            status_code=400,
            detail="No memory system configured. Initialize with memory_system."
        )

    from aegean.risk.data_seed import RiskKnowledgeSeeder
    seeder = RiskKnowledgeSeeder(coordinator.memory_system)

    async def _do_seed():
        count = await seeder.seed_all(skip_if_exists=not force)
        return count

    background_tasks.add_task(_do_seed)
    return {"status": "seeding_started", "message": "Knowledge base seeding in background"}


# ==================== Helpers ====================

def _aggregate_risk_token_usage(decision: RiskDecision) -> Dict[str, Any]:
    """Aggregate validator token usage into a unified response payload."""
    prompt = 0
    completion = 0
    saved = 0
    provider_counts: Dict[str, int] = {}

    for vr in decision.validator_results:
        meta = vr.metadata or {}
        usage = TokenUsage.from_raw(meta.get("usage") or meta)
        prompt += usage.tokens_prompt
        completion += usage.tokens_completion
        saved += int(meta.get("tokens_saved") or 0)

        p = meta.get("provider")
        if p:
            provider_counts[p] = provider_counts.get(p, 0) + 1

    provider = None
    if provider_counts:
        provider = max(provider_counts, key=provider_counts.get)

    usage_obj = TokenUsage(
        tokens_prompt=prompt,
        tokens_completion=completion,
        tokens_total=prompt + completion,
    )
    return {
        "tokens_prompt": usage_obj.tokens_prompt,
        "tokens_completion": usage_obj.tokens_completion,
        "tokens_saved": max(saved, 0),
        "provider": provider,
        "usage": usage_obj.model_dump(),
    }


def _build_response(decision: RiskDecision) -> RiskDecisionResponse:
    """Convert RiskDecision domain object to API response."""
    token_payload = _aggregate_risk_token_usage(decision)
    resp = RiskDecisionResponse(
        decision_id=decision.decision_id,
        request_id=decision.request_id,
        session_id=decision.session_id,
        decision=decision.decision.value,
        risk_level=decision.risk_level.value,
        confidence=decision.confidence,
        ttl=decision.ttl,
        rationale=decision.rationale,
        risk_indicators=decision.risk_indicators,
        challenge_eligible=decision.challenge_eligible,
        difficulty_level=decision.difficulty_level.value,
        participating_validators=decision.participating_validators,
        weighted_votes=decision.weighted_votes,
        rounds_used=decision.rounds_used,
        validator_results=[
            {
                "validator_type": vr.validator_type.value,
                "validator_id": vr.validator_id,
                "risk_level": vr.risk_level.value,
                "confidence": vr.confidence,
                "weight": vr.weight,
                "reasoning": vr.reasoning,
                "risk_indicators": vr.risk_indicators,
                "tokens_prompt": int((vr.metadata or {}).get("tokens_prompt") or 0),
                "tokens_completion": int((vr.metadata or {}).get("tokens_completion") or 0),
                "usage": (vr.metadata or {}).get("usage"),
                "provider": (vr.metadata or {}).get("provider"),
            }
            for vr in decision.validator_results
        ],
        execution_time=decision.execution_time,
        timestamp=decision.timestamp,
        tokens_prompt=token_payload["tokens_prompt"],
        tokens_completion=token_payload["tokens_completion"],
        tokens_saved=token_payload["tokens_saved"],
        provider=token_payload["provider"],
        usage=token_payload["usage"],
    )

    # If challenge decision, auto-issue challenge and include in response
    if decision.decision == RiskDecisionType.CHALLENGE and decision.challenge_eligible:
        try:
            coordinator = _get_coordinator()
            session = coordinator.session_manager.get_session(decision.session_id)
            challenge_count = session.challenge_count if session else 0
            challenge = coordinator.challenge_manager.issue_challenge(
                decision=decision,
                session_challenge_count=challenge_count,
            )
            resp.challenge_id = challenge.challenge_id
            resp.challenge_instructions = challenge.instructions
            resp.required_evidence = [e.value for e in challenge.required_evidence]
        except Exception:
            pass  # Challenge issuance failure doesn't block response

    return resp


def _rebuild_request_from_session(
    session,
    challenge_id: str,
    coordinator: RiskConsensusCoordinator,
) -> Optional[RiskRequest]:
    """
    Reconstruct a minimal RiskRequest for re-evaluation.

    In production the original request would be stored in the session.
    Here we build from the session's last decision metadata.
    """
    if not session.decisions:
        return None

    last_decision = session.decisions[-1]
    meta = last_decision.metadata

    subject = RiskSubject(
        subject_id=session.subject_id,
        subject_type=meta.get("subject_type", "user"),
        trust_score=meta.get("trust_score", 1.0),
    )
    context = RiskContext(
        action_type=meta.get("action_type", "unknown"),
        description=meta.get("description", "Re-evaluation after challenge"),
        amount=meta.get("amount"),
        currency=meta.get("currency"),
        trace_context=coordinator.challenge_manager.get_challenge_context(challenge_id),
    )
    return RiskRequest(
        subject=subject,
        context=context,
        session_id=session.session_id,
        priority="urgent",
    )

