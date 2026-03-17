"""
Aegean Risk Assessment Module.

Financial risk evaluation subsystem inspired by Trustline VAN architecture.
Provides multi-validator consensus-based risk scoring for agent-initiated actions.

Quick start::

    from aegean.risk import RiskConsensusCoordinator, RiskRequest, RiskSubject, RiskContext

    coordinator = RiskConsensusCoordinator.create_default(memory_system=memory)

    request = RiskRequest(
        subject=RiskSubject(subject_id="user_001", subject_type="user", trust_score=0.8),
        context=RiskContext(
            action_type="payment",
            description="Transfer $2000 to supplier",
            amount=2000.0,
            currency="USD",
        )
    )
    decision = await coordinator.evaluate(request)
    print(decision.decision, decision.risk_level, decision.confidence)
"""

from aegean.risk.models import (
    RiskLevel,
    RiskDecisionType,
    ValidatorType,
    DifficultyLevel,
    SessionStatus,
    ChallengeStatus,
    EvidenceType,
    RiskSubject,
    RiskContext,
    RiskRequest,
    ValidatorResult,
    RiskDecision,
    RiskSession,
    ChallengeRequest,
    ChallengeResponse,
    SequencerDecision,
)
from aegean.risk.risk_consensus import RiskConsensusCoordinator
from aegean.risk.sequencer import Sequencer
from aegean.risk.session import SessionManager
from aegean.risk.challenge import ChallengeManager

__all__ = [
    # Models
    "RiskLevel",
    "RiskDecisionType",
    "ValidatorType",
    "DifficultyLevel",
    "SessionStatus",
    "ChallengeStatus",
    "EvidenceType",
    "RiskSubject",
    "RiskContext",
    "RiskRequest",
    "ValidatorResult",
    "RiskDecision",
    "RiskSession",
    "ChallengeRequest",
    "ChallengeResponse",
    "SequencerDecision",
    # Core components
    "RiskConsensusCoordinator",
    "Sequencer",
    "SessionManager",
    "ChallengeManager",
]

