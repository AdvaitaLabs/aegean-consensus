"""
Risk assessment data models.

Defines the core data structures for the financial risk evaluation system,
inspired by the Trustline VAN (Verification Agent Network) architecture.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# ==================== Enums ====================

class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"           # Normal, approve
    MEDIUM = "medium"     # Caution, enhanced review
    HIGH = "high"         # Suspicious, human review recommended
    CRITICAL = "critical" # Block, reject immediately


class RiskDecisionType(str, Enum):
    """Final risk decision."""
    APPROVE = "approve"     # Transaction/action approved
    REJECT = "reject"       # Transaction/action rejected
    CHALLENGE = "challenge" # Requires additional evidence
    REVIEW = "review"       # Escalate to human review


class ValidatorType(str, Enum):
    """Specialist validator committee types (mirrors Trustline VAN specializations)."""
    IDENTITY = "identity"         # KYA, identity & credential verification
    ANOMALY = "anomaly"           # Behavioral and contextual anomaly detection
    COMPLIANCE = "compliance"     # Regulatory & AML compliance
    AMOUNT = "amount"             # Amount, frequency & velocity checks
    CONTEXT = "context"           # Reasoning trace & contextual analysis


class DifficultyLevel(str, Enum):
    """Request complexity level for sequencer routing."""
    SIMPLE = "simple"     # 2-3 validators, 1 round
    MEDIUM = "medium"     # 3-4 validators, 2 rounds
    HARD = "hard"         # All validators, multi-round + possible challenge


class SessionStatus(str, Enum):
    """Risk session lifecycle status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    CHALLENGED = "challenged"
    EXPIRED = "expired"
    FAILED = "failed"


class ChallengeStatus(str, Enum):
    """Challenge-response status."""
    PENDING = "pending"         # Challenge issued, awaiting response
    RESPONDED = "responded"     # Response submitted
    RE_EVALUATING = "re_evaluating"
    RESOLVED = "resolved"
    EXPIRED = "expired"


# ==================== Core Request/Response Models ====================

class RiskSubject(BaseModel):
    """
    The entity being evaluated (agent, user, transaction, etc.).
    
    This is the 'who' of a risk assessment request.
    """
    subject_id: str = Field(..., description="Unique identifier of the subject")
    subject_type: str = Field(..., description="Type: 'agent', 'user', 'transaction', 'payment'")
    
    # Identity attributes
    name: Optional[str] = None
    registered_at: Optional[datetime] = None
    jurisdiction: Optional[str] = None  # e.g. "CN", "US", "EU"
    
    # Behavioral history summary
    total_transactions: int = Field(0, description="Lifetime transaction count")
    flagged_count: int = Field(0, description="Previously flagged events")
    trust_score: float = Field(1.0, ge=0.0, le=1.0, description="Accumulated trust score")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskContext(BaseModel):
    """
    Full context of the risk evaluation request.
    
    Mirrors Trustline's trace_context + payment metadata combined.
    """
    # What is being evaluated
    action_type: str = Field(..., description="e.g. 'payment', 'credit_application', 'withdrawal'")
    description: str = Field(..., description="Human-readable description of the action")
    
    # Financial specifics
    amount: Optional[float] = Field(None, description="Transaction amount")
    currency: Optional[str] = Field(None, description="Currency code, e.g. 'USD', 'CNY'")
    counterparty_id: Optional[str] = Field(None, description="Recipient/counterparty ID")
    
    # Behavioral signals
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    geo_location: Optional[str] = None
    channel: Optional[str] = None  # "mobile", "web", "api"
    
    # Reasoning trace (mirrors Trustline trace_context)
    # The chain of reasoning or agent actions that led to this request
    trace_context: Optional[str] = Field(
        None,
        description="Agent reasoning trace or user action chain that led here"
    )
    
    # Historical velocity (recent activity)
    recent_transaction_count: int = Field(0, description="Transactions in last hour")
    recent_transaction_amount: float = Field(0.0, description="Total amount in last hour")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskRequest(BaseModel):
    """
    Top-level risk assessment request.
    
    Entry point into the Aegean risk evaluation pipeline.
    """
    request_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}",
        description="Unique request identifier"
    )
    session_id: Optional[str] = Field(
        None,
        description="Existing session ID (for re-evaluation/challenge flow)"
    )
    
    subject: RiskSubject = Field(..., description="Entity being evaluated")
    context: RiskContext = Field(..., description="Action context")
    
    # Request metadata
    requested_by: Optional[str] = Field(None, description="System or user requesting evaluation")
    priority: str = Field("normal", description="'low', 'normal', 'high', 'urgent'")
    debug_mode: bool = Field(False, description="Dry-run without real action execution")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "subject": {
                    "subject_id": "user_12345",
                    "subject_type": "user",
                    "trust_score": 0.85
                },
                "context": {
                    "action_type": "payment",
                    "description": "Transfer $5000 to overseas account",
                    "amount": 5000.0,
                    "currency": "USD",
                    "geo_location": "SH,CN",
                    "recent_transaction_count": 3
                }
            }
        }


# ==================== Validator Output ====================

class ValidatorResult(BaseModel):
    """
    Output from a single specialist validator.
    
    Each validator committee returns one of these after analyzing
    the risk request from their specialized perspective.
    """
    validator_type: ValidatorType
    validator_id: str = Field(..., description="Unique validator instance ID")
    
    # Core assessment
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this assessment")
    
    # Reasoning (visible trace, mirrors Trustline's reasoning tooltip)
    reasoning: str = Field(..., description="Detailed reasoning for this assessment")
    risk_indicators: List[str] = Field(
        default_factory=list,
        description="Specific risk signals identified"
    )
    
    # Weight for consensus (set by WeightedDecisionEngine)
    weight: float = Field(1.0, ge=0.0, description="Voting weight in consensus")
    
    # Evidence links
    knowledge_refs: List[str] = Field(
        default_factory=list,
        description="Knowledge base document IDs referenced"
    )
    case_refs: List[str] = Field(
        default_factory=list,
        description="Similar historical case IDs referenced"
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Consensus Output ====================

class RiskDecision(BaseModel):
    """
    Final risk decision produced by the RiskConsensusCoordinator.
    
    Mirrors Trustline's decision object:
    - decision, risk_level, confidence, ttl, rationale,
      challenge_eligible, liability_map
    """
    decision_id: str = Field(
        default_factory=lambda: f"dec-{uuid.uuid4().hex[:12]}"
    )
    request_id: str
    session_id: str
    
    # Core decision
    decision: RiskDecisionType
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Validity window (TTL in seconds, like Trustline)
    ttl: int = Field(3600, description="Decision validity in seconds")
    expires_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Human-readable explanation
    rationale: str = Field(..., description="Summary of validator reasoning")
    risk_indicators: List[str] = Field(
        default_factory=list,
        description="Aggregated risk signals from all validators"
    )
    
    # Validator details
    validator_results: List[ValidatorResult] = Field(
        default_factory=list,
        description="Individual validator outputs"
    )
    weighted_votes: Dict[str, float] = Field(
        default_factory=dict,
        description="risk_level -> aggregated weighted score"
    )
    
    # Challenge eligibility
    challenge_eligible: bool = Field(
        False,
        description="Whether this decision can be challenged with new evidence"
    )
    
    # Execution metadata
    difficulty_level: DifficultyLevel = Field(DifficultyLevel.SIMPLE)
    rounds_used: int = Field(1)
    participating_validators: List[str] = Field(default_factory=list)
    execution_time: float = Field(0.0)
    
    # Future: liability distribution map
    liability_map: Dict[str, Any] = Field(
        default_factory=dict,
        description="Role-based liability distribution (future use)"
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "approve",
                "risk_level": "low",
                "confidence": 0.92,
                "ttl": 3600,
                "rationale": "All validators agree: normal transaction pattern, verified identity",
                "challenge_eligible": False
            }
        }


# ==================== Session ====================

class RiskSession(BaseModel):
    """
    Risk evaluation session.
    
    A session spans the full lifecycle of a risk assessment,
    including potential challenge-response cycles.
    Default TTL: 24 hours (matching Trustline default).
    """
    session_id: str = Field(
        default_factory=lambda: f"sess-{uuid.uuid4().hex[:12]}"
    )
    request_id: str
    subject_id: str
    
    status: SessionStatus = Field(SessionStatus.ACTIVE)
    
    # Timeline
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # default 24h, set by SessionManager
    
    # Decisions made in this session (supports re-evaluation)
    decisions: List[RiskDecision] = Field(default_factory=list)
    current_decision_id: Optional[str] = None
    
    # Challenge history
    challenge_count: int = Field(0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Challenge-Response ====================

class EvidenceType(str, Enum):
    """Types of additional evidence that can satisfy a challenge."""
    PURPOSE_PROOF = "purpose_proof"       # Proof of payment purpose
    IDENTITY_PROOF = "identity_proof"     # Additional identity verification
    AUTHORIZATION = "authorization"       # Explicit authorization/consent
    TRANSACTION_LOG = "transaction_log"   # Detailed transaction history
    BUSINESS_JUSTIFICATION = "business_justification"
    OTHER = "other"


class ChallengeRequest(BaseModel):
    """
    Challenge issued to requester when decision is 'challenge'.
    
    The system pauses execution and asks for additional evidence.
    """
    challenge_id: str = Field(
        default_factory=lambda: f"chal-{uuid.uuid4().hex[:12]}"
    )
    session_id: str
    decision_id: str
    
    status: ChallengeStatus = Field(ChallengeStatus.PENDING)
    
    # What evidence is required
    required_evidence: List[EvidenceType] = Field(
        ...,
        description="Types of evidence required to resolve the challenge"
    )
    instructions: str = Field(
        ...,
        description="Human-readable instructions for what to provide"
    )
    
    # Why the challenge was issued
    trigger_reasons: List[str] = Field(
        default_factory=list,
        description="Risk indicators that triggered this challenge"
    )
    
    # Timing
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # Challenge must be responded to before this
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChallengeResponse(BaseModel):
    """
    Response submitted by requester to resolve a challenge.
    """
    response_id: str = Field(
        default_factory=lambda: f"resp-{uuid.uuid4().hex[:12]}"
    )
    challenge_id: str
    session_id: str
    
    # Submitted evidence
    evidence_type: EvidenceType
    evidence_content: str = Field(..., description="The actual evidence content")
    evidence_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_by: Optional[str] = None


# ==================== Sequencer Output ====================

class SequencerDecision(BaseModel):
    """
    Output from the Sequencer - routing decision for incoming request.
    """
    request_id: str
    difficulty_level: DifficultyLevel
    
    # Which validator types to activate
    active_validators: List[ValidatorType]
    
    # Consensus config for this difficulty level
    quorum_threshold: float = Field(0.5)
    max_rounds: int = Field(1)
    stability_horizon: int = Field(1)
    
    # Routing reasoning
    routing_reasons: List[str] = Field(default_factory=list)
    risk_signals_detected: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

