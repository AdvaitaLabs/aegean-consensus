"""
Challenge-Response Manager.

Implements the dynamic gatekeeper mechanism from Trustline:
- Issues structured challenges when a decision is RiskDecisionType.CHALLENGE
- Accepts evidence submissions
- Triggers re-evaluation via RiskConsensusCoordinator
- Enforces challenge TTL and max retry limits
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta, timezone
import logging

from aegean.risk.models import (
    ChallengeRequest,
    ChallengeResponse,
    ChallengeStatus,
    EvidenceType,
    RiskDecision,
    RiskDecisionType,
    RiskLevel,
)

logger = logging.getLogger(__name__)

DEFAULT_CHALLENGE_TTL_MINUTES = 30
MAX_CHALLENGES_PER_SESSION = 3


class ChallengeManager:
    """
    Manages challenge-response cycles for high-risk decisions.

    Flow:
    1. RiskConsensusCoordinator returns decision=CHALLENGE
    2. ChallengeManager.issue_challenge() creates a ChallengeRequest
    3. Caller submits evidence via submit_response()
    4. get_challenge_context() returns enriched context for re-evaluation
    5. After re-evaluation, resolve_challenge() closes the challenge

    All state is in-memory; production should use persistent storage.
    """

    def __init__(
        self,
        challenge_ttl_minutes: int = DEFAULT_CHALLENGE_TTL_MINUTES,
        max_challenges_per_session: int = MAX_CHALLENGES_PER_SESSION,
    ):
        self.challenge_ttl_minutes = challenge_ttl_minutes
        self.max_challenges_per_session = max_challenges_per_session
        self._challenges: Dict[str, ChallengeRequest] = {}
        self._responses: Dict[str, ChallengeResponse] = {}  # challenge_id -> response

    # ==================== Issue Challenge ====================

    def issue_challenge(
        self,
        decision: RiskDecision,
        session_challenge_count: int = 0,
    ) -> ChallengeRequest:
        """
        Issue a challenge request based on a CHALLENGE decision.

        Args:
            decision: The RiskDecision with decision=CHALLENGE
            session_challenge_count: How many challenges already issued in session

        Returns:
            ChallengeRequest to return to the caller

        Raises:
            ValueError: If decision is not CHALLENGE type or max retries exceeded
        """
        if decision.decision != RiskDecisionType.CHALLENGE:
            raise ValueError(
                f"Cannot issue challenge for decision type: {decision.decision}"
            )

        if session_challenge_count >= self.max_challenges_per_session:
            raise ValueError(
                f"Max challenges ({self.max_challenges_per_session}) exceeded for session"
            )

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.challenge_ttl_minutes)

        # Determine required evidence based on risk indicators
        required_evidence = self._determine_required_evidence(decision)
        instructions = self._generate_instructions(decision, required_evidence)

        challenge = ChallengeRequest(
            session_id=decision.session_id,
            decision_id=decision.decision_id,
            required_evidence=required_evidence,
            instructions=instructions,
            trigger_reasons=decision.risk_indicators[:5],  # top 5 signals
            issued_at=now,
            expires_at=expires_at,
        )

        self._challenges[challenge.challenge_id] = challenge
        logger.info(
            f"Challenge {challenge.challenge_id} issued for session {decision.session_id}. "
            f"Required evidence: {[e.value for e in required_evidence]}"
        )
        return challenge

    # ==================== Submit Response ====================

    def submit_response(
        self,
        challenge_id: str,
        evidence_type: EvidenceType,
        evidence_content: str,
        submitted_by: Optional[str] = None,
        evidence_metadata: Optional[dict] = None,
    ) -> ChallengeResponse:
        """
        Submit evidence in response to a challenge.

        Args:
            challenge_id: The challenge being responded to
            evidence_type: Type of evidence provided
            evidence_content: The actual evidence (text, JSON, etc.)
            submitted_by: ID of the submitter (user, agent, system)
            evidence_metadata: Additional metadata

        Returns:
            ChallengeResponse record

        Raises:
            KeyError: Challenge not found
            ValueError: Challenge already responded to or expired
        """
        challenge = self._challenges.get(challenge_id)
        if challenge is None:
            raise KeyError(f"Challenge {challenge_id} not found")

        now = datetime.now(timezone.utc)
        expires = challenge.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)

        if now > expires:
            challenge.status = ChallengeStatus.EXPIRED
            self._challenges[challenge_id] = challenge
            raise ValueError(f"Challenge {challenge_id} has expired")

        if challenge.status not in (ChallengeStatus.PENDING,):
            raise ValueError(
                f"Challenge {challenge_id} is in status {challenge.status}, "
                f"cannot accept response"
            )

        response = ChallengeResponse(
            challenge_id=challenge_id,
            session_id=challenge.session_id,
            evidence_type=evidence_type,
            evidence_content=evidence_content,
            evidence_metadata=evidence_metadata or {},
            submitted_at=now,
            submitted_by=submitted_by,
        )

        # Update challenge status
        challenge.status = ChallengeStatus.RESPONDED
        self._challenges[challenge_id] = challenge
        self._responses[challenge_id] = response

        logger.info(
            f"Response submitted for challenge {challenge_id}: "
            f"evidence_type={evidence_type.value}"
        )
        return response

    # ==================== Re-evaluation Context ====================

    def get_challenge_context(self, challenge_id: str) -> str:
        """
        Build an enriched context string for re-evaluation.

        This is injected into the RiskRequest.context.trace_context
        when the request is re-submitted to RiskConsensusCoordinator.

        Returns:
            Formatted string describing the challenge and submitted evidence
        """
        challenge = self._challenges.get(challenge_id)
        if not challenge:
            return ""

        response = self._responses.get(challenge_id)

        parts = [
            "[CHALLENGE CONTEXT]",
            f"This request was previously challenged due to: "
            f"{', '.join(challenge.trigger_reasons)}",
        ]

        if response:
            parts.append(
                f"Additional evidence submitted:"
                f"\n  Type: {response.evidence_type.value}"
                f"\n  Content: {response.evidence_content[:500]}"
                f"\n  Submitted at: {response.submitted_at.isoformat()}"
            )
        else:
            parts.append("No additional evidence provided yet.")

        return "\n".join(parts)

    # ==================== Resolve ====================

    def resolve_challenge(
        self,
        challenge_id: str,
        final_decision: RiskDecision,
    ) -> None:
        """
        Mark a challenge as resolved after re-evaluation.

        Args:
            challenge_id: Challenge to resolve
            final_decision: The new decision after re-evaluation
        """
        challenge = self._challenges.get(challenge_id)
        if challenge:
            challenge.status = ChallengeStatus.RESOLVED
            self._challenges[challenge_id] = challenge
            logger.info(
                f"Challenge {challenge_id} resolved: "
                f"new_decision={final_decision.decision.value} "
                f"risk_level={final_decision.risk_level.value}"
            )

    # ==================== Query ====================

    def get_challenge(self, challenge_id: str) -> Optional[ChallengeRequest]:
        """Get challenge by ID."""
        return self._challenges.get(challenge_id)

    def get_pending_challenges(self, session_id: str) -> List[ChallengeRequest]:
        """Get all pending challenges for a session."""
        return [
            c for c in self._challenges.values()
            if c.session_id == session_id
            and c.status == ChallengeStatus.PENDING
        ]

    # ==================== Internal Helpers ====================

    @staticmethod
    def _determine_required_evidence(
        decision: RiskDecision,
    ) -> List[EvidenceType]:
        """
        Determine what evidence to request based on risk indicators.
        """
        evidence: List[EvidenceType] = []
        indicators = set(decision.risk_indicators)

        # Identity-related signals → ask for identity proof
        if any(
            kw in ind
            for ind in indicators
            for kw in ["identity", "trust", "kya", "kya", "flag"]
        ):
            evidence.append(EvidenceType.IDENTITY_PROOF)

        # AML/compliance signals → ask for purpose
        if any(
            kw in ind
            for ind in indicators
            for kw in ["aml", "structur", "threshold", "compliance", "cross_border"]
        ):
            evidence.append(EvidenceType.PURPOSE_PROOF)
            evidence.append(EvidenceType.BUSINESS_JUSTIFICATION)

        # Anomaly signals → ask for transaction log
        if any(
            kw in ind
            for ind in indicators
            for kw in ["velocity", "anomal", "region", "geo"]
        ):
            evidence.append(EvidenceType.TRANSACTION_LOG)

        # Authorization signals
        if any(
            kw in ind
            for ind in indicators
            for kw in ["authorization", "consent", "injection"]
        ):
            evidence.append(EvidenceType.AUTHORIZATION)

        # Default: always ask for purpose if nothing matched
        if not evidence:
            evidence.append(EvidenceType.PURPOSE_PROOF)

        return list(dict.fromkeys(evidence))  # deduplicate, preserve order

    @staticmethod
    def _generate_instructions(
        decision: RiskDecision,
        required_evidence: List[EvidenceType],
    ) -> str:
        """Generate human-readable instructions for the challenge."""
        risk_summary = (
            f"Risk level: {decision.risk_level.value.upper()} "
            f"(confidence: {decision.confidence:.0%})"
        )
        evidence_list = "\n".join(
            f"  - {e.value.replace('_', ' ').title()}"
            for e in required_evidence
        )
        return (
            f"Your request requires additional verification before it can proceed.\n"
            f"{risk_summary}\n\n"
            f"Why this was flagged: {', '.join(decision.risk_indicators[:3]) or 'risk signals detected'}\n\n"
            f"Please re-submit supporting materials for review:\n"
            f"{evidence_list}\n\n"
            f"After submitting the materials, the system will automatically re-evaluate your request. "
            f"Please submit within {DEFAULT_CHALLENGE_TTL_MINUTES} minutes."
        )

