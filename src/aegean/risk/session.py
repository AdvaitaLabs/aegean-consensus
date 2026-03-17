"""
Risk Session Manager.

Manages the lifecycle of risk evaluation sessions.
A session spans one full evaluation including potential challenge-response cycles.
Default TTL: 24 hours (matching Trustline default).
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta, timezone
import logging

from aegean.risk.models import (
    RiskSession,
    RiskDecision,
    SessionStatus,
    RiskRequest,
)

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_HOURS = 24


class SessionManager:
    """
    Manages risk evaluation sessions in memory.

    Responsibilities:
    - Create sessions for new requests
    - Retrieve existing sessions (for re-evaluation / challenge flow)
    - Attach decisions to sessions
    - Expire stale sessions
    - Provide session summary for audit trail

    In production this would be backed by Redis or a database.
    The in-memory implementation is sufficient for development
    and single-node deployments.
    """

    def __init__(self, session_ttl_hours: int = DEFAULT_SESSION_TTL_HOURS):
        self.session_ttl_hours = session_ttl_hours
        self._sessions: Dict[str, RiskSession] = {}

    # ==================== Session Lifecycle ====================

    def create_session(self, request: RiskRequest) -> RiskSession:
        """
        Create a new risk session for the given request.

        If the request already carries a session_id pointing to an
        active session, that session is returned instead (idempotent).
        """
        # Re-use existing active session if provided
        if request.session_id:
            existing = self._sessions.get(request.session_id)
            if existing and existing.status == SessionStatus.ACTIVE:
                logger.debug(f"Reusing existing session {existing.session_id}")
                return existing

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.session_ttl_hours)

        session = RiskSession(
            request_id=request.request_id,
            subject_id=request.subject.subject_id,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )

        self._sessions[session.session_id] = session
        logger.info(
            f"Created risk session {session.session_id} "
            f"for subject {session.subject_id} "
            f"(expires {expires_at.isoformat()})"
        )
        return session

    def get_session(self, session_id: str) -> Optional[RiskSession]:
        """Retrieve session by ID. Returns None if not found or expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        # Auto-expire check
        now = datetime.now(timezone.utc)
        expires = session.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)

        if now > expires and session.status == SessionStatus.ACTIVE:
            session.status = SessionStatus.EXPIRED
            self._sessions[session_id] = session
            logger.info(f"Session {session_id} auto-expired")

        return session

    def attach_decision(
        self, session_id: str, decision: RiskDecision
    ) -> RiskSession:
        """
        Attach a RiskDecision to a session and update current_decision_id.
        Also marks the session as CHALLENGED if the decision requires it.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        from aegean.risk.models import RiskDecisionType

        session.decisions.append(decision)
        session.current_decision_id = decision.decision_id
        session.updated_at = datetime.now(timezone.utc)

        if decision.decision == RiskDecisionType.CHALLENGE:
            session.status = SessionStatus.CHALLENGED
            session.challenge_count += 1
        else:
            session.status = SessionStatus.COMPLETED

        self._sessions[session_id] = session
        return session

    def complete_session(
        self, session_id: str, status: SessionStatus = SessionStatus.COMPLETED
    ) -> None:
        """Manually mark a session as completed/failed."""
        session = self._sessions.get(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now(timezone.utc)
            self._sessions[session_id] = session

    # ==================== Query ====================

    def list_sessions(
        self,
        subject_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> List[RiskSession]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())

        if subject_id:
            sessions = [s for s in sessions if s.subject_id == subject_id]
        if status:
            sessions = [s for s in sessions if s.status == status]

        # Newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    def get_stats(self) -> dict:
        """Return session statistics for monitoring."""
        all_sessions = list(self._sessions.values())
        by_status: Dict[str, int] = {}
        for s in all_sessions:
            key = s.status.value
            by_status[key] = by_status.get(key, 0) + 1

        return {
            "total_sessions": len(all_sessions),
            "by_status": by_status,
            "challenged_sessions": sum(
                1 for s in all_sessions if s.challenge_count > 0
            ),
        }

    # ==================== Maintenance ====================

    def purge_expired(self) -> int:
        """
        Remove expired sessions from memory.
        Returns count of purged sessions.
        Call periodically in production (e.g. background task).
        """
        now = datetime.now(timezone.utc)
        expired_ids = []

        for sid, session in self._sessions.items():
            expires = session.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if now > expires:
                expired_ids.append(sid)

        for sid in expired_ids:
            del self._sessions[sid]

        if expired_ids:
            logger.info(f"Purged {len(expired_ids)} expired sessions")

        return len(expired_ids)
