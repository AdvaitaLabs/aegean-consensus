"""
Sequencer - Request complexity classifier and validator router.

Mirrors Trustline's Sequencer component:
- Classifies incoming risk requests by difficulty
- Routes to appropriate validator committee subset
- Configures consensus parameters per difficulty level

Routing logic:
  SIMPLE  → [AMOUNT, IDENTITY]            1 round, threshold=0.5
  MEDIUM  → [AMOUNT, IDENTITY, ANOMALY]   2 rounds, threshold=0.55
  HARD    → all 5 validators              3 rounds, threshold=0.6 + challenge eligible
"""

from typing import List, Tuple
import logging

from aegean.risk.models import (
    RiskRequest,
    RiskLevel,
    ValidatorType,
    DifficultyLevel,
    SequencerDecision,
)

logger = logging.getLogger(__name__)

# Validator sets per difficulty level
SIMPLE_VALIDATORS: List[ValidatorType] = [
    ValidatorType.AMOUNT,
    ValidatorType.IDENTITY,
]

MEDIUM_VALIDATORS: List[ValidatorType] = [
    ValidatorType.AMOUNT,
    ValidatorType.IDENTITY,
    ValidatorType.ANOMALY,
]

HARD_VALIDATORS: List[ValidatorType] = [
    ValidatorType.AMOUNT,
    ValidatorType.IDENTITY,
    ValidatorType.ANOMALY,
    ValidatorType.COMPLIANCE,
    ValidatorType.CONTEXT,
]

# Consensus config per difficulty
DIFFICULTY_CONFIG = {
    DifficultyLevel.SIMPLE: {
        "quorum_threshold": 0.50,
        "max_rounds": 1,
        "stability_horizon": 1,
    },
    DifficultyLevel.MEDIUM: {
        "quorum_threshold": 0.55,
        "max_rounds": 2,
        "stability_horizon": 1,
    },
    DifficultyLevel.HARD: {
        "quorum_threshold": 0.60,
        "max_rounds": 3,
        "stability_horizon": 2,
    },
}


class Sequencer:
    """
    Classifies risk requests and routes them to the right validator committee.

    Scoring logic (additive):
    - Amount signals    → +1 to +3 points
    - Identity signals  → +1 to +3 points
    - Velocity signals  → +1 to +2 points
    - Trace context     → +1 point (missing = +1 penalty)
    - Priority override → direct HARD routing

    Score 0-2  → SIMPLE
    Score 3-5  → MEDIUM
    Score 6+   → HARD
    """

    def classify(self, request: RiskRequest) -> SequencerDecision:
        """
        Classify request and produce routing decision.

        Args:
            request: Incoming risk assessment request

        Returns:
            SequencerDecision with difficulty, validators, and consensus config
        """
        score, reasons, signals = self._compute_score(request)
        difficulty = self._score_to_difficulty(score, request)

        validators = self._get_validators(difficulty)
        config = DIFFICULTY_CONFIG[difficulty]

        logger.debug(
            f"Sequencer: request={request.request_id} "
            f"score={score} difficulty={difficulty} "
            f"validators={[v.value for v in validators]}"
        )

        return SequencerDecision(
            request_id=request.request_id,
            difficulty_level=difficulty,
            active_validators=validators,
            quorum_threshold=config["quorum_threshold"],
            max_rounds=config["max_rounds"],
            stability_horizon=config["stability_horizon"],
            routing_reasons=reasons,
            risk_signals_detected=signals,
        )

    def _compute_score(self, request: RiskRequest) -> Tuple[int, List[str], List[str]]:
        """Compute routing score. Returns (score, reasons, signals)."""
        score = 0
        reasons: List[str] = []
        signals: List[str] = []

        ctx = request.context
        subject = request.subject
        amount = ctx.amount or 0

        # --- Amount scoring ---
        if amount >= 50_000:
            score += 3
            reasons.append(f"Very high amount: {amount:,.0f}")
            signals.append("very_high_amount")
        elif amount >= 10_000:
            score += 2
            reasons.append(f"High amount: {amount:,.0f}")
            signals.append("high_amount")
        elif amount >= 1_000:
            score += 1
            reasons.append(f"Moderate amount: {amount:,.0f}")
            signals.append("moderate_amount")

        # --- Identity scoring ---
        if subject.trust_score < 0.3:
            score += 3
            reasons.append(f"Low trust score: {subject.trust_score:.2f}")
            signals.append("low_trust_score")
        elif subject.trust_score < 0.6:
            score += 1
            reasons.append(f"Moderate trust score: {subject.trust_score:.2f}")
            signals.append("moderate_trust_score")

        if subject.flagged_count >= 3:
            score += 2
            reasons.append(f"High flag count: {subject.flagged_count}")
            signals.append("high_flag_count")
        elif subject.flagged_count >= 1:
            score += 1
            signals.append("prior_flags")

        # --- Velocity scoring ---
        if ctx.recent_transaction_count >= 10:
            score += 2
            reasons.append(f"High velocity: {ctx.recent_transaction_count} tx/hr")
            signals.append("high_velocity")
        elif ctx.recent_transaction_count >= 5:
            score += 1
            signals.append("moderate_velocity")

        # --- Trace context ---
        if amount >= 5_000 and not ctx.trace_context:
            score += 1
            reasons.append("Missing trace context for high-value action")
            signals.append("missing_trace")

        # --- Cross-border ---
        if ctx.geo_location and subject.jurisdiction:
            if subject.jurisdiction.upper() not in ctx.geo_location.upper():
                score += 1
                signals.append("cross_border")

        return score, reasons, signals

    @staticmethod
    def _score_to_difficulty(
        score: int, request: RiskRequest
    ) -> DifficultyLevel:
        """Map score to difficulty level."""
        # Priority override
        if request.priority == "urgent":
            return DifficultyLevel.HARD

        if score >= 6:
            return DifficultyLevel.HARD
        elif score >= 3:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.SIMPLE

    @staticmethod
    def _get_validators(difficulty: DifficultyLevel) -> List[ValidatorType]:
        """Get validator set for difficulty level."""
        mapping = {
            DifficultyLevel.SIMPLE: SIMPLE_VALIDATORS,
            DifficultyLevel.MEDIUM: MEDIUM_VALIDATORS,
            DifficultyLevel.HARD: HARD_VALIDATORS,
        }
        return mapping[difficulty]

