"""
RiskConsensusCoordinator - Risk-specialized consensus coordinator.

Extends the core ConsensusCoordinator with:
- Validator committee orchestration (replacing generic agents)
- Risk-specific decision aggregation
- Integration with Sequencer, SessionManager, ChallengeManager
- WeightedDecisionEngine reuse for validator voting

This is the central orchestrator of the Aegean risk pipeline,
analogous to Trustline's Consensus Coordinator.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import asyncio
import logging

from aegean.risk.models import (
    RiskRequest,
    RiskDecision,
    RiskDecisionType,
    RiskLevel,
    ValidatorType,
    ValidatorResult,
    DifficultyLevel,
    SequencerDecision,
    RISK_LEVEL_SCORE,
)
from aegean.risk.sequencer import Sequencer
from aegean.risk.session import SessionManager
from aegean.risk.challenge import ChallengeManager
from aegean.risk.validators.base_validator import BaseValidator
from aegean.memory.global_memory import GlobalMemorySystem

logger = logging.getLogger(__name__)

# Decision thresholds
CHALLENGE_CONFIDENCE_THRESHOLD = 0.55   # below this → issue challenge
REJECT_RISK_SCORE_THRESHOLD = 0.75      # weighted score above this → reject
CHALLENGE_RISK_SCORE_THRESHOLD = 0.50  # weighted score above this → challenge


class RiskConsensusCoordinator:
    """
    Orchestrates the full Aegean risk evaluation pipeline.

    Pipeline:
    1. Sequencer classifies request → routes to validator subset
    2. SessionManager creates/retrieves session
    3. Validators run in parallel (asyncio.gather)
    4. WeightedDecisionEngine aggregates votes
    5. Decision: approve / reject / challenge
    6. ChallengeManager handles challenge-response if needed
    7. ExperienceBase stores result for future RAG retrieval
    """

    def __init__(
        self,
        validators: Dict[ValidatorType, BaseValidator],
        memory_system: Optional[GlobalMemorySystem] = None,
        session_manager: Optional[SessionManager] = None,
        challenge_manager: Optional[ChallengeManager] = None,
        sequencer: Optional[Sequencer] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            validators: Map of ValidatorType → BaseValidator instance
            memory_system: For storing results in ExperienceBase
            session_manager: Session lifecycle manager
            challenge_manager: Challenge-response handler
            sequencer: Request classifier/router
            config: Optional overrides
        """
        self.validators = validators
        self.memory_system = memory_system
        self.session_manager = session_manager or SessionManager()
        self.challenge_manager = challenge_manager or ChallengeManager()
        self.sequencer = sequencer or Sequencer()
        self.config = config or {}

    async def evaluate(self, request: RiskRequest) -> RiskDecision:
        """
        Full risk evaluation pipeline entry point.

        Args:
            request: The risk assessment request

        Returns:
            RiskDecision with approve/reject/challenge + full audit trail
        """
        start = datetime.now(timezone.utc)

        # Step 1: Create/retrieve session
        session = self.session_manager.create_session(request)
        logger.info(
            f"Risk evaluation started: request={request.request_id} "
            f"session={session.session_id} subject={request.subject.subject_id}"
        )

        # Step 2: Sequencer classification
        seq_decision = self.sequencer.classify(request)
        logger.info(
            f"Sequencer: difficulty={seq_decision.difficulty_level.value} "
            f"validators={[v.value for v in seq_decision.active_validators]}"
        )

        # Step 3: Run validators in parallel
        validator_results = await self._run_validators(
            request=request,
            active_types=seq_decision.active_validators,
            context_hint=f"difficulty={seq_decision.difficulty_level.value}",
        )

        # Step 4: Weighted aggregation → decision
        decision = self._aggregate_decision(
            request=request,
            session_id=session.session_id,
            validator_results=validator_results,
            seq_decision=seq_decision,
            elapsed=(datetime.now(timezone.utc) - start).total_seconds(),
        )

        # Step 5: Attach decision to session
        self.session_manager.attach_decision(session.session_id, decision)

        # Step 6: Persist to ExperienceBase for future RAG
        await self._persist_result(request, decision)

        logger.info(
            f"Risk evaluation complete: "
            f"decision={decision.decision.value} "
            f"risk={decision.risk_level.value} "
            f"confidence={decision.confidence:.2f} "
            f"elapsed={decision.execution_time:.2f}s"
        )
        return decision

    async def re_evaluate(
        self,
        original_request: RiskRequest,
        challenge_id: str,
    ) -> RiskDecision:
        """
        Re-evaluate a request after challenge evidence is submitted.

        Injects challenge context into trace_context and runs
        the full pipeline again (always at HARD difficulty).
        """
        # Build enriched context
        challenge_ctx = self.challenge_manager.get_challenge_context(challenge_id)

        # Inject challenge context into trace
        existing_trace = original_request.context.trace_context or ""
        new_trace = f"{existing_trace}\n\n{challenge_ctx}".strip()

        # Create updated request (reuse session)
        updated_context = original_request.context.model_copy(
            update={"trace_context": new_trace}
        )
        updated_request = original_request.model_copy(
            update={"context": updated_context, "priority": "urgent"}
        )

        # Run full evaluation (sequencer will route to HARD due to urgent)
        new_decision = await self.evaluate(updated_request)

        # Resolve the challenge
        self.challenge_manager.resolve_challenge(challenge_id, new_decision)

        return new_decision

    # ==================== Internal Pipeline ====================

    async def _run_validators(
        self,
        request: RiskRequest,
        active_types: List[ValidatorType],
        context_hint: str,
    ) -> List[ValidatorResult]:
        """Run selected validators in parallel."""
        tasks = []
        active_validators = []

        for vtype in active_types:
            validator = self.validators.get(vtype)
            if validator:
                tasks.append(validator.evaluate(request, context_hint))
                active_validators.append(vtype)
            else:
                logger.warning(f"Validator {vtype} requested but not registered")

        if not tasks:
            raise ValueError("No validators available for evaluation")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Validator {active_validators[i]} raised exception: {result}"
                )
            else:
                valid_results.append(result)

        return valid_results

    def _aggregate_decision(
        self,
        request: RiskRequest,
        session_id: str,
        validator_results: List[ValidatorResult],
        seq_decision: SequencerDecision,
        elapsed: float,
    ) -> RiskDecision:
        """Aggregate validator results into a final RiskDecision."""
        if not validator_results:
            # Fail-safe: no validators ran
            return self._build_decision(
                request=request,
                session_id=session_id,
                decision_type=RiskDecisionType.CHALLENGE,
                risk_level=RiskLevel.HIGH,
                confidence=0.3,
                rationale="No validator results available. Manual review required.",
                indicators=["no_validator_results"],
                validator_results=[],
                seq_decision=seq_decision,
                elapsed=elapsed,
            )

        # Weighted vote aggregation
        level_scores: Dict[str, float] = {}
        level_weights: Dict[str, float] = {}
        total_weight = 0.0

        for result in validator_results:
            weight = result.weight * result.confidence
            score = RISK_LEVEL_SCORE[result.risk_level]
            level = result.risk_level.value

            level_scores[level] = level_scores.get(level, 0.0) + weight * score
            level_weights[level] = level_weights.get(level, 0.0) + weight
            total_weight += weight

        if total_weight == 0:
            weighted_risk_score = 0.3
        else:
            weighted_risk_score = sum(level_scores.values()) / total_weight

        # Determine winning risk level (highest weighted vote)
        winning_level = max(level_weights, key=level_weights.get)
        final_risk_level = RiskLevel(winning_level)

        # Overall confidence = average validator confidence weighted by capability
        total_cap_weight = sum(r.weight for r in validator_results)
        if total_cap_weight > 0:
            confidence = sum(
                r.confidence * r.weight for r in validator_results
            ) / total_cap_weight
        else:
            confidence = 0.5

        # Aggregate risk indicators
        all_indicators = []
        for r in validator_results:
            all_indicators.extend(r.risk_indicators)
        # Deduplicate, preserve order
        seen = set()
        unique_indicators = []
        for ind in all_indicators:
            if ind not in seen:
                seen.add(ind)
                unique_indicators.append(ind)

        # Build rationale from validator reasoning
        rationale = self._build_rationale(validator_results)

        # Map to decision type
        decision_type = self._score_to_decision(
            risk_score=weighted_risk_score,
            confidence=confidence,
            risk_level=final_risk_level,
            difficulty=seq_decision.difficulty_level,
        )

        return self._build_decision(
            request=request,
            session_id=session_id,
            decision_type=decision_type,
            risk_level=final_risk_level,
            confidence=confidence,
            rationale=rationale,
            indicators=unique_indicators,
            validator_results=validator_results,
            seq_decision=seq_decision,
            elapsed=elapsed,
            weighted_votes={k: round(v, 4) for k, v in level_weights.items()},
        )

    @staticmethod
    def _score_to_decision(
        risk_score: float,
        confidence: float,
        risk_level: RiskLevel,
        difficulty: DifficultyLevel,
    ) -> RiskDecisionType:
        """Map aggregated risk score + confidence to a decision type."""
        # Critical risk always rejects
        if risk_level == RiskLevel.CRITICAL:
            return RiskDecisionType.REJECT

        # High risk: reject if confident, challenge if uncertain
        if risk_level == RiskLevel.HIGH:
            if confidence >= 0.7:
                return RiskDecisionType.REJECT
            else:
                return RiskDecisionType.CHALLENGE

        # Medium risk on HARD difficulty: challenge for more info
        if risk_level == RiskLevel.MEDIUM and difficulty == DifficultyLevel.HARD:
            if confidence < CHALLENGE_CONFIDENCE_THRESHOLD:
                return RiskDecisionType.CHALLENGE
            return RiskDecisionType.REVIEW

        # Medium risk otherwise: review
        if risk_level == RiskLevel.MEDIUM:
            return RiskDecisionType.REVIEW

        # Low risk
        return RiskDecisionType.APPROVE

    @staticmethod
    def _build_rationale(validator_results: List[ValidatorResult]) -> str:
        """Build a concise rationale summary from all validator reasonings."""
        parts = []
        for result in validator_results:
            short = result.reasoning[:150].replace("\n", " ")
            parts.append(f"[{result.validator_type.value}] {short}")
        return " | ".join(parts)

    @staticmethod
    def _build_decision(
        request: RiskRequest,
        session_id: str,
        decision_type: RiskDecisionType,
        risk_level: RiskLevel,
        confidence: float,
        rationale: str,
        indicators: List[str],
        validator_results: List[ValidatorResult],
        seq_decision: SequencerDecision,
        elapsed: float,
        weighted_votes: Optional[Dict[str, float]] = None,
    ) -> RiskDecision:
        """Construct the final RiskDecision object."""
        now = datetime.now(timezone.utc)
        ttl = 3600  # 1 hour default
        if risk_level == RiskLevel.LOW:
            ttl = 7200
        elif risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            ttl = 300

        challenge_eligible = decision_type in (
            RiskDecisionType.CHALLENGE,
            RiskDecisionType.REVIEW,
        )

        return RiskDecision(
            request_id=request.request_id,
            session_id=session_id,
            decision=decision_type,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            ttl=ttl,
            expires_at=now + timedelta(seconds=ttl),
            rationale=rationale,
            risk_indicators=indicators[:20],  # cap at 20
            validator_results=validator_results,
            weighted_votes=weighted_votes or {},
            challenge_eligible=challenge_eligible,
            difficulty_level=seq_decision.difficulty_level,
            rounds_used=seq_decision.max_rounds,
            participating_validators=[
                v.value for v in seq_decision.active_validators
            ],
            execution_time=round(elapsed, 3),
        )

    async def _persist_result(
        self, request: RiskRequest, decision: RiskDecision
    ) -> None:
        """Store evaluation result in ExperienceBase for future RAG retrieval."""
        if not self.memory_system:
            return
        try:
            await self.memory_system.store_consensus_result(
                consensus_id=decision.decision_id,
                task=(
                    f"[RISK] {request.context.action_type}: "
                    f"{request.context.description} "
                    f"(subject={request.subject.subject_id}, "
                    f"amount={request.context.amount} {request.context.currency or ''})"
                ),
                final_answer=(
                    f"{decision.decision.value}:{decision.risk_level.value}:"
                    f"{decision.confidence:.2f}"
                ),
                rounds_used=decision.rounds_used,
                consensus_reached=decision.decision != RiskDecisionType.CHALLENGE,
                participating_agents=decision.participating_validators,
                execution_time=decision.execution_time,
                metadata={
                    "risk_indicators": decision.risk_indicators,
                    "subject_id": request.subject.subject_id,
                    "action_type": request.context.action_type,
                    "amount": request.context.amount,
                    "currency": request.context.currency,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to persist risk result to memory: {e}")

    # ==================== Factory ====================

    @classmethod
    def create_default(
        cls,
        memory_system: Optional[GlobalMemorySystem] = None,
        llm_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "RiskConsensusCoordinator":
        """
        Factory method: create a coordinator with all 5 default validators.

        Args:
            memory_system: Optional GlobalMemorySystem for RAG
            llm_client: Optional LLM client for deep analysis
            config: Optional configuration overrides

        Returns:
            Fully configured RiskConsensusCoordinator
        """
        from aegean.risk.validators.identity_validator import IdentityValidator
        from aegean.risk.validators.anomaly_validator import AnomalyValidator
        from aegean.risk.validators.compliance_validator import ComplianceValidator
        from aegean.risk.validators.amount_validator import AmountValidator
        from aegean.risk.validators.context_validator import ContextValidator

        cfg = config or {}
        validator_kwargs = dict(
            memory_system=memory_system,
            llm_client=llm_client,
            config=cfg.get("validator_config", {}),
        )

        validators: Dict[ValidatorType, BaseValidator] = {
            ValidatorType.IDENTITY: IdentityValidator(**validator_kwargs),
            ValidatorType.ANOMALY: AnomalyValidator(**validator_kwargs),
            ValidatorType.COMPLIANCE: ComplianceValidator(**validator_kwargs),
            ValidatorType.AMOUNT: AmountValidator(**validator_kwargs),
            ValidatorType.CONTEXT: ContextValidator(**validator_kwargs),
        }

        return cls(
            validators=validators,
            memory_system=memory_system,
            config=cfg,
        )


