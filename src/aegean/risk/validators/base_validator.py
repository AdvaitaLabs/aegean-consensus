"""
Abstract base class for all specialist risk validators.

Each validator committee inherits from this class and implements
its own domain-specific risk analysis logic, backed by:
- LLM reasoning (via PromptEnhancer + GlobalMemorySystem RAG)
- Deterministic rule checks (fast pre-screening)
- Weighted output compatible with WeightedDecisionEngine
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
import time

from aegean.risk.models import (
    RiskRequest,
    ValidatorResult,
    ValidatorType,
    RiskLevel,
)
from aegean.memory.global_memory import GlobalMemorySystem

logger = logging.getLogger(__name__)


# Risk level numeric mapping for aggregation
RISK_LEVEL_SCORE: Dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.1,
    RiskLevel.MEDIUM: 0.4,
    RiskLevel.HIGH: 0.75,
    RiskLevel.CRITICAL: 1.0,
}


def score_to_risk_level(score: float) -> RiskLevel:
    """Convert numeric score to RiskLevel enum."""
    if score < 0.2:
        return RiskLevel.LOW
    elif score < 0.5:
        return RiskLevel.MEDIUM
    elif score < 0.8:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


class BaseValidator(ABC):
    """
    Abstract base class for specialist risk validators.

    Each concrete validator:
    1. Runs deterministic pre-screening rules (fast, no LLM)
    2. Retrieves relevant RAG context via GlobalMemorySystem
    3. Calls LLM with enriched prompt for deep analysis
    4. Returns a ValidatorResult with risk_level, confidence, reasoning

    The capability_weight of each validator influences its voting
    power in WeightedDecisionEngine consensus.
    """

    # Override in subclasses
    validator_type: ValidatorType = NotImplemented

    # Base capability weight - subclasses tune this
    # Higher = more trusted domain expert
    base_capability_weight: float = 1.0

    def __init__(
        self,
        validator_id: str,
        memory_system: Optional[GlobalMemorySystem] = None,
        llm_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize validator.

        Args:
            validator_id: Unique ID for this validator instance
            memory_system: GlobalMemorySystem for RAG context retrieval
            llm_client: LLM client for deep analysis (optional)
            config: Validator-specific configuration
        """
        self.validator_id = validator_id
        self.memory_system = memory_system
        self.llm_client = llm_client
        self.config = config or {}

        # Performance tracking (feeds into WeightedDecisionEngine)
        self._total_evaluations: int = 0
        self._correct_evaluations: int = 0

    @property
    def capability_weight(self) -> float:
        """
        Dynamic capability weight based on historical accuracy.
        Starts at base_capability_weight, adjusts with feedback.
        """
        if self._total_evaluations < 10:
            return self.base_capability_weight  # Not enough data yet
        accuracy = self._correct_evaluations / self._total_evaluations
        # Blend base weight with earned accuracy
        return 0.4 * self.base_capability_weight + 0.6 * accuracy

    async def evaluate(
        self,
        request: RiskRequest,
        context_hint: Optional[str] = None,
    ) -> ValidatorResult:
        """
        Full evaluation pipeline:
        1. Pre-screening rules
        2. RAG context retrieval
        3. LLM analysis (if available)
        4. Build ValidatorResult

        Args:
            request: The risk assessment request
            context_hint: Optional extra context from sequencer

        Returns:
            ValidatorResult with assessment
        """
        start = time.time()
        self._total_evaluations += 1

        try:
            # Step 1: Fast deterministic pre-screening
            pre_result = self._pre_screen(request)

            # If pre-screening already gives CRITICAL confidence, skip LLM
            if pre_result and pre_result["confidence"] >= 0.95:
                logger.debug(
                    f"{self.validator_type} pre-screen high confidence, skipping LLM"
                )
                return self._build_result(
                    request=request,
                    risk_level=pre_result["risk_level"],
                    confidence=pre_result["confidence"],
                    reasoning=pre_result["reasoning"],
                    risk_indicators=pre_result.get("indicators", []),
                    elapsed=time.time() - start,
                )

            # Step 2: Retrieve RAG context
            rag_context = ""
            if self.memory_system:
                rag_context = await self._retrieve_context(request)

            # Step 3: LLM deep analysis
            if self.llm_client:
                llm_result = await self._analyze_with_llm(
                    request=request,
                    rag_context=rag_context,
                    pre_result=pre_result,
                    context_hint=context_hint,
                )
            else:
                # No LLM: fall back to pre-screen result or heuristic
                llm_result = pre_result or self._heuristic_fallback(request)

            elapsed = time.time() - start
            logger.debug(
                f"{self.validator_type}:{self.validator_id} evaluated in {elapsed:.2f}s "
                f"-> {llm_result['risk_level']} (conf={llm_result['confidence']:.2f})"
            )

            return self._build_result(
                request=request,
                risk_level=llm_result["risk_level"],
                confidence=llm_result["confidence"],
                reasoning=llm_result["reasoning"],
                risk_indicators=llm_result.get("indicators", []),
                elapsed=elapsed,
            )

        except Exception as e:
            logger.error(f"{self.validator_type} evaluation error: {e}", exc_info=True)
            # Fail safe: return MEDIUM risk on error
            return self._build_result(
                request=request,
                risk_level=RiskLevel.MEDIUM,
                confidence=0.3,
                reasoning=f"Validator error: {str(e)}. Defaulting to medium risk.",
                risk_indicators=["validator_error"],
                elapsed=time.time() - start,
            )

    # ==================== Abstract Methods ====================

    @abstractmethod
    def _pre_screen(
        self, request: RiskRequest
    ) -> Optional[Dict[str, Any]]:
        """
        Fast deterministic rule-based pre-screening.

        No LLM calls. Should complete in < 5ms.

        Returns:
            Dict with keys: risk_level, confidence, reasoning, indicators
            OR None if no strong signal found (defer to LLM)
        """
        pass

    @abstractmethod
    async def _analyze_with_llm(
        self,
        request: RiskRequest,
        rag_context: str,
        pre_result: Optional[Dict[str, Any]],
        context_hint: Optional[str],
    ) -> Dict[str, Any]:
        """
        Deep LLM-based analysis with RAG context.

        Returns:
            Dict with keys: risk_level, confidence, reasoning, indicators
        """
        pass

    @abstractmethod
    def _get_rag_query(self, request: RiskRequest) -> str:
        """
        Build the RAG retrieval query for this validator's domain.
        Each validator focuses its knowledge retrieval differently.
        """
        pass

    @abstractmethod
    def _get_rag_category(self) -> Optional[str]:
        """
        Knowledge base category filter for this validator.
        e.g. 'aml_regulations', 'fraud_patterns', 'identity_verification'
        """
        pass

    # ==================== Shared Utilities ====================

    async def _retrieve_context(self, request: RiskRequest) -> str:
        """
        Retrieve RAG context from GlobalMemorySystem.
        Shared implementation used by all validators.
        """
        if not self.memory_system:
            return ""

        try:
            query = self._get_rag_query(request)
            category = self._get_rag_category()

            context = await self.memory_system.retrieve_context(
                query=query,
                category=category,
                include_knowledge=True,
                include_cases=True,
                include_performance=False,
            )
            return context.format_for_prompt(max_docs=3, max_cases=2)
        except Exception as e:
            logger.warning(f"RAG retrieval failed for {self.validator_type}: {e}")
            return ""

    def _heuristic_fallback(
        self, request: RiskRequest
    ) -> Dict[str, Any]:
        """
        Simple heuristic fallback when both pre-screen and LLM are unavailable.
        Conservative: defaults to MEDIUM risk.
        """
        return {
            "risk_level": RiskLevel.MEDIUM,
            "confidence": 0.4,
            "reasoning": "Heuristic fallback: insufficient data for precise evaluation.",
            "indicators": ["insufficient_data"],
        }

    def _build_result(
        self,
        request: RiskRequest,
        risk_level: RiskLevel,
        confidence: float,
        reasoning: str,
        risk_indicators: List[str],
        elapsed: float = 0.0,
    ) -> ValidatorResult:
        """Build a ValidatorResult from analysis outputs."""
        return ValidatorResult(
            validator_type=self.validator_type,
            validator_id=self.validator_id,
            risk_level=risk_level,
            confidence=confidence,
            reasoning=reasoning,
            risk_indicators=risk_indicators,
            weight=self.capability_weight,
            metadata={
                "elapsed_seconds": elapsed,
                "request_id": request.request_id,
            },
        )

    def record_feedback(self, was_correct: bool) -> None:
        """
        Record outcome feedback to update capability_weight.
        Called by RiskConsensusCoordinator after ground truth is known.
        """
        if was_correct:
            self._correct_evaluations += 1
        # _total_evaluations already incremented in evaluate()

    def get_stats(self) -> Dict[str, Any]:
        """Return validator performance stats."""
        accuracy = (
            self._correct_evaluations / self._total_evaluations
            if self._total_evaluations > 0
            else None
        )
        return {
            "validator_id": self.validator_id,
            "validator_type": self.validator_type,
            "capability_weight": self.capability_weight,
            "total_evaluations": self._total_evaluations,
            "accuracy": accuracy,
        }

