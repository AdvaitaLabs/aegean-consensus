"""
Decision engine for evaluating consensus conditions.

Based on paper Section 5.2: Refinement Decision Engine
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from collections import Counter
from aegean.core.models import Solution


class DecisionEngine(ABC):
    """
    Abstract base class for consensus decision engines.

    The decision engine evaluates whether consensus has been reached
    and determines when to terminate the protocol.
    """

    @abstractmethod
    def evaluate(
        self,
        solutions: List[Solution],
        round_num: int,
        previous_candidate: Optional[Solution] = None
    ) -> Tuple[Optional[Solution], bool]:
        """
        Evaluate solutions and determine if consensus is reached.
        """
        pass


class DefaultDecisionEngine(DecisionEngine):
    """
    Default decision engine implementing the Aegean protocol.
    """

    def __init__(
        self,
        quorum_size: int,
        stability_horizon: int,
        similarity_threshold: float = 0.9
    ):
        self.quorum_size = quorum_size
        self.stability_horizon = stability_horizon
        self.similarity_threshold = similarity_threshold
        self.stability_counter = 0
        self.previous_candidate: Optional[Solution] = None

    def evaluate(
        self,
        solutions: List[Solution],
        round_num: int,
        previous_candidate: Optional[Solution] = None
    ) -> Tuple[Optional[Solution], bool]:
        if not solutions:
            return None, False

        answer_votes = self._count_votes(solutions)
        candidate_answer, vote_count = max(answer_votes.items(), key=lambda x: x[1])

        if vote_count < self.quorum_size:
            self.stability_counter = 0
            self.previous_candidate = None
            return None, False

        candidate_solution = self._pick_best_solution_for_answer(solutions, candidate_answer)

        if self._is_same_candidate(candidate_solution, self.previous_candidate):
            self.stability_counter += 1
        else:
            self.stability_counter = 1
            self.previous_candidate = candidate_solution

        should_terminate = self.stability_counter >= self.stability_horizon
        return candidate_solution, should_terminate

    def _count_votes(self, solutions: List[Solution]) -> Dict[str, int]:
        answers = [s.answer for s in solutions]
        return dict(Counter(answers))

    def _pick_best_solution_for_answer(
        self,
        solutions: List[Solution],
        candidate_answer: str,
    ) -> Solution:
        matches = [s for s in solutions if s.answer == candidate_answer]
        return max(matches, key=lambda s: (s.confidence, 1 if s.proposal else 0))

    def _is_same_candidate(
        self,
        current: Optional[Solution],
        previous: Optional[Solution]
    ) -> bool:
        if current is None or previous is None:
            return False
        return current.answer == previous.answer

    def reset(self) -> None:
        self.stability_counter = 0
        self.previous_candidate = None

    def get_state(self) -> Dict:
        return {
            "stability_counter": self.stability_counter,
            "previous_candidate": (
                self.previous_candidate.answer
                if self.previous_candidate else None
            ),
            "quorum_size": self.quorum_size,
            "stability_horizon": self.stability_horizon,
        }


class CustomDecisionEngine(DecisionEngine):
    def __init__(self, quorum_size: int, stability_horizon: int):
        self.quorum_size = quorum_size
        self.stability_horizon = stability_horizon

    def evaluate(
        self,
        solutions: List[Solution],
        round_num: int,
        previous_candidate: Optional[Solution] = None
    ) -> Tuple[Optional[Solution], bool]:
        raise NotImplementedError("Implement your custom logic")


class WeightedDecisionEngine(DecisionEngine):
    """
    Weighted decision engine for handling agent capability heterogeneity.
    """

    def __init__(
        self,
        quorum_threshold: float = 0.5,
        stability_horizon: int = 2,
        agent_registry=None
    ):
        self.quorum_threshold = quorum_threshold
        self.stability_horizon = stability_horizon
        self.agent_registry = agent_registry
        self.stability_counter = 0
        self.previous_candidate: Optional[Solution] = None
        self.agent_history: Dict[str, Dict] = {}

    def evaluate(
        self,
        solutions: List[Solution],
        round_num: int,
        previous_candidate: Optional[Solution] = None
    ) -> Tuple[Optional[Solution], bool]:
        if not solutions:
            return None, False

        weighted_votes, total_weight = self._calculate_weighted_votes(solutions)
        if not weighted_votes or total_weight <= 0:
            self.stability_counter = 0
            self.previous_candidate = None
            return None, False

        candidate_answer, candidate_weight = max(weighted_votes.items(), key=lambda x: x[1])
        vote_ratio = candidate_weight / total_weight if total_weight > 0 else 0.0

        if vote_ratio < self.quorum_threshold:
            self.stability_counter = 0
            self.previous_candidate = None
            return None, False

        candidate_solution = self._pick_best_solution_for_answer(solutions, candidate_answer)

        if self._is_same_candidate(candidate_solution, self.previous_candidate):
            self.stability_counter += 1
        else:
            self.stability_counter = 1
            self.previous_candidate = candidate_solution

        should_terminate = self.stability_counter >= self.stability_horizon
        return candidate_solution, should_terminate

    def _calculate_weighted_votes(
        self,
        solutions: List[Solution]
    ) -> Tuple[Dict[str, float], float]:
        weighted_votes: Dict[str, float] = {}
        total_weight = 0.0

        for solution in solutions:
            agent_weight = 1.0
            if self.agent_registry:
                agent = self.agent_registry.get_agent(solution.agent_id)
                if agent:
                    agent_weight = getattr(agent, "capability_weight", 1.0)

            confidence = solution.confidence if solution.confidence is not None else 1.0
            vote_weight = max(agent_weight * confidence, 0.0)

            weighted_votes[solution.answer] = weighted_votes.get(solution.answer, 0.0) + vote_weight
            total_weight += vote_weight

        return weighted_votes, total_weight

    def _pick_best_solution_for_answer(
        self,
        solutions: List[Solution],
        candidate_answer: str,
    ) -> Solution:
        matches = [s for s in solutions if s.answer == candidate_answer]
        return max(matches, key=lambda s: (s.confidence, 1 if s.proposal else 0))

    def _is_same_candidate(
        self,
        current: Optional[Solution],
        previous: Optional[Solution]
    ) -> bool:
        if current is None or previous is None:
            return False
        return current.answer == previous.answer
