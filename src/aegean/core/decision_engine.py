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
        
        Args:
            solutions: List of solutions from current round
            round_num: Current round number
            previous_candidate: Candidate solution from previous round
            
        Returns:
            Tuple of (candidate_solution, should_terminate)
            - candidate_solution: The current candidate (or None)
            - should_terminate: Whether to terminate the protocol
        """
        pass


class DefaultDecisionEngine(DecisionEngine):
    """
    Default decision engine implementing the Aegean protocol.
    
    Based on paper Algorithm 1 and Section 5.2:
    - Quorum detection: α = ⌈N/2⌉ agents must agree
    - Stability horizon: Candidate must be stable for β rounds
    """

    def __init__(
        self,
        quorum_size: int,
        stability_horizon: int,
        similarity_threshold: float = 0.9
    ):
        """
        Initialize decision engine.
        
        Args:
            quorum_size: Minimum agents needed for quorum (α)
            stability_horizon: Rounds to maintain stability (β)
            similarity_threshold: Threshold for considering answers similar
        """
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
        """
        Evaluate solutions using quorum detection and stability tracking.
        
        Algorithm:
        1. Count votes for each unique answer
        2. Find answer with most votes
        3. Check if it reaches quorum (≥ α votes)
        4. If yes, check stability:
           - If same as previous candidate: stability_counter++
           - If different: reset stability_counter = 1
        5. Terminate if stability_counter ≥ β
        """
        if not solutions:
            return None, False

        # Step 1: Count votes for each answer
        answer_votes = self._count_votes(solutions)
        
        # Step 2: Find candidate with most votes
        candidate_answer, vote_count = max(
            answer_votes.items(),
            key=lambda x: x[1]
        )
        
        # Step 3: Check quorum
        if vote_count < self.quorum_size:
            # No quorum reached
            self.stability_counter = 0
            self.previous_candidate = None
            return None, False
        
        # Find the solution object for the candidate answer
        candidate_solution = next(
            s for s in solutions if s.answer == candidate_answer
        )
        
        # Step 4: Check stability
        if self._is_same_candidate(candidate_solution, self.previous_candidate):
            # Same candidate as previous round
            self.stability_counter += 1
        else:
            # New candidate
            self.stability_counter = 1
            self.previous_candidate = candidate_solution
        
        # Step 5: Check termination condition
        should_terminate = self.stability_counter >= self.stability_horizon
        
        return candidate_solution, should_terminate

    def _count_votes(self, solutions: List[Solution]) -> Dict[str, int]:
        """
        Count votes for each unique answer.
        
        Returns:
            Dictionary mapping answer -> vote count
        """
        answers = [s.answer for s in solutions]
        return dict(Counter(answers))

    def _is_same_candidate(
        self,
        current: Optional[Solution],
        previous: Optional[Solution]
    ) -> bool:
        """
        Check if two candidates are the same.
        
        Uses exact string matching for simplicity.
        Can be extended to use similarity metrics.
        """
        if current is None or previous is None:
            return False
        
        return current.answer == previous.answer

    def reset(self) -> None:
        """Reset the decision engine state."""
        self.stability_counter = 0
        self.previous_candidate = None

    def get_state(self) -> Dict:
        """Get current state for debugging."""
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
    """
    Example of a custom decision engine.
    
    Users can extend this to implement custom consensus logic,
    such as:
    - Weighted voting based on agent confidence
    - Semantic similarity for answer comparison
    - Dynamic quorum size adjustment
    """

    def __init__(self, quorum_size: int, stability_horizon: int):
        self.quorum_size = quorum_size
        self.stability_horizon = stability_horizon
        # Add your custom state here

    def evaluate(
        self,
        solutions: List[Solution],
        round_num: int,
        previous_candidate: Optional[Solution] = None
    ) -> Tuple[Optional[Solution], bool]:
        """
        Implement your custom consensus logic here.
        
        Example ideas:
        - Use solution.confidence for weighted voting
        - Use NLP similarity for answer comparison
        - Adjust quorum_size based on round_num
        """
        # Your implementation here
        raise NotImplementedError("Implement your custom logic")


class WeightedDecisionEngine(DecisionEngine):
    """
    Weighted decision engine for handling agent capability heterogeneity.
    
    Implements weighted voting where different agents have different voting power
    based on their capability, confidence, and historical accuracy.
    
    Weight calculation:
        vote_weight = capability_weight × confidence × historical_accuracy
    
    This solves the "college student vs elementary student" problem where
    high-capability agents should have more influence than low-capability ones.
    """

    def __init__(
        self,
        quorum_threshold: float = 0.5,
        stability_horizon: int = 2,
        agent_registry = None
    ):
        """
        Initialize weighted decision engine.
        
        Args:
            quorum_threshold: Minimum weighted vote ratio needed (0.0-1.0)
            stability_horizon: Rounds to maintain stability (β)
            agent_registry: Optional agent registry for historical accuracy lookup
        """
        self.quorum_threshold = quorum_threshold
        self.stability_horizon = stability_horizon
        self.agent_registry = agent_registry
        self.stability_counter = 0
        self.previous_candidate: Optional[Solution] = None
        
        # Track agent historical accuracy
        self.agent_history: Dict[str, Dict] = {}

    def evaluate(
        self,
        solutions: List[Solution],
        round_num: int,
        previous_candidate: Optional[Solution] = None
    ) -> Tuple[Optional[Solution], bool]:
        """
        Evaluate solutions using weighted voting.
        
        Algorithm:
        1. Calculate weighted votes for each answer
        2. Find answer with highest weighted score
        3. Check if it reaches weighted quorum threshold
        4. Check stability across rounds
        5. Terminate if stable for β rounds
        """
        if not solutions:
            return None, False

        # Step 1: Calculate weighted votes
        weighted_votes, total_weight = self._calculate_weighted_votes(solutions)
        
        if total_weight == 0:
            return None, False
        
        # Step 2: Find candidate with highest weighted score
        candidate_answer = max(weighted_votes, key=weighted_votes.get)
        candidate_weight = weighted_votes[candidate_answer]
        
        # Step 3: Check weighted quorum
        weight_ratio = candidate_weight / total_weight
        
        if weight_ratio < self.quorum_threshold:
            # No quorum reached
            self.stability_counter = 0
            self.previous_candidate = None
            return None, False
        
        # Find the solution object for the candidate answer
        candidate_solution = next(
            s for s in solutions if s.answer == candidate_answer
        )
        
        # Step 4: Check stability
        if self._is_same_candidate(candidate_solution, self.previous_candidate):
            # Same candidate as previous round
            self.stability_counter += 1
        else:
            # New candidate
            self.stability_counter = 1
            self.previous_candidate = candidate_solution
        
        # Step 5: Check termination condition
        should_terminate = self.stability_counter >= self.stability_horizon
        
        return candidate_solution, should_terminate

    def _calculate_weighted_votes(
        self,
        solutions: List[Solution]
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculate weighted votes for each answer.
        
        Returns:
            Tuple of (weighted_votes_dict, total_weight)
        """
        weighted_votes: Dict[str, float] = {}
        total_weight = 0.0
        
        for solution in solutions:
            # Calculate comprehensive weight
            weight = self._calculate_solution_weight(solution)
            total_weight += weight
            
            # Add to weighted votes
            answer = solution.answer
            if answer not in weighted_votes:
                weighted_votes[answer] = 0.0
            weighted_votes[answer] += weight
        
        return weighted_votes, total_weight

    def _calculate_solution_weight(self, solution: Solution) -> float:
        """
        Calculate comprehensive weight for a solution.
        
        Weight = capability_weight × confidence × historical_accuracy
        
        Args:
            solution: Solution to calculate weight for
            
        Returns:
            Comprehensive weight (0.0-1.0)
        """
        # Get agent from registry
        agent = None
        if self.agent_registry:
            agent = self.agent_registry.get_agent(solution.agent_id)
        
        # 1. Capability weight (from agent)
        capability_weight = 1.0
        if agent and hasattr(agent, 'capability_weight'):
            capability_weight = agent.capability_weight
        
        # 2. Confidence (from solution)
        confidence = solution.confidence if solution.confidence else 1.0
        
        # 3. Historical accuracy (from tracking)
        historical_accuracy = self._get_historical_accuracy(solution.agent_id)
        
        # Comprehensive weight
        return capability_weight * confidence * historical_accuracy

    def _get_historical_accuracy(self, agent_id: str) -> float:
        """
        Get agent's historical accuracy.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Historical accuracy (0.0-1.0), defaults to 1.0 for new agents
        """
        if agent_id not in self.agent_history:
            return 1.0  # Default for new agents
        
        history = self.agent_history[agent_id]
        total = history.get("total", 0)
        correct = history.get("correct", 0)
        
        if total == 0:
            return 1.0
        
        return correct / total

    def update_agent_accuracy(
        self,
        agent_id: str,
        was_correct: bool
    ) -> None:
        """
        Update agent's historical accuracy based on feedback.
        
        Args:
            agent_id: Agent identifier
            was_correct: Whether the agent's answer was correct
        """
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = {"total": 0, "correct": 0}
        
        self.agent_history[agent_id]["total"] += 1
        if was_correct:
            self.agent_history[agent_id]["correct"] += 1

    def _is_same_candidate(
        self,
        current: Optional[Solution],
        previous: Optional[Solution]
    ) -> bool:
        """Check if two candidates are the same."""
        if current is None or previous is None:
            return False
        
        return current.answer == previous.answer

    def reset(self) -> None:
        """Reset the decision engine state."""
        self.stability_counter = 0
        self.previous_candidate = None

    def get_state(self) -> Dict:
        """Get current state for debugging."""
        return {
            "stability_counter": self.stability_counter,
            "previous_candidate": (
                self.previous_candidate.answer 
                if self.previous_candidate else None
            ),
            "quorum_threshold": self.quorum_threshold,
            "stability_horizon": self.stability_horizon,
            "agent_history": self.agent_history,
        }

