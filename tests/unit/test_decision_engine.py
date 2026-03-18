"""
Unit tests for DecisionEngine.
"""

import pytest
from aegean.core.decision_engine import DefaultDecisionEngine
from aegean.core.models import Solution


class TestDefaultDecisionEngine:
    """Tests for DefaultDecisionEngine."""
    
    def test_engine_creation(self):
        """Test creating decision engine."""
        engine = DefaultDecisionEngine(
            quorum_size=2,
            stability_horizon=2,
        )
        
        assert engine.quorum_size == 2
        assert engine.stability_horizon == 2
        assert engine.stability_counter == 0
        assert engine.previous_candidate is None
    
    def test_no_quorum(self):
        """Test when quorum is not reached."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=2)
        
        solutions = [
            Solution(agent_id="agent_0", answer="4"),
            Solution(agent_id="agent_1", answer="5"),
            Solution(agent_id="agent_2", answer="6"),
        ]
        
        candidate, should_terminate = engine.evaluate(solutions, round_num=1)
        
        # No quorum (each answer has only 1 vote)
        assert candidate is None
        assert should_terminate is False
        assert engine.stability_counter == 0
    
    def test_quorum_reached(self):
        """Test when quorum is reached."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=2)
        
        solutions = [
            Solution(agent_id="agent_0", answer="4"),
            Solution(agent_id="agent_1", answer="4"),
            Solution(agent_id="agent_2", answer="5"),
        ]
        
        candidate, should_terminate = engine.evaluate(solutions, round_num=1)
        
        # Quorum reached (2 votes for "4")
        assert candidate is not None
        assert candidate.answer == "4"
        assert should_terminate is False  # Need stability_horizon rounds
        assert engine.stability_counter == 1
    
    def test_stability_tracking(self):
        """Test stability tracking across rounds."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=2)
        
        solutions = [
            Solution(agent_id="agent_0", answer="4"),
            Solution(agent_id="agent_1", answer="4"),
            Solution(agent_id="agent_2", answer="5"),
        ]
        
        # Round 1
        candidate1, terminate1 = engine.evaluate(solutions, round_num=1)
        assert candidate1.answer == "4"
        assert terminate1 is False
        assert engine.stability_counter == 1
        
        # Round 2 - same candidate
        candidate2, terminate2 = engine.evaluate(solutions, round_num=2)
        assert candidate2.answer == "4"
        assert terminate2 is True  # Stable for 2 rounds
        assert engine.stability_counter == 2
    
    def test_stability_reset(self):
        """Test stability counter reset when candidate changes."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=2)
        
        # Round 1 - candidate "4"
        solutions1 = [
            Solution(agent_id="agent_0", answer="4"),
            Solution(agent_id="agent_1", answer="4"),
            Solution(agent_id="agent_2", answer="5"),
        ]
        candidate1, _ = engine.evaluate(solutions1, round_num=1)
        assert candidate1.answer == "4"
        assert engine.stability_counter == 1
        
        # Round 2 - candidate changes to "5"
        solutions2 = [
            Solution(agent_id="agent_0", answer="5"),
            Solution(agent_id="agent_1", answer="5"),
            Solution(agent_id="agent_2", answer="4"),
        ]
        candidate2, terminate2 = engine.evaluate(solutions2, round_num=2)
        assert candidate2.answer == "5"
        assert terminate2 is False
        assert engine.stability_counter == 1  # Reset
    
    def test_empty_solutions(self):
        """Test with empty solutions list."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=2)
        
        candidate, should_terminate = engine.evaluate([], round_num=1)
        
        assert candidate is None
        assert should_terminate is False
    
    def test_unanimous_consensus(self):
        """Test unanimous consensus."""
        engine = DefaultDecisionEngine(quorum_size=3, stability_horizon=1)
        
        solutions = [
            Solution(agent_id="agent_0", answer="42"),
            Solution(agent_id="agent_1", answer="42"),
            Solution(agent_id="agent_2", answer="42"),
        ]
        
        candidate, should_terminate = engine.evaluate(solutions, round_num=1)
        
        assert candidate is not None
        assert candidate.answer == "42"
        assert should_terminate is True  # Stable for 1 round (β=1)
    
    def test_reset(self):
        """Test resetting engine state."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=2)
        
        solutions = [
            Solution(agent_id="agent_0", answer="4"),
            Solution(agent_id="agent_1", answer="4"),
        ]
        
        engine.evaluate(solutions, round_num=1)
        assert engine.stability_counter == 1
        assert engine.previous_candidate is not None
        
        engine.reset()
        assert engine.stability_counter == 0
        assert engine.previous_candidate is None
    
    def test_get_state(self):
        """Test getting engine state."""
        engine = DefaultDecisionEngine(quorum_size=2, stability_horizon=3)
        
        state = engine.get_state()
        
        assert state["quorum_size"] == 2
        assert state["stability_horizon"] == 3
        assert state["stability_counter"] == 0
        assert state["previous_candidate"] is None

