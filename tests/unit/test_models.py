"""
Unit tests for data models.
"""

import pytest
from datetime import datetime
from aegean.core.models import (
    Solution,
    ConsensusConfig,
    ConsensusState,
    ConsensusResult,
    ConsensusStatus,
)


class TestSolution:
    """Tests for Solution model."""
    
    def test_solution_creation(self):
        """Test creating a solution."""
        solution = Solution(
            agent_id="agent_0",
            answer="42",
            reasoning="The answer to everything",
            confidence=0.95,
        )
        
        assert solution.agent_id == "agent_0"
        assert solution.answer == "42"
        assert solution.reasoning == "The answer to everything"
        assert solution.confidence == 0.95
        assert isinstance(solution.timestamp, datetime)
    
    def test_solution_defaults(self):
        """Test solution with default values."""
        solution = Solution(
            agent_id="agent_0",
            answer="42",
        )
        
        assert solution.reasoning == ""
        assert solution.confidence == 1.0
        assert solution.metadata == {}
    
    def test_solution_validation(self):
        """Test solution validation."""
        # Confidence must be between 0 and 1
        with pytest.raises(ValueError):
            Solution(
                agent_id="agent_0",
                answer="42",
                confidence=1.5,  # Invalid
            )


class TestConsensusConfig:
    """Tests for ConsensusConfig model."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = ConsensusConfig()
        
        assert config.quorum_size == 2
        assert config.stability_horizon == 2
        assert config.max_rounds == 5
        assert config.timeout == 300
        assert config.enable_early_termination is True
        assert config.enable_openclaw is False
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = ConsensusConfig(
            quorum_size=3,
            stability_horizon=3,
            max_rounds=10,
            enable_openclaw=True,
        )
        
        assert config.quorum_size == 3
        assert config.stability_horizon == 3
        assert config.max_rounds == 10
        assert config.enable_openclaw is True


class TestConsensusState:
    """Tests for ConsensusState model."""
    
    def test_state_creation(self):
        """Test creating consensus state."""
        state = ConsensusState(
            consensus_id="test-123",
            status=ConsensusStatus.INITIALIZING,
        )
        
        assert state.consensus_id == "test-123"
        assert state.status == ConsensusStatus.INITIALIZING
        assert state.current_round == 0
        assert state.leader_id is None
        assert state.participating_agents == []
        assert isinstance(state.started_at, datetime)
    
    def test_state_update(self):
        """Test updating state."""
        state = ConsensusState(
            consensus_id="test-123",
            status=ConsensusStatus.INITIALIZING,
        )
        
        # Update state
        state.status = ConsensusStatus.REFINING
        state.current_round = 2
        state.leader_id = "agent_0"
        state.participating_agents = ["agent_0", "agent_1", "agent_2"]
        
        assert state.status == ConsensusStatus.REFINING
        assert state.current_round == 2
        assert state.leader_id == "agent_0"
        assert len(state.participating_agents) == 3


class TestConsensusResult:
    """Tests for ConsensusResult model."""
    
    def test_result_success(self, sample_solution):
        """Test successful consensus result."""
        result = ConsensusResult(
            consensus_id="test-123",
            success=True,
            final_solution=sample_solution,
            rounds_used=3,
            participating_agents=["agent_0", "agent_1", "agent_2"],
            execution_time=5.2,
            consensus_reached=True,
        )
        
        assert result.success is True
        assert result.consensus_reached is True
        assert result.final_solution == sample_solution
        assert result.rounds_used == 3
        assert result.execution_time == 5.2
        assert result.error_message is None
    
    def test_result_failure(self):
        """Test failed consensus result."""
        result = ConsensusResult(
            consensus_id="test-123",
            success=False,
            final_solution=None,
            rounds_used=5,
            participating_agents=["agent_0", "agent_1"],
            execution_time=10.0,
            consensus_reached=False,
            error_message="Timeout",
        )
        
        assert result.success is False
        assert result.consensus_reached is False
        assert result.final_solution is None
        assert result.error_message == "Timeout"

