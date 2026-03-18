"""
Pytest configuration and shared fixtures.
"""

import pytest
from aegean.core.models import Solution


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return "What is 2 + 2?"


@pytest.fixture
def sample_solution():
    """Sample solution for testing."""
    return Solution(
        agent_id="test_agent",
        answer="4",
        reasoning="2 + 2 equals 4",
        confidence=1.0,
    )


@pytest.fixture
def sample_solutions():
    """Sample list of solutions for testing."""
    return [
        Solution(
            agent_id="agent_0",
            answer="4",
            reasoning="2 + 2 = 4",
            confidence=1.0,
        ),
        Solution(
            agent_id="agent_1",
            answer="4",
            reasoning="Two plus two equals four",
            confidence=0.95,
        ),
        Solution(
            agent_id="agent_2",
            answer="5",
            reasoning="I think it's 5",
            confidence=0.6,
        ),
    ]


@pytest.fixture
def mock_agent():
    """Mock agent for testing."""
    from unittest.mock import AsyncMock
    from aegean.core.agent import Agent
    
    class MockAgent(Agent):
        def __init__(self, agent_id: str, answer: str = "42"):
            super().__init__(agent_id)
            self.answer = answer
            self.generate_called = False
            self.refine_called = False
        
        async def generate_solution(self, task: str) -> Solution:
            self.generate_called = True
            return Solution(
                agent_id=self.agent_id,
                answer=self.answer,
                reasoning=f"Mock reasoning for {task}",
                confidence=1.0,
            )
        
        async def refine_solution(self, refinement_set) -> Solution:
            self.refine_called = True
            return Solution(
                agent_id=self.agent_id,
                answer=self.answer,
                reasoning="Mock refined reasoning",
                confidence=1.0,
            )
    
    return MockAgent

