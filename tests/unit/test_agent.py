"""
Unit tests for Agent and AgentRegistry.
"""

import pytest
from aegean.core.agent import Agent, AgentRegistry
from aegean.core.models import Solution


class TestAgent:
    """Tests for Agent base class."""
    
    def test_agent_is_abstract(self):
        """Test that Agent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Agent("test_agent")
    
    @pytest.mark.asyncio
    async def test_mock_agent(self, mock_agent):
        """Test mock agent implementation."""
        agent = mock_agent("test_agent", answer="42")
        
        # Test generate_solution
        solution = await agent.generate_solution("What is the answer?")
        assert solution.agent_id == "test_agent"
        assert solution.answer == "42"
        assert agent.generate_called is True
        
        # Test refine_solution
        refinement_set = [solution]
        refined = await agent.refine_solution(refinement_set)
        assert refined.agent_id == "test_agent"
        assert agent.refine_called is True


class TestAgentRegistry:
    """Tests for AgentRegistry."""
    
    def test_registry_creation(self):
        """Test creating an empty registry."""
        registry = AgentRegistry()
        assert registry.count() == 0
        assert registry.get_all_agents() == []
    
    def test_register_agent(self, mock_agent):
        """Test registering agents."""
        registry = AgentRegistry()
        agent1 = mock_agent("agent_1")
        agent2 = mock_agent("agent_2")
        
        registry.register_agent(agent1, is_static=True)
        registry.register_agent(agent2, is_static=True)
        
        assert registry.count() == 2
        assert len(registry.get_all_agents()) == 2
        assert len(registry.get_static_agents()) == 2
        assert len(registry.get_dynamic_agents()) == 0
    
    def test_register_dynamic_agent(self, mock_agent):
        """Test registering dynamic agents."""
        registry = AgentRegistry()
        agent = mock_agent("openclaw_agent")
        
        registry.register_agent(agent, is_static=False)
        
        assert registry.count() == 1
        assert len(registry.get_static_agents()) == 0
        assert len(registry.get_dynamic_agents()) == 1
    
    def test_get_agent(self, mock_agent):
        """Test retrieving agent by ID."""
        registry = AgentRegistry()
        agent = mock_agent("agent_1")
        
        registry.register_agent(agent)
        
        retrieved = registry.get_agent("agent_1")
        assert retrieved is agent
        assert retrieved.agent_id == "agent_1"
    
    def test_get_nonexistent_agent(self):
        """Test retrieving non-existent agent."""
        registry = AgentRegistry()
        
        result = registry.get_agent("nonexistent")
        assert result is None
    
    def test_unregister_agent(self, mock_agent):
        """Test unregistering agents."""
        registry = AgentRegistry()
        agent = mock_agent("agent_1")
        
        registry.register_agent(agent)
        assert registry.count() == 1
        
        registry.unregister_agent("agent_1")
        assert registry.count() == 0
        assert registry.get_agent("agent_1") is None
    
    def test_mixed_agents(self, mock_agent):
        """Test registry with mixed static and dynamic agents."""
        registry = AgentRegistry()
        
        static1 = mock_agent("static_1")
        static2 = mock_agent("static_2")
        dynamic1 = mock_agent("dynamic_1")
        
        registry.register_agent(static1, is_static=True)
        registry.register_agent(static2, is_static=True)
        registry.register_agent(dynamic1, is_static=False)
        
        assert registry.count() == 3
        assert len(registry.get_static_agents()) == 2
        assert len(registry.get_dynamic_agents()) == 1

