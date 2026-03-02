"""
AutoGen framework integration adapter.

Adapts Microsoft AutoGen agents to work with Aegean consensus protocol.
"""

from typing import List, Optional
import logging

from aegean.core.agent import Agent
from aegean.core.models import Solution

logger = logging.getLogger(__name__)


class AutoGenAgentAdapter(Agent):
    """
    Adapter for Microsoft AutoGen agents.
    
    Wraps an AutoGen agent (AssistantAgent, UserProxyAgent, etc.)
    to implement the Aegean Agent interface.
    
    Example:
        >>> from autogen import AssistantAgent
        >>> from aegean.integrations import AutoGenAgentAdapter
        >>> 
        >>> # Create AutoGen agent
        >>> autogen_agent = AssistantAgent(
        ...     name="assistant",
        ...     llm_config={"model": "gpt-4"}
        ... )
        >>> 
        >>> # Adapt to Aegean
        >>> aegean_agent = AutoGenAgentAdapter(autogen_agent)
        >>> 
        >>> # Use in consensus
        >>> solution = await aegean_agent.generate_solution("What is 2+2?")
    """

    def __init__(
        self,
        autogen_agent,
        agent_id: Optional[str] = None,
        system_message_template: Optional[str] = None,
    ):
        """
        Initialize AutoGen adapter.
        
        Args:
            autogen_agent: AutoGen agent instance (AssistantAgent, etc.)
            agent_id: Optional custom agent ID (uses autogen_agent.name if None)
            system_message_template: Optional template for system messages
        """
        # Use AutoGen agent's name as ID if not provided
        agent_id = agent_id or getattr(autogen_agent, "name", "autogen_agent")
        super().__init__(agent_id)
        
        self.autogen_agent = autogen_agent
        self.system_message_template = system_message_template or (
            "You are a helpful AI assistant participating in a consensus protocol. "
            "Provide clear, accurate answers with reasoning."
        )

    async def generate_solution(self, task: str) -> Solution:
        """
        Generate initial solution using AutoGen agent.
        
        Args:
            task: The task description
            
        Returns:
            Solution with answer and reasoning
        """
        try:
            # Prepare message
            message = f"{self.system_message_template}\n\nTask: {task}\n\nProvide your answer and reasoning."
            
            # Call AutoGen agent
            # Note: AutoGen's generate_reply is synchronous, but we wrap it in async
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            
            # Extract answer and reasoning
            answer, reasoning = self._parse_response(response)
            
            logger.info(f"Agent {self.agent_id} generated solution: {answer[:50]}...")
            
            return Solution(
                agent_id=self.agent_id,
                answer=answer,
                reasoning=reasoning,
                confidence=1.0,  # AutoGen doesn't provide confidence by default
            )
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate solution: {e}")
            # Return error solution
            return Solution(
                agent_id=self.agent_id,
                answer="ERROR",
                reasoning=f"Failed to generate solution: {str(e)}",
                confidence=0.0,
            )

    async def refine_solution(self, refinement_set: List[Solution]) -> Solution:
        """
        Refine solution based on previous solutions from other agents.
        
        Args:
            refinement_set: List of solutions from previous round
            
        Returns:
            Refined solution
        """
        try:
            # Prepare refinement message
            message = self._build_refinement_message(refinement_set)
            
            # Call AutoGen agent
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            
            # Extract answer and reasoning
            answer, reasoning = self._parse_response(response)
            
            logger.info(f"Agent {self.agent_id} refined solution: {answer[:50]}...")
            
            return Solution(
                agent_id=self.agent_id,
                answer=answer,
                reasoning=reasoning,
                confidence=1.0,
            )
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to refine solution: {e}")
            # Return error solution
            return Solution(
                agent_id=self.agent_id,
                answer="ERROR",
                reasoning=f"Failed to refine solution: {str(e)}",
                confidence=0.0,
            )

    def _build_refinement_message(self, refinement_set: List[Solution]) -> str:
        """
        Build message for refinement round.
        
        Args:
            refinement_set: Solutions from previous round
            
        Returns:
            Formatted message for the agent
        """
        message = (
            f"{self.system_message_template}\n\n"
            "You are participating in a consensus protocol. "
            "Below are solutions from other agents in the previous round.\n\n"
        )
        
        # Add each solution
        for i, solution in enumerate(refinement_set, 1):
            message += f"Agent {solution.agent_id}:\n"
            message += f"  Answer: {solution.answer}\n"
            message += f"  Reasoning: {solution.reasoning}\n\n"
        
        message += (
            "Based on these solutions, provide your refined answer. "
            "You may:\n"
            "- Maintain your previous answer if you believe it's correct\n"
            "- Adopt another agent's answer if you find it more convincing\n"
            "- Propose a new answer if you identify issues with all previous answers\n\n"
            "Provide your final answer and reasoning."
        )
        
        return message

    def _parse_response(self, response) -> tuple[str, str]:
        """
        Parse AutoGen response into answer and reasoning.
        
        Args:
            response: Response from AutoGen agent
            
        Returns:
            Tuple of (answer, reasoning)
        """
        # Handle different response formats
        if isinstance(response, dict):
            content = response.get("content", str(response))
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        # Try to extract structured answer
        # Look for patterns like "Answer: X" or "Final Answer: X"
        lines = content.split("\n")
        answer = None
        reasoning = content
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("answer:") or line_lower.startswith("final answer:"):
                # Extract answer
                answer = line.split(":", 1)[1].strip()
                break
        
        # If no structured answer found, use first line as answer
        if answer is None:
            answer = lines[0].strip() if lines else content[:100]
        
        return answer, reasoning

    def __repr__(self) -> str:
        return f"AutoGenAgentAdapter(agent_id='{self.agent_id}', autogen_agent={self.autogen_agent.name})"

