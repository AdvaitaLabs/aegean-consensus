"""
AutoGen framework integration adapter.

Adapts Microsoft AutoGen agents to work with Aegean consensus protocol.
"""

from typing import List, Optional
import json
import logging

from aegean.core.agent import Agent
from aegean.core.models import ActionProposal, Solution

logger = logging.getLogger(__name__)


class AutoGenAgentAdapter(Agent):
    """
    Adapter for Microsoft AutoGen agents.
    """

    def __init__(
        self,
        autogen_agent,
        agent_id: Optional[str] = None,
        capability_weight: float = 1.0,
        specialization: Optional[dict] = None,
        role: Optional[str] = None,
        system_message_template: Optional[str] = None,
    ):
        agent_id = agent_id or getattr(autogen_agent, "name", "autogen_agent")
        super().__init__(
            agent_id=agent_id,
            capability_weight=capability_weight,
            specialization=specialization,
            role=role
        )

        self.autogen_agent = autogen_agent
        self.system_message_template = system_message_template or (
            "You are a helpful AI assistant participating in a consensus protocol. "
            "Provide clear, accurate answers with reasoning."
        )

    async def generate_solution(self, task: str) -> Solution:
        try:
            message = (
                f"{self.system_message_template}\n\n"
                f"Task: {task}\n\n"
                "Return a concise answer. If the task is an action-selection task, you may output JSON with "
                "primary_action, backup_action, confidence, and reason."
            )
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            answer, reasoning, proposal = self._parse_response(response)

            logger.info(f"Agent {self.agent_id} generated solution: {answer[:50]}...")
            solution = Solution(
                agent_id=self.agent_id,
                answer=answer,
                reasoning=reasoning,
                confidence=proposal.confidence if proposal else 1.0,
            )
            solution.set_proposal(proposal)
            return solution

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate solution: {e}")
            return Solution(
                agent_id=self.agent_id,
                answer="ERROR",
                reasoning=f"Failed to generate solution: {str(e)}",
                confidence=0.0,
            )

    async def refine_solution(self, refinement_set: List[Solution]) -> Solution:
        try:
            message = self._build_refinement_message(refinement_set)
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            answer, reasoning, proposal = self._parse_response(response)

            logger.info(f"Agent {self.agent_id} refined solution: {answer[:50]}...")
            solution = Solution(
                agent_id=self.agent_id,
                answer=answer,
                reasoning=reasoning,
                confidence=proposal.confidence if proposal else 1.0,
            )
            solution.set_proposal(proposal)
            return solution

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to refine solution: {e}")
            return Solution(
                agent_id=self.agent_id,
                answer="ERROR",
                reasoning=f"Failed to refine solution: {str(e)}",
                confidence=0.0,
            )

    def _build_refinement_message(self, refinement_set: List[Solution]) -> str:
        message = (
            f"{self.system_message_template}\n\n"
            "You are participating in a consensus protocol. "
            "Below are solutions from other agents in the previous round.\n\n"
        )

        for solution in refinement_set:
            message += f"Agent {solution.agent_id}:\n"
            message += f"  Answer: {solution.answer}\n"
            if solution.proposal:
                message += f"  Proposal: {solution.proposal.model_dump_json()}\n"
            message += f"  Reasoning: {solution.reasoning}\n\n"

        message += (
            "Based on these solutions, provide your refined answer. "
            "You may maintain your answer, adopt another answer, or propose a new answer. "
            "For action-selection tasks, JSON output with primary_action, backup_action, confidence, and reason is allowed."
        )
        return message

    def _parse_response(self, response) -> tuple[str, str, Optional[ActionProposal]]:
        if isinstance(response, dict):
            content = response.get("content", str(response))
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)

        proposal = self._extract_proposal(content)
        if proposal:
            return proposal.primary_action, content, proposal

        lines = content.split("\n")
        answer = None
        reasoning = content

        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("answer:") or line_lower.startswith("final answer:"):
                answer = line.split(":", 1)[1].strip()
                break

        if answer is None:
            answer = lines[0].strip() if lines else content[:100]

        return answer, reasoning, None

    def _extract_proposal(self, content: str) -> Optional[ActionProposal]:
        text = content.strip()
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return ActionProposal.from_raw(payload)

    def __repr__(self) -> str:
        return f"AutoGenAgentAdapter(agent_id='{self.agent_id}', autogen_agent={self.autogen_agent.name})"
