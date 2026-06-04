"""
Chat Q&A handler (V2 feature).

Lets a user @-mention a specific sports agent in chat and receive a
direct one-on-one answer. Stateless: no user database, no long-term
memory. Each request optionally carries a few recent messages as
session-level context (handled at the request layer, not stored here).

This module is independent of the consensus loop — it's a single-agent
call used for live conversational Q&A while consensus predictions
remain the primary output.

Usage:
    handler = ChatQAHandler(agent_registry=registry)
    response = await handler.answer(
        agent_id="stats_specialist",
        question="will Brazil win their group?",
        match_context="<unified prompt>",
        recent_messages=[...optional...],
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aegean.agents.sports.base_sports_agent import (
    BaseSportsAgent,
    _extract_json,
)
from aegean.core.models import Solution, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class ChatQAResponse:
    """Result of a single @-mention Q&A turn."""
    agent_id: str
    question: str
    answer: str                  # natural-language reply
    confidence: float = 0.5
    rationale: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }


# ----------------------------- prompt -----------------------------

QA_OUTPUT_CONTRACT = """OUTPUT FORMAT (strict JSON, no prose around it):
{
  "answer": "Your natural-language answer to the user. Keep it under 100 words.",
  "confidence": 0.7,
  "rationale": "One sentence on why."
}"""


def build_qa_prompt(
    agent: BaseSportsAgent,
    question: str,
    match_context: Optional[str] = None,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None,
) -> str:
    """
    Build the one-shot Q&A prompt.

    The agent's specialisation hint colours its answer. The match_context
    (if provided) gives the agent the latest match facts. recent_messages
    are short-term session context (last few user turns), not persistent
    history.
    """
    parts: List[str] = [
        agent.SPECIALIZATION_HINT,
        "",
        "You are answering a single user question in a live chat.",
        "Stay in character based on your specialisation above.",
        "Keep the answer short (1-3 sentences). Be specific.",
        "",
    ]
    if match_context:
        parts.append("--- MATCH CONTEXT (current match being discussed) ---")
        parts.append(match_context)
        parts.append("")
    if recent_messages:
        parts.append("--- RECENT CONVERSATION (most recent last) ---")
        for msg in recent_messages[-5:]:
            who = msg.get("user_name", "user")
            text = (msg.get("text") or "").strip()
            parts.append(f"  - {who}: {text}")
        parts.append("")
    parts.append("--- USER QUESTION ---")
    user_prefix = f"@{user_name}: " if user_name else ""
    parts.append(f"{user_prefix}{question}")
    parts.append("")
    parts.append(QA_OUTPUT_CONTRACT)
    return "\n".join(parts)


# ----------------------------- handler -----------------------------


class ChatQAHandler:
    """
    Routes @-mention questions to the right sports agent.

    The handler expects an AgentRegistry-like object that maps agent_id ->
    agent instance. Each sports agent's llm_client is reused.
    """

    def __init__(self, agent_registry):
        self.agent_registry = agent_registry

    async def answer(
        self,
        agent_id: str,
        question: str,
        match_context: Optional[str] = None,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        user_name: Optional[str] = None,
    ) -> ChatQAResponse:
        """
        Route the question to the named agent and return a structured response.

        Returns a fallback ChatQAResponse if the agent is unknown, the LLM
        is unconfigured, or the call fails. This handler never raises into
        the chat-service callers.
        """
        agent = self.agent_registry.get_agent(agent_id)
        if agent is None:
            return ChatQAResponse(
                agent_id=agent_id,
                question=question,
                answer=f"Sorry, no agent named '{agent_id}' is available.",
                confidence=0.0,
                metadata={"error": "unknown_agent"},
            )
        if not isinstance(agent, BaseSportsAgent):
            return ChatQAResponse(
                agent_id=agent_id,
                question=question,
                answer="That agent does not handle conversational Q&A.",
                confidence=0.0,
                metadata={"error": "wrong_agent_type"},
            )
        if agent._llm is None:
            return ChatQAResponse(
                agent_id=agent_id,
                question=question,
                answer="LLM not configured for this agent.",
                confidence=0.0,
                metadata={"error": "no_llm"},
            )

        prompt = build_qa_prompt(
            agent=agent,
            question=question,
            match_context=match_context,
            recent_messages=recent_messages,
            user_name=user_name,
        )

        start = time.perf_counter()
        try:
            raw = await agent._llm.complete(prompt)
        except Exception as e:
            logger.warning("chat-qa LLM call failed for %s: %s", agent_id, e)
            return ChatQAResponse(
                agent_id=agent_id,
                question=question,
                answer="Sorry, I could not answer right now. Try again in a moment.",
                confidence=0.0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                metadata={"error": f"llm_error: {e}"},
            )

        try:
            parsed = _extract_json(raw)
        except ValueError:
            # Some LLMs may return raw prose despite the contract; accept it
            return ChatQAResponse(
                agent_id=agent_id,
                question=question,
                answer=(raw or "").strip()[:500],
                confidence=0.5,
                latency_ms=int((time.perf_counter() - start) * 1000),
                metadata={"warning": "non_json_response"},
            )

        usage = getattr(agent._llm, "last_usage", None) or {}
        tu = TokenUsage.from_raw(usage)

        return ChatQAResponse(
            agent_id=agent_id,
            question=question,
            answer=str(parsed.get("answer", "")).strip(),
            confidence=float(parsed.get("confidence", 0.5)),
            rationale=str(parsed.get("rationale", "")),
            latency_ms=int((time.perf_counter() - start) * 1000),
            tokens_used=tu.tokens_prompt + tu.tokens_completion,
            metadata={"model": agent._model_name},
        )
