"""
Shared base class for World Cup 2026 sports prediction agents.

Each specialised sub-agent (Stats / Player / Strategy / Market / News / Occult)
overrides:
    ROLE                 unique identifier and aegean.Agent role
    DEFAULT_WEIGHT       capability_weight in [0, 1]
    SPECIALIZATION_HINT  prepended to the LLM prompt to bias the lens

All the LLM-call, JSON parsing, and Solution-wrapping plumbing lives here so
each concrete agent stays ~50 lines.

LLM client contract (matches the rest of aegean-consensus):
    Any object exposing  async def complete(prompt: str) -> str
    plus an optional  last_usage: dict  for token accounting.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from aegean.core.agent import Agent
from aegean.core.models import Solution, TokenUsage

logger = logging.getLogger(__name__)


# ----------------------------- prompt template -----------------------------

OUTPUT_CONTRACT = """OUTPUT FORMAT (strict JSON, no prose before or after):
{
  "p_home_win": 0.45,
  "p_draw": 0.28,
  "p_away_win": 0.27,
  "confidence": 0.65,
  "rationale": "Short one-paragraph explanation."
}
The three probabilities MUST sum to exactly 1.0. Do not include other keys."""


# ----------------------------- JSON parsing -----------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Pull a JSON object out of an LLM response, tolerating markdown fences
    or surrounding prose. Raises ValueError if nothing parses.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def _normalize_probs(p_home: float, p_draw: float, p_away: float) -> Dict[str, float]:
    """Clip negatives and renormalise so the three sum to 1.0."""
    p_home = max(0.0, p_home)
    p_draw = max(0.0, p_draw)
    p_away = max(0.0, p_away)
    total = p_home + p_draw + p_away
    if total == 0:
        return {"p_home_win": 1 / 3, "p_draw": 1 / 3, "p_away_win": 1 / 3}
    return {
        "p_home_win": p_home / total,
        "p_draw": p_draw / total,
        "p_away_win": p_away / total,
    }


# ----------------------------- base agent -----------------------------


class BaseSportsAgent(Agent):
    """
    Shared base for all six sports agents.

    Subclasses define class-level constants only; the heavy lifting
    (LLM call, parsing, Solution wrap) is implemented here.
    """

    # ---- Subclasses override these ----
    ROLE: str = "sports_generic"
    DEFAULT_WEIGHT: float = 0.50
    SPECIALIZATION_HINT: str = (
        "You are a football match analyst. Base your reasoning on the data "
        "provided."
    )

    def __init__(
        self,
        agent_id: Optional[str] = None,
        llm_client: Any = None,
        model_name: str = "",
        capability_weight: Optional[float] = None,
    ):
        super().__init__(
            agent_id=agent_id or self.ROLE,
            capability_weight=(
                capability_weight if capability_weight is not None
                else self.DEFAULT_WEIGHT
            ),
            specialization={"sports_prediction": 1.0},
            role=self.ROLE,
        )
        self._llm = llm_client
        self._model_name = model_name

    # ----------------- Agent contract -----------------

    async def generate_solution(self, task: str) -> Solution:
        """
        First-round solution: build agent-specific prompt, call LLM, parse JSON.

        The 'task' string is the unified match-context prompt produced by the
        AegeanBench prompt builder. We prepend the agent's specialisation hint
        so each role brings its own lens to the same underlying facts.
        """
        prompt = self._build_prompt(task)
        return await self._call_and_wrap(prompt, fallback_confidence=0.30)

    async def refine_solution(self, refinement_set: List[Solution]) -> Solution:
        """
        Second-round solution: show this agent the peers' answers and let it
        revise. We surface only the answer + short reasoning to avoid blowing
        the context window.
        """
        if not refinement_set:
            return await self.generate_solution("")

        peer_lines: List[str] = []
        for s in refinement_set:
            if s.agent_id == self.agent_id:
                continue
            short_reasoning = (s.reasoning or "")[:200]
            peer_lines.append(
                f"- {s.agent_id} (conf {s.confidence:.2f}): "
                f"{s.answer}  ::  {short_reasoning}"
            )
        peer_summary = "\n".join(peer_lines) if peer_lines else "(no peers)"

        prompt = (
            f"{self.SPECIALIZATION_HINT}\n\n"
            f"Your previous answer was based on your specialised lens.\n"
            f"Here are the other agents' answers from this round:\n\n"
            f"{peer_summary}\n\n"
            f"Reconsider your prediction. If a peer's reasoning convinces you, "
            f"adjust. If not, hold your position. Output the same strict JSON "
            f"format as before.\n\n"
            f"{OUTPUT_CONTRACT}"
        )
        # Fall back to majority of peers if the LLM call fails
        majority_fallback = self._majority_fallback(refinement_set)
        return await self._call_and_wrap(
            prompt,
            fallback_confidence=0.50,
            fallback_answer=majority_fallback,
        )

    # ----------------- internals -----------------

    def _build_prompt(self, task: str) -> str:
        """
        Combine the per-agent lens hint with the shared match-context task.
        Output contract goes last so the LLM remembers the JSON format.
        """
        return (
            f"{self.SPECIALIZATION_HINT}\n\n"
            f"--- MATCH CONTEXT ---\n{task}\n\n"
            f"{OUTPUT_CONTRACT}"
        )

    async def _call_and_wrap(
        self,
        prompt: str,
        fallback_confidence: float,
        fallback_answer: Optional[str] = None,
    ) -> Solution:
        """Run the LLM, parse, wrap into Solution. Robust against all failure modes."""
        if self._llm is None:
            return self._fallback_solution(
                "[no LLM configured]",
                confidence=fallback_confidence,
                answer_override=fallback_answer,
            )

        try:
            raw = await self._llm.complete(prompt)
        except Exception as e:
            logger.warning("%s (%s) LLM call failed: %s", self.agent_id, self._model_name, e)
            return self._fallback_solution(
                f"[LLM error: {e}]",
                confidence=fallback_confidence,
                answer_override=fallback_answer,
            )

        try:
            parsed = _extract_json(raw)
        except ValueError as e:
            logger.warning("%s could not parse LLM response: %s", self.agent_id, e)
            return self._fallback_solution(
                f"[JSON parse error] raw: {raw[:100]}",
                confidence=fallback_confidence,
                answer_override=fallback_answer,
            )

        probs = _normalize_probs(
            float(parsed.get("p_home_win", 0)),
            float(parsed.get("p_draw", 0)),
            float(parsed.get("p_away_win", 0)),
        )
        answer = json.dumps(probs, sort_keys=True)

        # Token usage from the LLM client (if it tracks it)
        raw_usage = getattr(self._llm, "last_usage", None) or {}
        tu = TokenUsage.from_raw(raw_usage)

        return Solution(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=str(parsed.get("rationale", "")),
            confidence=float(parsed.get("confidence", max(probs.values()))),
            tokens_prompt=tu.tokens_prompt,
            tokens_completion=tu.tokens_completion,
            usage=tu,
            metadata={
                "role": self.ROLE,
                "model": self._model_name,
                "probs": probs,
            },
        )

    def _fallback_solution(
        self,
        reasoning: str,
        confidence: float,
        answer_override: Optional[str] = None,
    ) -> Solution:
        """Return a safe uniform-distribution Solution when LLM path fails."""
        uniform = {"p_home_win": 1 / 3, "p_draw": 1 / 3, "p_away_win": 1 / 3}
        answer = answer_override or json.dumps(uniform, sort_keys=True)
        return Solution(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=reasoning,
            confidence=confidence,
            metadata={"role": self.ROLE, "model": self._model_name, "fallback": True},
        )

    def _majority_fallback(self, refinement_set: List[Solution]) -> Optional[str]:
        """For refine fallback: use majority answer string from peers."""
        if not refinement_set:
            return None
        answers = [s.answer for s in refinement_set if s.answer]
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][0]
