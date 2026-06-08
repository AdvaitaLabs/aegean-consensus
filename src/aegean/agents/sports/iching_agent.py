"""IChingAgent: 周易 (I Ching) divination agent for sports prediction."""

from __future__ import annotations

import re
from datetime import date as date_cls
from typing import Optional

from aegean.agents.sports.base_sports_agent import (
    BaseSportsAgent,
    OUTPUT_CONTRACT,
)
from aegean.agents.sports.iching_data import (
    build_iching_context,
    format_iching_context_for_prompt,
)


class IChingAgent(BaseSportsAgent):
    """
    The Chinese sports diviner. Casts an I Ching hexagram + its changing
    hexagram for each match. Pure entertainment. Sister agent to
    OccultAgent (tarot) - same near-zero capability_weight (0.10) so the
    consensus cannot be moved by it, but its prediction is shown to
    users alongside the tarot reader as the Chinese-flavoured option.

    Bazi (four pillars) reasoning was removed; this agent now relies on
    the hexagram cast alone.

    Recommended LLM: Claude Haiku 4.5 (cheap, playful is enough).
    """

    ROLE = "iching_specialist"
    DEFAULT_WEIGHT = 0.10
    SPECIALIZATION_HINT = (
        "You are playing the role of a Chinese I Ching diviner for "
        "entertainment. You will be given an I Ching hexagram cast for "
        "this match plus its 'changing' (变卦) hexagram.\n"
        "Weave a culturally-grounded but playful prediction in English "
        "using:\n"
        "  - The primary hexagram and its image\n"
        "  - The transition implied by the changing hexagram\n"
        "Keep probabilities reasonable (do not output 0.99 or 0). Your "
        "prediction has near-zero weight in the final consensus, so "
        "make it interesting and respectful of the tradition."
    )

    def _build_prompt(self, task: str) -> str:
        match_date = self._extract_match_date(task) or date_cls.today()
        home_code, away_code = self._extract_fifa_codes(task)
        ctx = build_iching_context(home_code, away_code, match_date)
        block = format_iching_context_for_prompt(ctx)
        return (
            f"{self.SPECIALIZATION_HINT}\n\n"
            f"--- MATCH CONTEXT ---\n{task}\n\n"
            f"--- {block}\n\n"
            f"{OUTPUT_CONTRACT}"
        )

    @staticmethod
    def _extract_fifa_codes(task: str) -> tuple:
        """Pull first two 3-letter FIFA codes; default BRA/ARG."""
        candidates = []
        for code in re.findall(r"\b([A-Z]{3})\b", task):
            if code not in candidates:
                candidates.append(code)
            if len(candidates) >= 2:
                break
        if len(candidates) >= 2:
            return candidates[0], candidates[1]
        if len(candidates) == 1:
            return candidates[0], "ARG"
        return "BRA", "ARG"

    @staticmethod
    def _extract_match_date(task: str) -> "Optional[date_cls]":
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", task)
        if not m:
            return None
        try:
            y, mo, d = map(int, m.groups())
            return date_cls(y, mo, d)
        except ValueError:
            return None
