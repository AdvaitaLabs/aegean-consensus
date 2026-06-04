"""OccultAgent: zodiac, tarot, and football mysticism (entertainment)."""

from __future__ import annotations

import os
import re
from datetime import date as date_cls
from typing import List

from aegean.agents.sports.base_sports_agent import (
    BaseSportsAgent,
    OUTPUT_CONTRACT,
)
from aegean.agents.sports.occult_data import (
    TEAM_CAPTAIN_ZODIAC,
    build_mystic_context,
    format_mystic_context_for_prompt,
)


class OccultAgent(BaseSportsAgent):
    """
    The sports astrologer. Pure entertainment. Weight is intentionally
    near-zero (0.10) so it cannot materially shift consensus outcomes,
    but its predictions are shown to users for fun.

    Now backed by:
      - Real 78-card tarot deck (3-card draw per match)
      - Zodiac sign of each team's captain
      - Numerology of the match date
      - Optional live horoscope via aztro (free, no key) - mock by default

    Recommended LLM: Claude Haiku 4.5 (cheap, playful is enough).
    """

    ROLE = "occult_specialist"
    DEFAULT_WEIGHT = 0.10
    SPECIALIZATION_HINT = (
        "You are playing the role of a sports astrologer for entertainment.\n"
        "You will be given a real tarot draw, captain zodiac signs, and a\n"
        "numerology number for the match date. Weave a playful but coherent\n"
        "narrative tying these to a probability prediction.\n"
        "Keep probabilities reasonable (do not output 0.99/0/0.01 unless the\n"
        "spread is overwhelming). Your prediction has near-zero weight in\n"
        "the final consensus, so make it interesting."
    )

    def __init__(
        self,
        *args,
        mock_horoscope: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Allow operator to opt into live aztro via env var
        env_flag = os.getenv("OCCULT_LIVE_HOROSCOPE", "").strip()
        self.mock_horoscope = mock_horoscope and env_flag != "1"

    def _build_prompt(self, task: str) -> str:
        """
        Inject a real tarot draw + zodiac context into the prompt.

        Extracts home/away FIFA codes from the task text. Falls back to
        a default code pair when the task doesn't include identifiable
        team codes (the LLM will still get the mystic context, just with
        generic captain signs).
        """
        home_code, away_code = self._extract_fifa_codes(task)
        match_date = self._extract_match_date(task) or date_cls.today()
        mystic = build_mystic_context(
            home_fifa=home_code,
            away_fifa=away_code,
            match_date=match_date,
            mock_horoscope=self.mock_horoscope,
        )
        mystic_block = format_mystic_context_for_prompt(mystic)

        return (
            f"{self.SPECIALIZATION_HINT}\n\n"
            f"--- MATCH CONTEXT ---\n{task}\n\n"
            f"--- {mystic_block}\n\n"
            f"{OUTPUT_CONTRACT}"
        )

    # ---------- helpers ----------

    @staticmethod
    def _extract_fifa_codes(task: str) -> tuple:
        """
        Pull the first two recognised FIFA codes from the task prompt.

        Returns:
            (home_code, away_code). Defaults to ("BRA", "ARG") so the
            mystic context always has something to draw on.
        """
        # Look for explicit 3-letter codes (BRA, ARG, etc.) appearing as
        # standalone tokens. Filter to known captain-zodiac entries to
        # avoid false-positives on uppercase words in prose.
        candidates: List[str] = []
        for code in re.findall(r"\b([A-Z]{3})\b", task):
            if code in TEAM_CAPTAIN_ZODIAC and code not in candidates:
                candidates.append(code)
            if len(candidates) >= 2:
                break
        if len(candidates) >= 2:
            return candidates[0], candidates[1]
        if len(candidates) == 1:
            return candidates[0], "ARG"
        return "BRA", "ARG"

    @staticmethod
    def _extract_match_date(task: str) -> "date_cls | None":
        """Extract YYYY-MM-DD from kickoff / match-date lines if present."""
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", task)
        if not m:
            return None
        try:
            y, mo, d = map(int, m.groups())
            return date_cls(y, mo, d)
        except ValueError:
            return None
