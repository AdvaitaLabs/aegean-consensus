"""OccultAgent: zodiac, tarot, and football mysticism (entertainment)."""

from __future__ import annotations

from aegean.agents.sports.base_sports_agent import BaseSportsAgent


class OccultAgent(BaseSportsAgent):
    """
    The sports astrologer. Pure entertainment. Weight is intentionally
    near-zero (0.10) so it cannot materially shift consensus outcomes,
    but its predictions are shown to users for fun.

    Recommended LLM: Claude Haiku 4.5 (cheap, playful is enough).
    """

    ROLE = "occult_specialist"
    DEFAULT_WEIGHT = 0.10
    SPECIALIZATION_HINT = (
        "You are playing the role of a sports astrologer for entertainment "
        "purposes. Predict the match using:\n"
        "  - Numerology of the date (lucky/unlucky days)\n"
        "  - Star sign compatibility of the two team captains (if known)\n"
        "  - Team jersey colour symbolism in this lunar phase\n"
        "  - Generic mystical hand-waving\n"
        "Keep probabilities reasonable (do not output 0.99 / 0 / 0.01). "
        "Have fun with the rationale; users see this output as a novelty "
        "alongside the serious analysts. Your prediction has near-zero "
        "weight in the final consensus, so do not be afraid to make "
        "interesting calls."
    )
