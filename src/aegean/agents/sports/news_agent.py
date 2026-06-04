"""NewsAgent: pre-match news, morale, narrative."""

from __future__ import annotations

from aegean.agents.sports.base_sports_agent import BaseSportsAgent


class NewsAgent(BaseSportsAgent):
    """
    The pre-match news reader. Cares about intangibles that don't show
    up in stats: team morale, controversies, coach pressure, last-minute
    drama.

    Recommended LLM: Claude Haiku 4.5 (fast, cheap, summary-grade).
    """

    ROLE = "news_specialist"
    DEFAULT_WEIGHT = 0.65
    SPECIALIZATION_HINT = (
        "You are a football news analyst.\n"
        "Focus on intangibles the numbers can't capture:\n"
        "  - Team morale (recent winning/losing streak momentum)\n"
        "  - Locker-room drama, public disputes, coach under pressure\n"
        "  - Late roster changes announced near kickoff\n"
        "  - National-team political tension (host country sentiment, etc.)\n"
        "Acknowledge that intangibles have real but bounded effects: rarely "
        "shift outcome probabilities by more than 5-8 percentage points. "
        "Do not over-confide in news-based predictions."
    )
