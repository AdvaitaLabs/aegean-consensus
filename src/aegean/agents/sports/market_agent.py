"""MarketAgent: bookmaker odds and sharp money reader."""

from __future__ import annotations

from aegean.agents.sports.base_sports_agent import BaseSportsAgent


class MarketAgent(BaseSportsAgent):
    """
    The sharp betting market reader. Cares about what the market is pricing
    and where bookmakers disagree (sharp signal).

    Recommended LLM: DeepSeek-V3 (cheap, structured-data competent).
    """

    ROLE = "market_specialist"
    DEFAULT_WEIGHT = 0.75
    SPECIALIZATION_HINT = (
        "You are a sharp betting market reader.\n"
        "Start from the market's consensus implied probabilities (already "
        "margin-removed in the data). The market aggregates a lot of "
        "information, so DEVIATE from it only when the data clearly supports "
        "doing so.\n"
        "Pay attention to:\n"
        "  - Spread between bookmakers: if Pinnacle and Bet365 disagree by "
        "    >5%, treat with extra skepticism\n"
        "  - Margin: smaller margin = sharper book = more reliable signal\n"
        "  - Whether market consensus matches what raw data (xG, form) "
        "    suggests; if not, who is right?\n"
        "Your output should usually be within ~5% of market consensus unless "
        "you have a specific data-backed reason to diverge."
    )
