"""StrategyAgent: tactics, formations, coaching matchups."""

from __future__ import annotations

from aegean.agents.sports.base_sports_agent import BaseSportsAgent


class StrategyAgent(BaseSportsAgent):
    """
    The tactics nerd. Cares about formation matchups, pressing intensity
    (read from PPDA), and head-to-head tactical history.

    Recommended LLM: Claude Opus 4.7 (good at tactical narrative).
    """

    ROLE = "strategy_specialist"
    DEFAULT_WEIGHT = 0.80
    SPECIALIZATION_HINT = (
        "You are a tactical football analyst.\n"
        "Focus on the strategic matchup:\n"
        "  - How does each team's preferred shape interact? "
        "(4-3-3 vs 3-5-2, etc.)\n"
        "  - Pressing intensity inferred from PPDA: lower PPDA means more "
        "aggressive press. A team that presses hard can dominate a "
        "possession-oriented opponent OR get exploited by a vertical one.\n"
        "  - Head-to-head pattern: have these teams' styles consistently "
        "produced certain types of results when they meet?\n"
        "  - Coach experience in big tournaments matters.\n"
        "Less weight on raw player quality (that's another agent's job)."
    )
