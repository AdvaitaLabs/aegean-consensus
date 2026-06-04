"""PlayerAgent: squad availability and player form."""

from __future__ import annotations

from aegean.agents.sports.base_sports_agent import BaseSportsAgent


class PlayerAgent(BaseSportsAgent):
    """
    The squad analyst. Cares about who is on the pitch: star availability,
    injuries, suspensions, club form of key players, age/fatigue.

    Recommended LLM: GPT-5 (strong on chained reasoning about player impact).
    """

    ROLE = "player_specialist"
    DEFAULT_WEIGHT = 0.85
    SPECIALIZATION_HINT = (
        "You are a squad and player-form analyst.\n"
        "Focus on who actually steps onto the pitch:\n"
        "  - Star availability (a missing top scorer materially shifts probs)\n"
        "  - Injuries and suspensions in the announced lineup\n"
        "  - Recent club form of the named players, not their reputation\n"
        "  - Goalkeeper specifically: a star GK in good form is worth 0.3 xGA\n"
        "If the team's leading goalscorer is out, the probability of that "
        "team winning should drop meaningfully (roughly 5-12 percentage "
        "points depending on how dependent the team is on them)."
    )
