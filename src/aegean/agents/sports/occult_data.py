"""
Data helpers for the entertainment OccultAgent: tarot deck + zodiac
horoscope fetcher.

These are pure helpers - they have no opinion about football. The agent
combines their output with the match prompt and lets the LLM weave a
playful narrative.
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ----------------------------- tarot deck -----------------------------

# Standard 78-card Rider-Waite-Smith deck. Each entry: name + a short
# "keyword" hint the LLM can lean on.
TAROT_DECK: List[Dict[str, str]] = [
    # Major Arcana (22)
    {"name": "The Fool",            "keyword": "new beginning, unexpected outcome"},
    {"name": "The Magician",        "keyword": "skill, talent breakthrough"},
    {"name": "The High Priestess",  "keyword": "intuition, hidden tactical depth"},
    {"name": "The Empress",         "keyword": "abundance of chances created"},
    {"name": "The Emperor",         "keyword": "discipline, structured defence"},
    {"name": "The Hierophant",      "keyword": "tradition, conservative approach"},
    {"name": "The Lovers",          "keyword": "harmony in midfield partnership"},
    {"name": "The Chariot",         "keyword": "momentum, willpower advantage"},
    {"name": "Strength",            "keyword": "stamina, late-game resilience"},
    {"name": "The Hermit",          "keyword": "lone striker decides it"},
    {"name": "Wheel of Fortune",    "keyword": "luck swing, deflections matter"},
    {"name": "Justice",             "keyword": "VAR / refereeing in spotlight"},
    {"name": "The Hanged Man",      "keyword": "pause, unexpected tactical shift"},
    {"name": "Death",               "keyword": "end of an era, reset"},
    {"name": "Temperance",          "keyword": "balanced game, draws likely"},
    {"name": "The Devil",           "keyword": "indiscipline, red card risk"},
    {"name": "The Tower",           "keyword": "sudden collapse, big upset"},
    {"name": "The Star",            "keyword": "hope, rising young player shines"},
    {"name": "The Moon",            "keyword": "uncertainty, illusions in the press"},
    {"name": "The Sun",             "keyword": "clarity, dominant performance"},
    {"name": "Judgement",           "keyword": "decisive moment near full time"},
    {"name": "The World",           "keyword": "completion, tournament-defining"},
]
# Pad the minor arcana programmatically; LLM doesn't need full Waite keywords.
for suit, motif in (
    ("Cups", "emotion, team chemistry"),
    ("Pentacles", "physical condition, fitness"),
    ("Swords", "intellect, tactical clarity"),
    ("Wands", "energy, attacking momentum"),
):
    for rank in (
        "Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
        "Page", "Knight", "Queen", "King",
    ):
        TAROT_DECK.append(
            {"name": f"{rank} of {suit}", "keyword": f"{rank.lower()} energy in {motif}"}
        )

assert len(TAROT_DECK) == 78, f"deck should have 78 cards, got {len(TAROT_DECK)}"


def draw_tarot_spread(rng: Optional[random.Random] = None, n: int = 3) -> List[Dict[str, str]]:
    """
    Draw a Celtic-Cross-mini spread: past / present / future.

    Args:
        rng: optional seeded RNG for deterministic test draws.
        n: how many cards (default 3, classic past-present-future).

    Returns:
        List of cards with positional meaning attached.
    """
    rng = rng or random.Random()
    positions = ["past", "present", "future"][:n] or [f"slot_{i}" for i in range(n)]
    drawn = rng.sample(TAROT_DECK, k=n)
    return [
        {**card, "position": positions[i]}
        for i, card in enumerate(drawn)
    ]


# ----------------------------- zodiac -----------------------------

# Captain birthday lookup table. We don't ship a personal-data database;
# instead we store the captain's *star sign* per FIFA code (publicly known).
# Update this annually before the tournament.
TEAM_CAPTAIN_ZODIAC: Dict[str, str] = {
    "BRA": "Cancer",       # Marquinhos
    "ARG": "Cancer",       # Messi
    "FRA": "Sagittarius",  # Mbappe (rotating, illustrative)
    "GER": "Virgo",        # Kimmich
    "ESP": "Gemini",       # Rodri
    "ENG": "Capricorn",    # Kane
    "POR": "Aquarius",     # Ronaldo
    "NED": "Cancer",       # Van Dijk
    "BEL": "Libra",        # De Bruyne
    "ITA": "Pisces",       # Donnarumma
    "CRO": "Virgo",        # Modric (illustrative)
    "URU": "Sagittarius",  # Bentancur
    "MEX": "Scorpio",
    "USA": "Leo",
    "JPN": "Pisces",
    "KOR": "Cancer",
}


# Generic mock horoscope lines used when the live API is unavailable.
# Real aztro service returns a "description" field; this fallback mimics it.
MOCK_HOROSCOPES: Dict[str, Dict[str, str]] = {
    "Aries":       {"description": "Bold attacking energy; quick early goals favoured.", "mood": "Confident", "lucky_number": "7"},
    "Taurus":      {"description": "Defensive solidity wins out; expect grinding play.", "mood": "Patient", "lucky_number": "12"},
    "Gemini":      {"description": "Quick passing and tactical fluidity.", "mood": "Adaptive", "lucky_number": "23"},
    "Cancer":      {"description": "Emotional surge for the captain; home advantage emphasised.", "mood": "Protective", "lucky_number": "9"},
    "Leo":         {"description": "Star striker steps up under pressure.", "mood": "Proud", "lucky_number": "5"},
    "Virgo":       {"description": "Precision passing; few errors.", "mood": "Focused", "lucky_number": "11"},
    "Libra":       {"description": "Balanced midfield; expect a tight scoreline.", "mood": "Diplomatic", "lucky_number": "6"},
    "Scorpio":     {"description": "Intense battle; late comeback possible.", "mood": "Intense", "lucky_number": "18"},
    "Sagittarius": {"description": "Long-range strike likely; ambitious play.", "mood": "Optimistic", "lucky_number": "21"},
    "Capricorn":   {"description": "Disciplined approach; favourite holds nerve.", "mood": "Determined", "lucky_number": "8"},
    "Aquarius":    {"description": "Surprise tactical change pays off.", "mood": "Inventive", "lucky_number": "4"},
    "Pisces":      {"description": "Goalkeeper plays a hero role.", "mood": "Dreamy", "lucky_number": "3"},
}


def fetch_horoscope(sign: str, mock: bool = True, timeout: float = 3.0) -> Dict[str, str]:
    """
    Pull today's horoscope for a zodiac sign.

    Args:
        sign: e.g. "Cancer" (case-insensitive).
        mock: when True, return the deterministic MOCK_HOROSCOPES entry.
        timeout: HTTP timeout for the live call.

    Returns:
        {"description", "mood", "lucky_number", "sign", "source"}
    """
    sign_canonical = sign.capitalize()
    fallback = MOCK_HOROSCOPES.get(sign_canonical, MOCK_HOROSCOPES["Cancer"])
    if mock:
        return {**fallback, "sign": sign_canonical, "source": "mock"}

    # Live: aztro project (free, no key needed)
    try:
        import requests
        url = "https://aztro.sameerkumar.website"
        resp = requests.post(
            url,
            params={"sign": sign_canonical.lower(), "day": "today"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "description": data.get("description", fallback["description"]),
            "mood": data.get("mood", fallback["mood"]),
            "lucky_number": str(data.get("lucky_number", fallback["lucky_number"])),
            "sign": sign_canonical,
            "source": "aztro",
        }
    except Exception as e:
        logger.warning("aztro fetch failed for %s (%s); using mock", sign, e)
        return {**fallback, "sign": sign_canonical, "source": "mock"}


# ----------------------------- entry point -----------------------------


def build_mystic_context(
    home_fifa: str,
    away_fifa: str,
    match_date: Optional[date] = None,
    rng: Optional[random.Random] = None,
    mock_horoscope: bool = True,
) -> Dict[str, Any]:
    """
    Compose everything the OccultAgent needs into one dict.

    Args:
        home_fifa / away_fifa: 3-letter team codes
        match_date: defaults to today
        rng: optional RNG for deterministic tarot draws
        mock_horoscope: whether to use the offline mock

    Returns:
        {
          "tarot_spread": [...],
          "home_zodiac": {...},
          "away_zodiac": {...},
          "numerology": {...},
        }
    """
    match_date = match_date or date.today()
    rng = rng or random.Random(f"{home_fifa}{away_fifa}{match_date.isoformat()}")
    tarot = draw_tarot_spread(rng=rng, n=3)
    home_sign = TEAM_CAPTAIN_ZODIAC.get(home_fifa, "Leo")
    away_sign = TEAM_CAPTAIN_ZODIAC.get(away_fifa, "Leo")
    home_horoscope = fetch_horoscope(home_sign, mock=mock_horoscope)
    away_horoscope = fetch_horoscope(away_sign, mock=mock_horoscope)
    # Simple numerology: digit sum of date
    digits = [int(c) for c in match_date.isoformat() if c.isdigit()]
    numerology_number = sum(digits) % 9 or 9
    return {
        "tarot_spread": tarot,
        "home_zodiac": home_horoscope,
        "away_zodiac": away_horoscope,
        "numerology": {
            "number": numerology_number,
            "match_date": match_date.isoformat(),
        },
    }


def format_mystic_context_for_prompt(ctx: Dict[str, Any]) -> str:
    """Human-readable summary suitable for embedding into the agent prompt."""
    lines: List[str] = []
    lines.append("MYSTIC CONTEXT (entertainment only):")
    lines.append(f"  Numerology of match date: {ctx['numerology']['number']}")
    lines.append(
        f"  Home captain sign ({ctx['home_zodiac']['sign']}): "
        f"{ctx['home_zodiac']['description']} (mood: {ctx['home_zodiac']['mood']})"
    )
    lines.append(
        f"  Away captain sign ({ctx['away_zodiac']['sign']}): "
        f"{ctx['away_zodiac']['description']} (mood: {ctx['away_zodiac']['mood']})"
    )
    lines.append("  Tarot spread (past / present / future):")
    for card in ctx["tarot_spread"]:
        lines.append(
            f"    {card['position']:8} - {card['name']} "
            f"({card['keyword']})"
        )
    return "\n".join(lines)
