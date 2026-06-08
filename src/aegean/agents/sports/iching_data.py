"""
I Ching helper data for IChingAgent.

The Bazi (four pillars) variant has been removed - we now only cast a
hexagram for each match. If you need to reintroduce bazi later see git
history for the original implementation.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any, Dict, List, Optional


# 64 hexagrams. Each entry: id (01-64), Chinese name, English keyword hint
# oriented to match outcomes.
HEXAGRAMS: List[Dict[str, str]] = [
    {"id": "01", "name": "乾",   "keyword": "creative force, dominant attack"},
    {"id": "02", "name": "坤",   "keyword": "receptive, defensive posture"},
    {"id": "03", "name": "屯",   "keyword": "difficulty at the start, slow open"},
    {"id": "04", "name": "蒙",   "keyword": "youthful inexperience, mistakes likely"},
    {"id": "05", "name": "需",   "keyword": "waiting, patient build-up"},
    {"id": "06", "name": "讼",   "keyword": "conflict, refereeing controversy"},
    {"id": "07", "name": "师",   "keyword": "the army, disciplined collective"},
    {"id": "08", "name": "比",   "keyword": "holding together, midfield bond"},
    {"id": "09", "name": "小畜", "keyword": "small accumulation, edge in margins"},
    {"id": "10", "name": "履",   "keyword": "treading carefully, away advantage"},
    {"id": "11", "name": "泰",   "keyword": "peace, balanced flow"},
    {"id": "12", "name": "否",   "keyword": "stagnation, scoreless first half"},
    {"id": "13", "name": "同人", "keyword": "fellowship, team chemistry"},
    {"id": "14", "name": "大有", "keyword": "great possessions, possession dominance"},
    {"id": "15", "name": "谦",   "keyword": "modesty, underdog overperforms"},
    {"id": "16", "name": "豫",   "keyword": "enthusiasm, momentum swings"},
    {"id": "17", "name": "随",   "keyword": "following, reactive tactics"},
    {"id": "18", "name": "蛊",   "keyword": "decay, slow start needs reset"},
    {"id": "19", "name": "临",   "keyword": "approach, decisive late attack"},
    {"id": "20", "name": "观",   "keyword": "observation, cautious opening"},
    {"id": "21", "name": "噬嗑", "keyword": "biting through, breakthrough"},
    {"id": "22", "name": "贲",   "keyword": "adornment, surface flair masks weakness"},
    {"id": "23", "name": "剥",   "keyword": "splitting apart, collapse"},
    {"id": "24", "name": "复",   "keyword": "return, comeback"},
    {"id": "25", "name": "无妄", "keyword": "innocence, no manipulation"},
    {"id": "26", "name": "大畜", "keyword": "great accumulation, stored power"},
    {"id": "27", "name": "颐",   "keyword": "nourishment, careful preparation"},
    {"id": "28", "name": "大过", "keyword": "great excess, over-extension"},
    {"id": "29", "name": "坎",   "keyword": "abysmal, repeated danger"},
    {"id": "30", "name": "离",   "keyword": "the clinging, brilliant attack"},
    {"id": "31", "name": "咸",   "keyword": "influence, mutual attraction"},
    {"id": "32", "name": "恒",   "keyword": "duration, sustained rhythm"},
    {"id": "33", "name": "遁",   "keyword": "retreat, parking the bus"},
    {"id": "34", "name": "大壮", "keyword": "great power, aggressive press"},
    {"id": "35", "name": "晋",   "keyword": "progress, climb up the table"},
    {"id": "36", "name": "明夷", "keyword": "darkening light, error in judgement"},
    {"id": "37", "name": "家人", "keyword": "the family, defensive solidity"},
    {"id": "38", "name": "睽",   "keyword": "opposition, formation clash"},
    {"id": "39", "name": "蹇",   "keyword": "obstruction, missed chances"},
    {"id": "40", "name": "解",   "keyword": "deliverance, tactical change works"},
    {"id": "41", "name": "损",   "keyword": "decrease, missing key player"},
    {"id": "42", "name": "益",   "keyword": "increase, returning star boosts squad"},
    {"id": "43", "name": "夬",   "keyword": "breakthrough, late winner"},
    {"id": "44", "name": "姤",   "keyword": "coming to meet, unexpected encounter"},
    {"id": "45", "name": "萃",   "keyword": "gathering together, crowd support"},
    {"id": "46", "name": "升",   "keyword": "pushing upward, climb in the second half"},
    {"id": "47", "name": "困",   "keyword": "oppression, low-scoring affair"},
    {"id": "48", "name": "井",   "keyword": "the well, deep squad depth"},
    {"id": "49", "name": "革",   "keyword": "revolution, complete tactical overhaul"},
    {"id": "50", "name": "鼎",   "keyword": "the cauldron, fortune favours"},
    {"id": "51", "name": "震",   "keyword": "thunder, sudden goal"},
    {"id": "52", "name": "艮",   "keyword": "keeping still, defensive lockdown"},
    {"id": "53", "name": "渐",   "keyword": "gradual progress, slow build"},
    {"id": "54", "name": "归妹", "keyword": "marrying maiden, mismatched roles"},
    {"id": "55", "name": "丰",   "keyword": "abundance, goalfest"},
    {"id": "56", "name": "旅",   "keyword": "the wanderer, neutral venue effect"},
    {"id": "57", "name": "巽",   "keyword": "the gentle, fluid passing"},
    {"id": "58", "name": "兑",   "keyword": "joyous, celebrating fans"},
    {"id": "59", "name": "涣",   "keyword": "dispersion, formation breakdown"},
    {"id": "60", "name": "节",   "keyword": "limitation, disciplined defence"},
    {"id": "61", "name": "中孚", "keyword": "inner truth, captain leads"},
    {"id": "62", "name": "小过", "keyword": "small excess, late drama"},
    {"id": "63", "name": "既济", "keyword": "after completion, lead held"},
    {"id": "64", "name": "未济", "keyword": "before completion, unfinished business"},
]
assert len(HEXAGRAMS) == 64


def draw_hexagram(rng: Optional[random.Random] = None) -> Dict[str, str]:
    """Cast a single I Ching hexagram."""
    rng = rng or random.Random()
    return dict(rng.choice(HEXAGRAMS))


def build_iching_context(
    home_fifa: str,
    away_fifa: str,
    match_date: Optional[date] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Compose I Ching context for an auto-mode agent prompt."""
    match_date = match_date or date.today()
    rng = rng or random.Random(f"{home_fifa}{away_fifa}{match_date.isoformat()}")
    return {
        "primary_hexagram": draw_hexagram(rng),
        "changing_hexagram": draw_hexagram(rng),
        "match_date": match_date.isoformat(),
    }


def format_iching_context_for_prompt(ctx: Dict[str, Any]) -> str:
    """Human-readable I Ching summary for embedding into the agent prompt."""
    lines: List[str] = ["I CHING CONTEXT (entertainment only):"]
    p = ctx.get("primary_hexagram") or {}
    c = ctx.get("changing_hexagram") or {}
    lines.append(
        f"  Primary hexagram: {p.get('name', '?')} ({p.get('id', '?')}) "
        f"- {p.get('keyword', '')}"
    )
    lines.append(
        f"  Changing hexagram: {c.get('name', '?')} ({c.get('id', '?')}) "
        f"- {c.get('keyword', '')}"
    )
    lines.append(f"  Match date: {ctx.get('match_date')}")
    return "\n".join(lines)
