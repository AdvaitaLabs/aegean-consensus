"""Adversarial debate prompt builders for investment analysis.

Models the TradingAgents pattern:
1. Bull vs Bear researcher alternate for N rounds, each reading the
   opposing side's prior argument.
2. Research Manager synthesizes an investment plan from the transcript.
3. Aggressive / Neutral / Conservative risk analysts debate the plan.
4. Chair / Portfolio Manager renders the final decision.

This module contains *only* the prompt builders — orchestration lives on
:class:`~aegean.investment.service.InvestmentAnalysisService`. Keeping the
prompts here makes them independently testable and easy to override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DebateContext:
    symbol: str
    market: str
    asset_type: str
    horizon: str
    risk_profile: str
    objective: str
    analyst_summaries: List[str] = field(default_factory=list)
    analyst_signals: Dict[str, str] = field(default_factory=dict)
    provider_signals: List[str] = field(default_factory=list)
    group_context: str = ""
    portfolio_risk_reasoning: str = ""
    candidate_action: str = ""
    candidate_target_exposure_pct: float = 0.0

    def header(self) -> str:
        return (
            f"Target asset: {self.symbol} ({self.market}, {self.asset_type}).\n"
            f"Holding horizon: {self.horizon}. "
            f"Risk profile: {self.risk_profile}. Objective: {self.objective}."
        )

    def analyst_block(self) -> str:
        if not self.analyst_summaries:
            return "No analyst summaries available."
        return "\n".join(f"- {line}" for line in self.analyst_summaries[:8])


def _clip(text: str, limit: int = 800) -> str:
    text = (text or "").strip().replace("\r", "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_researcher_prompt(
    stance: str,
    round_number: int,
    context: DebateContext,
    own_history: List[str],
    opponent_history: List[str],
) -> str:
    """Prompt for a Bull or Bear researcher at round ``round_number`` (1-indexed).

    Each side sees the *other* side's last argument and is asked to rebut
    it while strengthening their own case.
    """
    if stance not in ("bull", "bear"):
        raise ValueError("stance must be 'bull' or 'bear'")

    role_title = "Bull Researcher (arguing BUY)" if stance == "bull" else "Bear Researcher (arguing SELL)"
    opposing_label = "Bear" if stance == "bull" else "Bull"
    directive = (
        "Argue the strongest *buy* case. Ground every claim in analyst data, "
        "flag disconfirming evidence honestly, and directly rebut the Bear's "
        "last argument — do not ignore their points."
        if stance == "bull"
        else
        "Argue the strongest *sell / avoid* case. Ground every claim in "
        "analyst data, surface downside catalysts, and directly rebut the "
        "Bull's last argument — do not ignore their points."
    )

    opponent_line = _clip(opponent_history[-1]) if opponent_history else f"No {opposing_label} statement yet."
    own_line = _clip(own_history[-1]) if own_history else "This is your opening statement."

    return (
        f"You are the {role_title}. Round {round_number} of the investment debate.\n"
        f"{context.header()}\n\n"
        f"Analyst summaries (from the 4-role panel):\n{context.analyst_block()}\n\n"
        f"Provider signals: {', '.join(context.provider_signals) or 'none'}\n"
        f"{context.group_context}"
        f"\nOpposing side's last argument ({opposing_label}): {opponent_line}\n"
        f"Your last argument: {own_line}\n\n"
        f"{directive}\n"
        "Output:\n"
        "1) Your concise argument (<= 150 words).\n"
        "2) 2–3 pieces of supporting evidence (data or catalysts).\n"
        "3) The specific claim from the opposing side you are rebutting.\n"
        "End with a single token signal on its own line: BULL_BUY or BEAR_SELL."
    )


def build_research_manager_prompt(
    context: DebateContext,
    bull_transcript: List[str],
    bear_transcript: List[str],
) -> str:
    bull_joined = "\n\n".join(f"Bull-R{i+1}: {_clip(msg)}" for i, msg in enumerate(bull_transcript))
    bear_joined = "\n\n".join(f"Bear-R{i+1}: {_clip(msg)}" for i, msg in enumerate(bear_transcript))
    return (
        "You are the Research Manager. Your job is to read the Bull/Bear debate "
        "transcript below and produce an actionable investment plan.\n"
        f"{context.header()}\n\n"
        "Bull transcript:\n" + (bull_joined or "(empty)") + "\n\n"
        "Bear transcript:\n" + (bear_joined or "(empty)") + "\n\n"
        "Produce:\n"
        "- action: one of BUY / HOLD / SELL / WATCH (single token on its own line prefixed with 'ACTION:')\n"
        "- target_exposure_pct: percentage of portfolio value, 0-100 (single line prefixed with 'TARGET_EXPOSURE_PCT:')\n"
        "- key_conditions: 2–4 bullet points describing what must remain true for this plan\n"
        "- kill_switch: what single development would invalidate the thesis\n"
        "- confidence: 0.0–1.0 (single line prefixed with 'CONFIDENCE:')\n"
        "Be decisive. If the debate was close, say so and lean on the risk profile "
        f"({context.risk_profile}) as the tiebreaker."
    )


def build_risk_debate_prompt(
    stance: str,
    round_number: int,
    context: DebateContext,
    plan_summary: str,
    peer_statements: Dict[str, str],
) -> str:
    if stance not in ("aggressive", "neutral", "conservative"):
        raise ValueError("stance must be aggressive/neutral/conservative")

    persona = {
        "aggressive": (
            "You are the Aggressive Risk Analyst. You value alpha capture and "
            "believe the firm is chronically under-sized on conviction trades."
        ),
        "neutral": (
            "You are the Neutral Risk Analyst. You focus on expected value "
            "and challenge both overly bullish and overly cautious framings."
        ),
        "conservative": (
            "You are the Conservative Risk Analyst. You prioritize capital "
            "preservation, tail risk, and liquidity; you push back on size."
        ),
    }[stance]

    peers_block = "\n".join(
        f"- {peer}: {_clip(msg, 400)}"
        for peer, msg in peer_statements.items()
        if msg
    ) or "(no peer statements yet)"

    return (
        f"{persona} Round {round_number} of the risk debate.\n"
        f"{context.header()}\n\n"
        f"Research Manager's proposed plan:\n{_clip(plan_summary)}\n\n"
        f"Peer risk analysts' latest statements:\n{peers_block}\n\n"
        "Output:\n"
        "1) Your stance on the plan (support / modify / block) and why (<= 120 words).\n"
        "2) Concrete size adjustment you would make to target_exposure_pct (up, down, unchanged, by how much).\n"
        "3) One risk you think peers are underweighting.\n"
        "End with a single token line prefixed 'SIZE_ADJ:' followed by one of UP/DOWN/SAME."
    )


def build_chair_prompt(
    context: DebateContext,
    plan_summary: str,
    risk_statements: Dict[str, str],
) -> str:
    risk_block = "\n".join(
        f"- {persona}: {_clip(msg, 400)}"
        for persona, msg in risk_statements.items()
        if msg
    ) or "(no risk debate statements available)"

    return (
        "You are the Chair / Portfolio Manager. You have the final call.\n"
        f"{context.header()}\n\n"
        f"Research Manager plan:\n{_clip(plan_summary)}\n\n"
        f"Risk debate statements:\n{risk_block}\n\n"
        f"Portfolio-risk engine notes: {context.portfolio_risk_reasoning or 'n/a'}\n\n"
        "Render the final decision. Weigh the three risk stances against the "
        "client's risk profile and the Research Manager's plan. If you diverge "
        "from the plan, say why in one sentence.\n\n"
        "Output strictly:\n"
        "ACTION: <BUY|HOLD|SELL|WATCH>\n"
        "TARGET_EXPOSURE_PCT: <0-100>\n"
        "CONFIDENCE: <0.0-1.0>\n"
        "RATIONALE: <2-3 sentences>\n"
        "KILL_SWITCH: <one sentence>"
    )


def parse_target_exposure_pct(text: str) -> Optional[float]:
    for line in (text or "").splitlines():
        upper = line.upper()
        if "TARGET_EXPOSURE_PCT" in upper:
            _, _, value = line.partition(":")
            value = value.strip().rstrip("%").strip()
            try:
                pct = float(value)
            except ValueError:
                continue
            if pct > 1.0:
                pct = pct / 100.0
            return max(0.0, min(1.0, pct))
    return None


def parse_confidence(text: str) -> Optional[float]:
    for line in (text or "").splitlines():
        upper = line.upper()
        if "CONFIDENCE" in upper:
            _, _, value = line.partition(":")
            value = value.strip().rstrip("%").strip()
            try:
                c = float(value)
            except ValueError:
                continue
            if c > 1.0:
                c = c / 100.0
            return max(0.0, min(1.0, c))
    return None
