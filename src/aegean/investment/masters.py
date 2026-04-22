"""Master-investor persona prompts for the investment panel.

Inspired by ai-hedge-fund's per-investor agents (Buffett, Munger, Burry,
Lynch, Wood, Ackman, Graham, Fisher). Each persona brings a distinct
investing philosophy, so running the same analysis through several of
them surfaces framing disagreements that a homogeneous panel would miss.

Like :mod:`aegean.investment.debate`, this module holds *only* the
prompt builders and persona registry — orchestration stays in the
service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MasterPersona:
    key: str
    display_name: str
    philosophy: str
    signature_lens: str      # What they look for first in any company.
    output_bias: str         # Their default stance when evidence is mixed.


MASTER_PERSONAS: Dict[str, MasterPersona] = {
    "buffett": MasterPersona(
        key="buffett",
        display_name="Warren Buffett",
        philosophy=(
            "Buy wonderful businesses at fair prices. Demand a durable moat, "
            "honest and capable management, and a clear circle of competence. "
            "Hold forever when the thesis holds, sell instantly when the moat breaks."
        ),
        signature_lens=(
            "owner-earnings, return on tangible capital, pricing power, "
            "balance-sheet conservatism, and an intrinsic-value estimate with "
            "a margin of safety of at least 25%."
        ),
        output_bias=(
            "Refuse to act when the business is outside circle of competence or "
            "when the margin of safety is thin."
        ),
    ),
    "munger": MasterPersona(
        key="munger",
        display_name="Charlie Munger",
        philosophy=(
            "Invert, always invert. High-quality compounders bought at "
            "reasonable prices beat clever trades. Avoid stupidity rather "
            "than chasing brilliance."
        ),
        signature_lens=(
            "mental models, incentive structures, regulatory/structural moats, "
            "and ways the thesis could fail catastrophically."
        ),
        output_bias=(
            "Prefer to do nothing unless the setup is obviously good; be "
            "unusually harsh on management incentives and accounting."
        ),
    ),
    "burry": MasterPersona(
        key="burry",
        display_name="Michael Burry",
        philosophy=(
            "Deep contrarian value. Look where the crowd won't — hated "
            "sectors, distressed balance sheets, mispriced tails. If the "
            "short-side setup is better than the long, take it."
        ),
        signature_lens=(
            "enterprise value vs liquidation value, insider activity, "
            "short interest, debt maturity walls, and specific catalysts "
            "that force revaluation."
        ),
        output_bias=(
            "Willing to be early, willing to be alone, willing to be short. "
            "Refuses to own crowded consensus longs without a structural edge."
        ),
    ),
    "lynch": MasterPersona(
        key="lynch",
        display_name="Peter Lynch",
        philosophy=(
            "Invest in what you understand and can see working in the real "
            "world. Sort companies into fast-growers, stalwarts, cyclicals, "
            "turnarounds, and asset plays; the right metric depends on the bucket."
        ),
        signature_lens=(
            "PEG ratio, same-store-sales / unit-economics trajectory, insider "
            "buying, and the 'two-minute' thesis — can you explain it to a child?"
        ),
        output_bias=(
            "Bias toward under-followed, still-scaling names; suspicious of "
            "diworsifying management and Wall Street-darling multiples."
        ),
    ),
    "wood": MasterPersona(
        key="wood",
        display_name="Cathie Wood",
        philosophy=(
            "Disruptive innovation compounds exponentially. Look five years "
            "out, size up TAM at full adoption, and accept high short-term "
            "drawdowns to capture convexity."
        ),
        signature_lens=(
            "Wright's Law cost curves, addressable market expansion, R&D as "
            "a share of revenue, and platform optionality."
        ),
        output_bias=(
            "Tolerant of current unprofitability if the cost-curve / TAM math "
            "works; quick to size up on drawdowns when the thesis is intact."
        ),
    ),
}


def get_persona(key: str) -> MasterPersona:
    try:
        return MASTER_PERSONAS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown master persona '{key}'. "
            f"Known: {sorted(MASTER_PERSONAS)}"
        ) from exc


def available_personas() -> List[str]:
    return list(MASTER_PERSONAS.keys())


def _clip(text: str, limit: int = 800) -> str:
    text = (text or "").strip().replace("\r", "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_master_prompt(
    key: str,
    symbol: str,
    market: str,
    asset_type: str,
    horizon: str,
    analyst_summaries: List[str],
    provider_signals: Optional[List[str]] = None,
    group_context: str = "",
) -> str:
    """Prompt a single master persona to render a stance on ``symbol``.

    Output is schema-enforced so the caller can parse ACTION / CONFIDENCE /
    TARGET_EXPOSURE_PCT with the existing helpers in :mod:`debate`.
    """
    persona = get_persona(key)
    analyst_block = (
        "\n".join(f"- {line}" for line in analyst_summaries[:8])
        if analyst_summaries else "No analyst summaries available."
    )
    signals = ", ".join(provider_signals or []) or "none"
    context_block = f"\nGroup context:\n{_clip(group_context)}\n" if group_context else ""

    return (
        f"You are {persona.display_name}. Stay fully in character — your "
        f"philosophy, your vocabulary, your biases.\n\n"
        f"Philosophy: {persona.philosophy}\n"
        f"What you look at first: {persona.signature_lens}\n"
        f"Your default when the evidence is mixed: {persona.output_bias}\n\n"
        f"Target asset: {symbol} ({market}, {asset_type}). Horizon: {horizon}.\n\n"
        f"Analyst panel summaries:\n{analyst_block}\n\n"
        f"Provider signals: {signals}{context_block}\n\n"
        "Deliver your verdict in your own voice, but strictly in this schema:\n"
        "ACTION: <BUY|HOLD|SELL|WATCH>\n"
        "CONFIDENCE: <0.0-1.0>\n"
        "TARGET_EXPOSURE_PCT: <0-100>\n"
        "RATIONALE: <3-5 sentences, cite the specific lens you used>\n"
        "WOULD_PASS_IF: <one sentence describing what would flip your vote>"
    )


def build_panel_prompts(
    persona_keys: List[str],
    symbol: str,
    market: str,
    asset_type: str,
    horizon: str,
    analyst_summaries: List[str],
    provider_signals: Optional[List[str]] = None,
    group_context: str = "",
) -> List[Tuple[str, str]]:
    """Build (persona_key, prompt) pairs for a whole panel run."""
    out: List[Tuple[str, str]] = []
    for key in persona_keys:
        out.append(
            (
                key,
                build_master_prompt(
                    key=key,
                    symbol=symbol,
                    market=market,
                    asset_type=asset_type,
                    horizon=horizon,
                    analyst_summaries=analyst_summaries,
                    provider_signals=provider_signals,
                    group_context=group_context,
                ),
            )
        )
    return out
