"""Unit tests for debate prompt builders."""

from __future__ import annotations

import pytest

from aegean.investment.debate import (
    DebateContext,
    build_chair_prompt,
    build_research_manager_prompt,
    build_researcher_prompt,
    build_risk_debate_prompt,
    parse_confidence,
    parse_target_exposure_pct,
)


def _ctx() -> DebateContext:
    return DebateContext(
        symbol="AAPL",
        market="US",
        asset_type="equity",
        horizon="1m",
        risk_profile="balanced",
        objective="alpha",
        analyst_summaries=[
            "fundamental_specialist [bullish, conf=0.70]: earnings beat",
            "risk_specialist [neutral, conf=0.50]: moderate drawdown risk",
        ],
        provider_signals=["yfinance_ok"],
    )


def test_researcher_rejects_invalid_stance():
    with pytest.raises(ValueError):
        build_researcher_prompt("chicken", 1, _ctx(), [], [])


def test_bull_prompt_references_opponent_last_argument():
    ctx = _ctx()
    prompt = build_researcher_prompt(
        "bull",
        round_number=2,
        context=ctx,
        own_history=["Opening buy case: fundamentals strong."],
        opponent_history=["Bear opener: margin risk ahead."],
    )
    assert "Bull Researcher" in prompt
    assert "margin risk ahead" in prompt
    assert "Round 2" in prompt
    assert "BULL_BUY" in prompt or "BEAR_SELL" in prompt  # signal token block present


def test_bear_prompt_directive_is_sell_side():
    prompt = build_researcher_prompt("bear", 1, _ctx(), [], [])
    assert "sell / avoid" in prompt.lower() or "sell" in prompt.lower()
    assert "This is your opening statement." in prompt


def test_research_manager_prompt_includes_both_transcripts():
    prompt = build_research_manager_prompt(
        _ctx(),
        bull_transcript=["bull round 1", "bull round 2"],
        bear_transcript=["bear round 1"],
    )
    assert "Bull-R1: bull round 1" in prompt
    assert "Bull-R2: bull round 2" in prompt
    assert "Bear-R1: bear round 1" in prompt
    assert "ACTION:" in prompt
    assert "TARGET_EXPOSURE_PCT:" in prompt


def test_risk_debate_prompt_persona_and_peers():
    prompt = build_risk_debate_prompt(
        "conservative",
        1,
        _ctx(),
        plan_summary="BUY 5% with tight stop",
        peer_statements={"aggressive": "Go 15%!", "neutral": "Stay 5%."},
    )
    assert "Conservative Risk Analyst" in prompt
    assert "Go 15%!" in prompt
    assert "SIZE_ADJ:" in prompt


def test_chair_prompt_enforces_output_schema():
    prompt = build_chair_prompt(
        _ctx(),
        plan_summary="BUY 5%",
        risk_statements={"aggressive": "UP", "conservative": "DOWN"},
    )
    for key in ("ACTION:", "TARGET_EXPOSURE_PCT:", "CONFIDENCE:", "RATIONALE:", "KILL_SWITCH:"):
        assert key in prompt


def test_parse_target_exposure_pct_handles_percent_and_fraction():
    assert parse_target_exposure_pct("TARGET_EXPOSURE_PCT: 15%") == pytest.approx(0.15)
    assert parse_target_exposure_pct("target_exposure_pct: 0.08") == pytest.approx(0.08)
    assert parse_target_exposure_pct("no value here") is None
    assert parse_target_exposure_pct("TARGET_EXPOSURE_PCT: not-a-number") is None


def test_parse_confidence_clamps_and_normalizes():
    assert parse_confidence("CONFIDENCE: 0.72") == pytest.approx(0.72)
    assert parse_confidence("confidence: 85%") == pytest.approx(0.85)
    assert parse_confidence("nope") is None
