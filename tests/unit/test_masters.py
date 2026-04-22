"""Unit tests for master-persona panel prompt builders."""

from __future__ import annotations

import pytest

from aegean.investment.masters import (
    MASTER_PERSONAS,
    available_personas,
    build_master_prompt,
    build_panel_prompts,
    get_persona,
)


def test_all_personas_have_required_fields():
    assert set(available_personas()) >= {"buffett", "munger", "burry", "lynch", "wood"}
    for key, persona in MASTER_PERSONAS.items():
        assert persona.key == key
        assert persona.display_name
        assert persona.philosophy
        assert persona.signature_lens
        assert persona.output_bias


def test_get_persona_rejects_unknown_key():
    with pytest.raises(ValueError):
        get_persona("cramer")


def test_build_master_prompt_embeds_persona_and_schema():
    prompt = build_master_prompt(
        key="buffett",
        symbol="AAPL",
        market="US",
        asset_type="equity",
        horizon="1y",
        analyst_summaries=[
            "fundamental_specialist [bullish, conf=0.7]: margins expanding",
        ],
        provider_signals=["yfinance_ok"],
    )
    assert "Warren Buffett" in prompt
    assert "margin of safety" in prompt
    assert "AAPL" in prompt and "1y" in prompt
    for key in ("ACTION:", "CONFIDENCE:", "TARGET_EXPOSURE_PCT:", "RATIONALE:", "WOULD_PASS_IF:"):
        assert key in prompt


def test_build_master_prompt_handles_empty_summaries():
    prompt = build_master_prompt(
        key="burry",
        symbol="TSLA",
        market="US",
        asset_type="equity",
        horizon="6m",
        analyst_summaries=[],
    )
    assert "Michael Burry" in prompt
    assert "No analyst summaries available." in prompt
    assert "Provider signals: none" in prompt


def test_panel_prompts_one_per_persona_and_persona_specific():
    pairs = build_panel_prompts(
        persona_keys=["buffett", "burry", "wood"],
        symbol="NVDA",
        market="US",
        asset_type="equity",
        horizon="1m",
        analyst_summaries=["a"],
    )
    assert [k for k, _ in pairs] == ["buffett", "burry", "wood"]
    buffett_prompt = pairs[0][1]
    burry_prompt = pairs[1][1]
    wood_prompt = pairs[2][1]
    assert "Warren Buffett" in buffett_prompt and "Michael Burry" not in buffett_prompt
    assert "Michael Burry" in burry_prompt
    assert "Cathie Wood" in wood_prompt and "Wright" in wood_prompt


def test_group_context_block_inserted_only_when_provided():
    with_ctx = build_master_prompt(
        "munger", "AAPL", "US", "equity", "1y",
        analyst_summaries=["x"], group_context="prior round flagged low confidence",
    )
    without_ctx = build_master_prompt(
        "munger", "AAPL", "US", "equity", "1y", analyst_summaries=["x"],
    )
    assert "Group context:" in with_ctx
    assert "prior round flagged" in with_ctx
    assert "Group context:" not in without_ctx
