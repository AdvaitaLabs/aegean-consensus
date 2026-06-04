"""
Tests for World Cup 2026 sports agents.

Uses a deterministic mock LLM client so tests are reproducible and never
hit a real network. The mock returns JSON in the strict contract format
expected by BaseSportsAgent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional

import pytest

from aegean.agents.sports import (
    ChatAgent,
    ChatFetcher,
    ChatQAHandler,
    DEFAULT_AGENT_MODEL_MAP,
    MarketAgent,
    NewsAgent,
    OccultAgent,
    PlayerAgent,
    StatsAgent,
    StrategyAgent,
    build_worldcup_agents,
)
from aegean.agents.sports.base_sports_agent import (
    _extract_json,
    _normalize_probs,
)
from aegean.core.agent import AgentRegistry
from aegean.core.models import Solution


# ----------------------------- mock LLM -----------------------------


class MockLLMClient:
    """Deterministic stand-in for an aegean LLM client."""

    def __init__(self, model_name: str = "mock", bias: float = 0.0):
        self.model_name = model_name
        self.bias = bias
        self.last_usage = {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000}
        self.calls: List[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        perturb_home = ((h % 100) / 1000.0) - 0.05
        perturb_away = (((h // 100) % 100) / 1000.0) - 0.05
        p_home = max(0.15, min(0.75, 0.40 + perturb_home + self.bias))
        p_away = max(0.15, min(0.75, 0.32 + perturb_away))
        p_draw = max(0.10, 1.0 - p_home - p_away)
        total = p_home + p_draw + p_away
        return json.dumps(
            {
                "p_home_win": round(p_home / total, 4),
                "p_draw": round(p_draw / total, 4),
                "p_away_win": round(p_away / total, 4),
                "confidence": round(max(p_home, p_draw, p_away) / total, 2),
                "rationale": f"[mock {self.model_name}] deterministic test response",
            }
        )


class MockQAClient:
    """Returns natural-language answers in JSON, for ChatQAHandler tests."""

    def __init__(self, model_name: str = "mock-qa"):
        self.model_name = model_name
        self.last_usage = {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380}

    async def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "answer": "Based on the data, Brazil has a slight edge.",
                "confidence": 0.7,
                "rationale": "Recent form favours Brazil.",
            }
        )


SAMPLE_TASK = """Match: Brazil (home) vs Argentina (away)
Competition: FIFA World Cup 2026
Stage: group
Match ID: WC2026-A1

BRA xG: 2.05, ARG xG: 2.10
BRA Elo 1981, ARG Elo 2114
Market consensus: H 38%, D 30%, A 32%
"""


# ----------------------------- helpers -----------------------------


def _parse_solution_answer(sol: Solution) -> Dict[str, float]:
    data = _extract_json(sol.answer)
    return {
        "p_home_win": float(data["p_home_win"]),
        "p_draw": float(data["p_draw"]),
        "p_away_win": float(data["p_away_win"]),
    }


# ----------------------------- JSON util -----------------------------


class TestJSONHelpers:
    def test_extract_clean_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_extract_fenced_json(self):
        text = 'Reasoning...\n```json\n{"a": 2}\n```\nDone.'
        assert _extract_json(text) == {"a": 2}

    def test_extract_embedded_json(self):
        text = "Here is my answer:\n\n{\"a\": 3, \"b\": 4}\n\nThanks."
        assert _extract_json(text) == {"a": 3, "b": 4}

    def test_normalize_clips_and_sums(self):
        probs = _normalize_probs(0.5, -0.1, 0.7)
        assert probs["p_draw"] == 0.0
        assert abs(sum(probs.values()) - 1.0) < 1e-9


# ----------------------------- predictor agents -----------------------------


@pytest.mark.parametrize(
    "agent_cls,expected_weight",
    [
        (StatsAgent, 0.90),
        (PlayerAgent, 0.85),
        (StrategyAgent, 0.80),
        (MarketAgent, 0.75),
        (NewsAgent, 0.65),
        (OccultAgent, 0.10),
    ],
)
def test_agent_weights_match_paper_safety_bound(agent_cls, expected_weight):
    """Each agent class declares its default capability weight."""
    agent = agent_cls(llm_client=MockLLMClient())
    assert agent.capability_weight == expected_weight


def test_aggregate_max_weight_below_50pct():
    """
    Refinement Validity holds iff max(w) / sum(w) < 0.5.
    Verify the sprint default weights satisfy this.
    """
    weights = [0.90, 0.85, 0.80, 0.75, 0.65, 0.10, 0.20]  # +chat
    max_share = max(weights) / sum(weights)
    assert max_share < 0.5, f"single agent has {max_share:.1%} of total weight"


@pytest.mark.parametrize(
    "agent_cls",
    [StatsAgent, PlayerAgent, StrategyAgent, MarketAgent, NewsAgent, OccultAgent],
)
def test_agent_generate_solution_returns_valid_probs(agent_cls):
    """Each predictor agent emits a normalised probability JSON via mock LLM."""
    agent = agent_cls(llm_client=MockLLMClient(model_name="mock"))
    sol = asyncio.run(agent.generate_solution(SAMPLE_TASK))
    assert isinstance(sol, Solution)
    probs = _parse_solution_answer(sol)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(0 <= p <= 1 for p in probs.values())
    assert sol.agent_id == agent_cls.ROLE


def test_agent_refine_with_peer_solutions():
    """refine_solution accepts peer solutions and produces a new Solution."""
    agent = StatsAgent(llm_client=MockLLMClient())
    peer_solutions = [
        Solution(
            agent_id="player_specialist",
            answer=json.dumps({"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2}),
            reasoning="Star striker available.",
            confidence=0.7,
        ),
        Solution(
            agent_id="market_specialist",
            answer=json.dumps({"p_home_win": 0.45, "p_draw": 0.28, "p_away_win": 0.27}),
            reasoning="Market consensus.",
            confidence=0.6,
        ),
    ]
    sol = asyncio.run(agent.refine_solution(peer_solutions))
    probs = _parse_solution_answer(sol)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_agent_without_llm_falls_back_uniform():
    """When llm_client is None, agent returns a uniform fallback Solution."""
    agent = StatsAgent(llm_client=None)
    sol = asyncio.run(agent.generate_solution(SAMPLE_TASK))
    probs = _parse_solution_answer(sol)
    assert probs["p_home_win"] == pytest.approx(1 / 3)
    assert probs["p_draw"] == pytest.approx(1 / 3)
    assert probs["p_away_win"] == pytest.approx(1 / 3)
    assert sol.metadata.get("fallback") is True


def test_agent_handles_llm_error_gracefully():
    class BrokenLLM:
        async def complete(self, prompt: str) -> str:
            raise RuntimeError("simulated network failure")

    agent = StatsAgent(llm_client=BrokenLLM())
    sol = asyncio.run(agent.generate_solution(SAMPLE_TASK))
    probs = _parse_solution_answer(sol)
    # Fallback to uniform
    assert all(p == pytest.approx(1 / 3) for p in probs.values())


# ----------------------------- ChatAgent -----------------------------


def _mock_chat_window(match_id: str) -> Dict[str, Any]:
    return {
        "match_id": match_id,
        "total_messages": 4,
        "messages": [
            {"user_name": "fanA", "text": "Brazil looks unstoppable", "language": "en"},
            {"user_name": "fanB", "text": "梅西今天感觉不太好", "language": "zh"},
            {"user_name": "fanC", "text": "go ARG!!", "language": "en"},
            {"user_name": "fanD", "text": "draw incoming", "language": "en"},
        ],
    }


def test_chat_agent_default_weight_low():
    agent = ChatAgent(llm_client=MockLLMClient())
    assert agent.capability_weight == 0.20


def test_chat_agent_calls_fetcher_and_includes_chat_in_prompt():
    llm = MockLLMClient()
    agent = ChatAgent(
        llm_client=llm,
        chat_fetch_fn=_mock_chat_window,
    )
    sol = asyncio.run(agent.generate_solution(SAMPLE_TASK))
    assert llm.calls, "LLM should have been called"
    assert "CROWD CHAT" in llm.calls[0]
    assert "Brazil looks unstoppable" in llm.calls[0]
    probs = _parse_solution_answer(sol)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_chat_agent_resilient_to_fetch_error():
    def broken_fetch(match_id):
        raise RuntimeError("chat service down")

    agent = ChatAgent(llm_client=MockLLMClient(), chat_fetch_fn=broken_fetch)
    sol = asyncio.run(agent.generate_solution(SAMPLE_TASK))
    # Should still produce a valid Solution (empty chat window)
    probs = _parse_solution_answer(sol)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


# ----------------------------- ChatQAHandler -----------------------------


def test_chat_qa_routes_to_named_agent():
    registry = AgentRegistry()
    qa_client = MockQAClient()
    stats = StatsAgent(llm_client=qa_client, model_name="claude-opus-4-7")
    registry.register_agent(stats)

    handler = ChatQAHandler(agent_registry=registry)
    resp = asyncio.run(
        handler.answer(
            agent_id="stats_specialist",
            question="will Brazil win their group?",
            match_context=SAMPLE_TASK,
            user_name="dongqi",
        )
    )
    assert resp.agent_id == "stats_specialist"
    assert "Brazil" in resp.answer
    assert resp.confidence == 0.7
    assert resp.tokens_used > 0


def test_chat_qa_unknown_agent_returns_friendly_error():
    registry = AgentRegistry()
    handler = ChatQAHandler(agent_registry=registry)
    resp = asyncio.run(handler.answer(agent_id="does_not_exist", question="hi"))
    assert resp.confidence == 0.0
    assert "no agent" in resp.answer.lower() or "unknown" in resp.metadata.get("error", "")


# ----------------------------- Factory -----------------------------


def test_factory_builds_seven_agents_with_chat():
    def fake_llm_factory(model_name: str):
        return MockLLMClient(model_name=model_name)

    agents = build_worldcup_agents(llm_factory=fake_llm_factory)
    assert len(agents) == 7
    roles = [a.ROLE for a in agents]
    assert roles[:6] == [
        "stats_specialist",
        "player_specialist",
        "strategy_specialist",
        "market_specialist",
        "news_specialist",
        "occult_specialist",
    ]
    assert roles[6] == "chat_specialist"


def test_factory_skips_chat_when_disabled():
    agents = build_worldcup_agents(llm_factory=None, include_chat=False)
    assert len(agents) == 6
    assert all(a.ROLE != "chat_specialist" for a in agents)


def test_factory_assigns_models_from_map():
    seen: List[str] = []

    def tracking_factory(model_name: str):
        seen.append(model_name)
        return MockLLMClient(model_name=model_name)

    build_worldcup_agents(llm_factory=tracking_factory)
    # Every entry in DEFAULT_AGENT_MODEL_MAP should appear at least once
    expected_models = set(DEFAULT_AGENT_MODEL_MAP.values())
    assert expected_models.issubset(set(seen)), f"got {seen}"


def test_factory_capability_weight_overrides():
    def factory(model_name: str):
        return MockLLMClient(model_name=model_name)

    agents = build_worldcup_agents(
        llm_factory=factory,
        capability_weight_overrides={"occult_specialist": 0.05},
    )
    occult = next(a for a in agents if a.ROLE == "occult_specialist")
    assert occult.capability_weight == 0.05
