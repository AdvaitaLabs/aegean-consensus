"""Integration-style unit tests for service.py hook-ins:

- ``_format_document_context_for_prompt`` (static, just text shaping)
- ``_augment_with_sentiment`` (mocks the insider endpoint to verify
  sentiment gets attached to normalized data)
- ``_run_masters_panel`` (mocks an Agent and verifies a masters_panel
  discussion round gets appended when panel_type='masters')
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from aegean.core.agent import Agent, AgentRegistry
from aegean.core.models import Solution
from aegean.investment.debate import DebateContext
from aegean.investment.models import (
    AssetType,
    InvestmentAnalysisRequest,
    InvestmentAsset,
    InvestmentTimeframe,
    MarketCode,
)
from aegean.investment.service import InvestmentAnalysisService


class _FakeAgent(Agent):
    """Minimal Agent that returns a canned solution shape."""

    def __init__(self, agent_id: str, canned: str = "ACTION: BUY\nCONFIDENCE: 0.8\n", role: str | None = None):
        super().__init__(agent_id=agent_id, capability_weight=1.0, role=role)
        self._canned = canned
        self.calls: List[str] = []

    async def generate_solution(self, task: str) -> Solution:
        self.calls.append(task)
        return Solution(
            agent_id=self.agent_id,
            answer=self._canned,
            confidence=0.8,
            reasoning="fake",
        )

    async def refine_solution(self, refinement_set: List[Solution]) -> Solution:
        return await self.generate_solution("refine")


def _build_service() -> InvestmentAnalysisService:
    return InvestmentAnalysisService(agent_registry=AgentRegistry())


def _request(market: MarketCode = MarketCode.US, metadata: Dict[str, Any] | None = None) -> InvestmentAnalysisRequest:
    return InvestmentAnalysisRequest(
        asset=InvestmentAsset(symbol="AAPL", market=market, asset_type=AssetType.EQUITY),
        timeframe=InvestmentTimeframe(),
        metadata=metadata or {},
    )


# ---- _format_document_context_for_prompt ----------------------------------

def test_format_document_context_empty_returns_empty():
    assert InvestmentAnalysisService._format_document_context_for_prompt(None) == ""
    assert InvestmentAnalysisService._format_document_context_for_prompt([]) == ""


def test_format_document_context_embeds_section_and_source():
    block = InvestmentAnalysisService._format_document_context_for_prompt(
        [
            {"text": "Supply chain risk material.", "section": "Risk Factors", "source": "10k.pdf"},
            {"text": "Revenue grew 12%.", "section": "MD&A", "source": "10k.pdf"},
        ]
    )
    assert "Ingested document excerpts" in block
    assert "section=Risk Factors" in block
    assert "source=10k.pdf" in block
    assert "Supply chain risk material." in block
    assert "Revenue grew 12%." in block


def test_format_document_context_truncates_long_text():
    long_text = "x" * 2000
    block = InvestmentAnalysisService._format_document_context_for_prompt(
        [{"text": long_text, "section": "S", "source": "f.pdf"}],
        max_chars_per_chunk=500,
    )
    assert "..." in block
    assert len(block) < 2000


def test_format_document_context_skips_bad_entries():
    block = InvestmentAnalysisService._format_document_context_for_prompt(
        ["not a dict", {"text": ""}, {"text": "real content"}]
    )
    assert "real content" in block
    assert "not a dict" not in block


# ---- _augment_with_sentiment ---------------------------------------------

def test_augment_with_sentiment_us_uses_finnhub_insider(monkeypatch):
    service = _build_service()

    async def fake_insider(symbol: str, limit: int = 50) -> Dict[str, Any]:
        return {
            "status": "ok",
            "data": [
                {"name": "CEO", "change": 1000},
                {"name": "CFO", "change": 500},
            ],
            "signals": [],
            "message": "",
        }

    monkeypatch.setattr(service.news_data_provider, "fetch_insider_transactions", fake_insider)

    normalized: Dict[str, Any] = {
        "news": [
            {"title": "beat", "polarity": "positive"},
            {"title": "raise", "polarity": "positive"},
        ],
        "provider_status": {},
        "provider_signals": [],
    }
    asyncio.run(service._augment_with_sentiment(_request(MarketCode.US), normalized))

    assert normalized["sentiment"]["signal"] == "bullish"
    assert "SENTIMENT_BULLISH" in normalized["provider_signals"]
    assert normalized["provider_status"]["sentiment"]["signals"] == ["SENTIMENT_BULLISH"]
    assert normalized["sentiment"]["insider_trading"]["metrics"]["bullish"] == 2


def test_augment_with_sentiment_cn_uses_tushare_insider(monkeypatch):
    service = _build_service()

    async def fake_insider(symbol: str, limit: int = 50) -> Dict[str, Any]:
        return {
            "status": "ok",
            "data": [
                {"in_de": "DE", "change_vol": 50000, "holder_name": "x"},
                {"in_de": "DE", "change_vol": 30000, "holder_name": "y"},
            ],
            "signals": [],
            "message": "",
        }

    monkeypatch.setattr(service.cn_market_data_provider, "fetch_insider_transactions", fake_insider)

    normalized: Dict[str, Any] = {
        "news": [{"title": "down", "polarity": "negative"}],
        "provider_status": {},
        "provider_signals": [],
    }
    req = _request(MarketCode.CN)
    req.asset.symbol = "600519"
    asyncio.run(service._augment_with_sentiment(req, normalized))

    assert normalized["sentiment"]["signal"] == "bearish"
    assert "SENTIMENT_BEARISH" in normalized["provider_signals"]
    assert normalized["sentiment"]["insider_trading"]["metrics"]["bearish"] == 2


def test_augment_with_sentiment_no_data_noop(monkeypatch):
    service = _build_service()

    async def empty_insider(symbol: str, limit: int = 50) -> Dict[str, Any]:
        return {"status": "ok", "data": [], "signals": [], "message": ""}

    monkeypatch.setattr(service.news_data_provider, "fetch_insider_transactions", empty_insider)
    normalized: Dict[str, Any] = {"news": [], "provider_status": {}, "provider_signals": []}
    asyncio.run(service._augment_with_sentiment(_request(MarketCode.US), normalized))
    assert "sentiment" not in normalized
    assert normalized["provider_signals"] == []


def test_augment_swallows_provider_errors(monkeypatch):
    service = _build_service()

    async def boom(symbol: str, limit: int = 50):
        raise RuntimeError("network down")

    monkeypatch.setattr(service.news_data_provider, "fetch_insider_transactions", boom)
    normalized: Dict[str, Any] = {
        "news": [{"title": "x", "polarity": "positive"}],
        "provider_status": {},
        "provider_signals": [],
    }
    # Should not raise; sentiment may or may not attach depending on failure point,
    # but critically the pipeline does not crash.
    asyncio.run(service._augment_with_sentiment(_request(MarketCode.US), normalized))


# ---- _run_masters_panel ---------------------------------------------------

def _ctx() -> DebateContext:
    return DebateContext(
        symbol="AAPL",
        market="US",
        asset_type="equity",
        horizon="1m",
        risk_profile="balanced",
        objective="alpha",
        analyst_summaries=["fundamental: bullish"],
        provider_signals=["yfinance_ok"],
    )


def test_masters_panel_skipped_when_panel_type_missing():
    service = _build_service()
    agent = _FakeAgent("chair-agent", role="chair")
    service.agent_registry.register_agent(agent)

    rounds: List[Any] = []
    events: List[Dict[str, Any]] = []

    async def emit(event_type: str, **payload: Any) -> None:
        events.append({"type": event_type, **payload})

    summaries = asyncio.run(
        service._run_masters_panel(
            _request(metadata={}),  # no panel_type
            _ctx(),
            bull_history=["bull-1"],
            bear_history=["bear-1"],
            rounds=rounds,
            emit=emit,
            start_round_index=2,
        )
    )
    assert summaries == []
    assert rounds == []
    assert events == []


def test_masters_panel_runs_all_requested_personas():
    service = _build_service()
    agent = _FakeAgent(
        "chair-agent",
        canned="ACTION: BUY\nCONFIDENCE: 0.72\nRATIONALE: moat intact.",
        role="chair",
    )
    service.agent_registry.register_agent(agent)

    rounds: List[Any] = []
    events: List[Dict[str, Any]] = []

    async def emit(event_type: str, **payload: Any) -> None:
        events.append({"type": event_type, **payload})

    req = _request(metadata={"panel_type": "masters", "master_personas": ["buffett", "burry"]})
    summaries = asyncio.run(
        service._run_masters_panel(
            req, _ctx(), ["bull-1"], ["bear-1"], rounds, emit, start_round_index=2,
        )
    )
    assert len(summaries) == 2
    assert summaries[0].startswith("master/buffett")
    assert summaries[1].startswith("master/burry")
    assert len(rounds) == 1
    assert rounds[0].stage == "masters_panel"
    assert rounds[0].round_number == 3
    assert {entry.role for entry in rounds[0].agents} == {"master_buffett", "master_burry"}
    assert any(e["type"] == "debate_round_finished" and e["stage"] == "masters_panel" for e in events)
    # Buffett + Burry prompts include their display names
    calls = " || ".join(agent.calls)
    assert "Warren Buffett" in calls and "Michael Burry" in calls


def test_masters_panel_drops_unknown_personas():
    service = _build_service()
    service.agent_registry.register_agent(_FakeAgent("chair-agent", role="chair"))

    async def emit(*a: Any, **kw: Any) -> None:
        return None

    req = _request(metadata={"panel_type": "masters", "master_personas": ["cramer", "nobody"]})
    summaries = asyncio.run(
        service._run_masters_panel(req, _ctx(), [], [], [], emit, start_round_index=0)
    )
    assert summaries == []


def test_masters_panel_noop_when_no_agent_registered():
    service = _build_service()

    async def emit(*a: Any, **kw: Any) -> None:
        return None

    req = _request(metadata={"panel_type": "masters"})
    summaries = asyncio.run(
        service._run_masters_panel(req, _ctx(), [], [], [], emit, start_round_index=0)
    )
    assert summaries == []
