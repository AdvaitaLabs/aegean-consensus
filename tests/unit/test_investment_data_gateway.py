from __future__ import annotations

import pytest

from aegean.investment.providers.gateway import InvestmentDataGateway
from aegean.investment.providers.tavily_provider import TavilyProvider
from aegean.investment.providers.exa_provider import ExaProvider
from aegean.investment.providers.serpapi_provider import SerpAPIProvider
from aegean.investment.providers.tushare_provider import TushareProvider
from aegean.investment.providers.coingecko_provider import CoinGeckoProvider
from aegean.investment.service import InvestmentAnalysisService


class _StubProvider:
    def __init__(self, payload):
        self.payload = payload

    async def fetch(self, symbol: str, market: str, asset_type: str):
        return self.payload


@pytest.mark.asyncio
async def test_gateway_merges_partial_provider_results() -> None:
    gateway = InvestmentDataGateway(
        {
            "market": _StubProvider(
                {
                    "provider": "yfinance",
                    "market_data": {"symbol": "AAPL", "price": 100.0, "change_pct": 0.02},
                    "fundamentals": {},
                    "news": [],
                    "metadata": {"status": "ok", "signals": []},
                }
            ),
            "fundamentals": _StubProvider(
                {
                    "provider": "fmp",
                    "market_data": {},
                    "fundamentals": {"pe_ttm": 20.0, "pb": 5.0},
                    "news": [],
                    "metadata": {"status": "ok", "signals": []},
                }
            ),
            "news": _StubProvider(
                {
                    "provider": "finnhub",
                    "market_data": {},
                    "fundamentals": {},
                    "news": ["headline a", "headline b"],
                    "metadata": {"status": "ok", "signals": []},
                }
            ),
        }
    )

    result = await gateway.fetch_all("AAPL", "US", "equity")

    assert result["market"]["price"] == 100.0
    assert result["fundamentals"]["pe_ttm"] == 20.0
    assert result["news"] == ["headline a", "headline b"]
    assert result["provider_status"]["yfinance"]["status"] == "ok"


@pytest.mark.asyncio
async def test_gateway_returns_error_metadata_for_failing_provider() -> None:
    class _FailingProvider:
        async def fetch(self, symbol: str, market: str, asset_type: str):
            raise RuntimeError("boom")

    gateway = InvestmentDataGateway({"market": _FailingProvider()})
    result = await gateway.fetch_all("AAPL", "US", "equity")

    assert result["provider_status"]["market"]["status"] == "error"
    assert "MARKET_FAILED" in result["provider_status"]["market"]["signals"]


def test_service_formats_provider_status_and_signals() -> None:
    provider_status = {
        "fmp": {"status": "unavailable", "signals": ["FMP_UNAVAILABLE"], "message": "missing key"},
        "finnhub": {"status": "ok", "signals": [], "message": ""},
    }

    summary = InvestmentAnalysisService._format_provider_status_for_prompt(provider_status)
    signals = InvestmentAnalysisService._collect_provider_signals(provider_status)

    assert "fmp: status=unavailable" in summary
    assert "FMP_UNAVAILABLE" in summary
    assert signals == ["FMP_UNAVAILABLE"]


def test_gateway_routes_index_without_fundamental_provider() -> None:
    gateway = InvestmentDataGateway(
        {
            "market": _StubProvider({"provider": "yfinance", "market_data": {"price": 100}, "fundamentals": {}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "fundamentals": _StubProvider({"provider": "fmp", "market_data": {}, "fundamentals": {"pe_ttm": 20}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "news": _StubProvider({"provider": "finnhub", "market_data": {}, "fundamentals": {}, "news": ["headline"], "metadata": {"status": "ok", "signals": []}}),
            "search": _StubProvider({"provider": "tavily", "market_data": {}, "fundamentals": {}, "news": ["search headline"], "metadata": {"status": "ok", "signals": []}}),
        }
    )

    selected = gateway._providers_for_asset("index")
    assert set(selected.keys()) == {"market", "news", "search"}


def test_gateway_routes_cn_equity_to_tushare() -> None:
    gateway = InvestmentDataGateway(
        {
            "market": _StubProvider({"provider": "yfinance", "market_data": {}, "fundamentals": {}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "cn_market": _StubProvider({"provider": "tushare", "market_data": {"price": 10}, "fundamentals": {}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "fundamentals": _StubProvider({"provider": "fmp", "market_data": {}, "fundamentals": {"pe_ttm": 20}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "cn_fundamentals": _StubProvider({"provider": "tushare", "market_data": {}, "fundamentals": {"pb": 2}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "news": _StubProvider({"provider": "finnhub", "market_data": {}, "fundamentals": {}, "news": ["headline"], "metadata": {"status": "ok", "signals": []}}),
            "search": _StubProvider({"provider": "tavily", "market_data": {}, "fundamentals": {}, "news": ["search headline"], "metadata": {"status": "ok", "signals": []}}),
        }
    )

    selected = gateway._providers_for_asset("equity", "CN")
    assert set(selected.keys()) == {"cn_market", "cn_fundamentals", "news", "search"}


def test_gateway_routes_crypto_to_coingecko_market() -> None:
    gateway = InvestmentDataGateway(
        {
            "market": _StubProvider({"provider": "yfinance", "market_data": {}, "fundamentals": {}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "crypto_market": _StubProvider({"provider": "coingecko", "market_data": {"price": 68000}, "fundamentals": {}, "news": [], "metadata": {"status": "ok", "signals": []}}),
            "news": _StubProvider({"provider": "finnhub", "market_data": {}, "fundamentals": {}, "news": ["headline"], "metadata": {"status": "ok", "signals": []}}),
            "search": _StubProvider({"provider": "tavily", "market_data": {}, "fundamentals": {}, "news": ["search headline"], "metadata": {"status": "ok", "signals": []}}),
        }
    )

    selected = gateway._providers_for_asset("crypto", "US")
    assert set(selected.keys()) == {"crypto_market", "news", "search"}


def test_tushare_uses_daily_as_price_fallback() -> None:
    basic_payload = {
        "data": {
            "fields": ["ts_code", "trade_date", "close", "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"],
            "items": [["600519.SH", "20260421", None, "1.2", "0.9", "30", "8", "20000", "18000"]],
        }
    }
    daily_payload = {
        "data": {
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"],
            "items": [["600519.SH", "20260421", "1500", "1510", "1490", "1505", "1490", "15", "1.01", "123.45", "999.0"]],
        }
    }

    assert TushareProvider._to_float(TushareProvider._first_row(basic_payload).get("close")) is None
    daily_row = TushareProvider._first_row(daily_payload)
    assert TushareProvider._to_float(daily_row.get("close")) == 1505.0
    assert TushareProvider._to_ratio(daily_row.get("pct_chg")) == 0.0101
    assert TushareProvider._scaled_volume(daily_row.get("vol")) == 12345.0


@pytest.mark.asyncio
async def test_tavily_provider_without_key_returns_unavailable() -> None:
    provider = TavilyProvider(api_key=None)
    result = await provider.fetch("AAPL", "US", "equity")
    assert result["metadata"]["status"] == "unavailable"
    assert "TAVILY_UNAVAILABLE" in result["metadata"]["signals"]


@pytest.mark.asyncio
async def test_exa_provider_without_key_returns_unavailable() -> None:
    provider = ExaProvider(api_key=None)
    result = await provider.fetch("AAPL", "US", "equity")
    assert result["metadata"]["status"] == "unavailable"
    assert "EXA_UNAVAILABLE" in result["metadata"]["signals"]


@pytest.mark.asyncio
async def test_serpapi_provider_without_key_returns_unavailable() -> None:
    provider = SerpAPIProvider(api_key=None)
    result = await provider.fetch("AAPL", "US", "equity")
    assert result["metadata"]["status"] == "unavailable"
    assert "SERPAPI_UNAVAILABLE" in result["metadata"]["signals"]


@pytest.mark.asyncio
async def test_tushare_provider_without_key_returns_unavailable() -> None:
    provider = TushareProvider(api_key=None)
    result = await provider.fetch("600519", "CN", "equity")
    assert result["metadata"]["status"] == "unavailable"
    assert "TUSHARE_UNAVAILABLE" in result["metadata"]["signals"]


@pytest.mark.asyncio
async def test_coingecko_provider_non_crypto_returns_unavailable() -> None:
    provider = CoinGeckoProvider(api_key=None)
    result = await provider.fetch("AAPL", "US", "equity")
    assert result["metadata"]["status"] == "unavailable"
    assert "COINGECKO_UNAVAILABLE" in result["metadata"]["signals"]


def test_coingecko_symbol_mapping_prefers_known_ids() -> None:
    assert CoinGeckoProvider._to_ratio("12.5") == 0.125
    assert CoinGeckoProvider._to_ratio(None) is None


def test_build_external_evidence_uses_news_and_provider_signals() -> None:
    evidence = InvestmentAnalysisService._build_external_evidence(
        {
            "news": [
                "Reuters - Apple launches new AI feature",
                "Bloomberg - Analysts warn of valuation risk",
            ],
            "provider_signals": ["FMP_UNAVAILABLE"],
        }
    )
    assert evidence["supporting"][0] == "Reuters - Apple launches new AI feature"
    assert "Bloomberg - Analysts warn of valuation risk" in evidence["negative"]


def test_build_external_evidence_nodes_creates_graph_friendly_nodes() -> None:
    nodes = InvestmentAnalysisService._build_external_evidence_nodes(
        {
            "supporting": ["Reuters - Apple launches new AI feature"],
            "negative": ["Bloomberg - Analysts warn of valuation risk"],
            "organizations": ["Reuters", "Bloomberg"],
        }
    )
    assert nodes[0].node_type == "evidence"
    assert any(node.node_type == "negative_evidence" for node in nodes)
    assert any(node.node_type == "organization" for node in nodes)

