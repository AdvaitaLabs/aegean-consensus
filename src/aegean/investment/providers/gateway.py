from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List


_DEFAULT_PROVIDER_TIMEOUTS = {
    "yfinance": 3.0,
    "fmp": 4.0,
    "finnhub": 4.0,
    "tushare": 4.0,
    "coingecko": 4.0,
    "tavily": 6.0,
    "exa": 6.0,
    "serpapi": 6.0,
}

_ASSET_PROVIDER_ORDER = {
    "equity": ["market", "fundamentals", "news", "search", "search_fallback", "search_fallback_2"],
    "etf": ["market", "fundamentals", "news", "search", "search_fallback", "search_fallback_2"],
    "index": ["market", "news", "search", "search_fallback", "search_fallback_2"],
    "fund": ["market", "fundamentals", "news", "search"],
    "convertible_bond": ["market", "fundamentals", "news", "search"],
    "futures": ["market", "news", "search", "search_fallback", "search_fallback_2"],
    "options": ["market", "news", "search"],
    "crypto": ["market", "news", "search", "search_fallback", "search_fallback_2"],
}


class InvestmentDataGateway:
    """Unified in-process gateway for external investment data providers."""

    def __init__(self, providers: Dict[str, Any]):
        self.providers = providers
        for provider in providers.values():
            name = getattr(provider, "provider_name", "")
            if name in _DEFAULT_PROVIDER_TIMEOUTS and getattr(provider, "timeout_seconds", None) == 4.0:
                provider.timeout_seconds = _DEFAULT_PROVIDER_TIMEOUTS[name]

    async def fetch_all(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        selected = self._providers_for_asset(asset_type, market)
        tasks = {
            name: asyncio.create_task(provider.fetch(symbol, market, asset_type))
            for name, provider in selected.items()
        }
        results: Dict[str, Any] = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as exc:
                provider = selected[name]
                if hasattr(provider, "_error_result"):
                    results[name] = provider._error_result(symbol, market, asset_type, str(exc))
                else:
                    results[name] = {
                        "provider": name,
                        "market_data": {"symbol": symbol, "market": market, "asset_type": asset_type},
                        "fundamentals": {},
                        "news": [],
                        "metadata": {
                            "status": "error",
                            "message": str(exc),
                            "signals": [f"{name.upper()}_FAILED"],
                        },
                    }
        return self._merge(results.values())

    def _providers_for_asset(self, asset_type: str, market: str | None = None) -> Dict[str, Any]:
        ordered_keys = _ASSET_PROVIDER_ORDER.get(asset_type, ["market", "fundamentals", "news"])
        if asset_type == "crypto" and market:
            market_key = "crypto_market"
        elif market == "CN":
            market_key = "cn_market"
            if asset_type in {"equity", "etf", "fund", "convertible_bond"}:
                fundamentals_key = "cn_fundamentals"
            else:
                fundamentals_key = "fundamentals"
        else:
            market_key = "market"
            fundamentals_key = "fundamentals"

        aliases = {
            "market": market_key,
            "fundamentals": fundamentals_key if 'fundamentals_key' in locals() else "fundamentals",
        }
        selected: Dict[str, Any] = {}
        for key in ordered_keys:
            provider_key = aliases.get(key, key)
            provider = self.providers.get(provider_key)
            if provider is not None:
                selected[provider_key] = provider
        return selected

    @staticmethod
    def _merge(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        merged_market: Dict[str, Any] = {}
        merged_fundamentals: Dict[str, Any] = {}
        merged_news: List[Dict[str, Any]] = []
        providers: Dict[str, str] = {}
        provider_status: Dict[str, Any] = {}

        for result in results:
            provider_name = str(result.get("provider") or "unknown")
            metadata = dict(result.get("metadata") or {})
            market_data = dict(result.get("market_data") or {})
            fundamentals = dict(result.get("fundamentals") or {})
            news = list(result.get("news") or [])

            for key, value in market_data.items():
                if value is not None and key not in merged_market:
                    merged_market[key] = value
            for key, value in fundamentals.items():
                if value is not None and key not in merged_fundamentals:
                    merged_fundamentals[key] = value
            merged_news.extend(item for item in news if item)
            providers[provider_name] = provider_name
            provider_status[provider_name] = metadata

        deduped_news: List[Dict[str, Any]] = []
        seen = set()
        for item in merged_news:
            key = (item.get("title"), item.get("url"), item.get("provider"))
            if key in seen:
                continue
            seen.add(key)
            deduped_news.append(item)
            if len(deduped_news) >= 10:
                break
        provider_signals = sorted(
            {
                str(signal)
                for meta in provider_status.values()
                if isinstance(meta, dict)
                for signal in meta.get("signals", [])
                if str(signal).strip()
            }
        )
        return {
            "market": merged_market,
            "fundamentals": merged_fundamentals,
            "news": deduped_news,
            "providers": providers,
            "provider_status": provider_status,
            "provider_signals": provider_signals,
        }
