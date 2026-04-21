from __future__ import annotations

from typing import Any, Dict
from urllib.parse import quote_plus

import aiohttp

from .base import ExternalDataProvider


class SerpAPIProvider(ExternalDataProvider):
    provider_name = "serpapi"
    search_url = "https://serpapi.com/search.json"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 6.0):
        super().__init__(api_key=api_key or self._env("SERPAPI_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing SERPAPI_API_KEY",
                "signals": ["SERPAPI_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        params = {
            "engine": "google_news",
            "q": quote_plus(f"{symbol} {market} {asset_type} stock news risk catalyst"),
            "api_key": self.api_key,
        }
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 429:
                        return self._rate_limited_result(symbol, market, asset_type)
                    data = await response.json()
            items = data.get("news_results", []) if isinstance(data, dict) else []
            result["news"] = [
                self._news_item(
                    title=item.get("title"),
                    source=item.get("source", "SerpAPI"),
                    url=item.get("link", ""),
                    summary=item.get("snippet", ""),
                    published_at=item.get("date", ""),
                    polarity="negative" if any(token in str(item.get("snippet", "")).lower() for token in ["risk", "warn", "fall", "drop", "probe", "lawsuit", "cut", "weak"]) else "neutral",
                )
                for item in items[:5]
                if item.get("title")
            ]
            result["metadata"]["status"] = "ok"
            return result
        except TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

