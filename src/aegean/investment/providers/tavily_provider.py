from __future__ import annotations

from typing import Any, Dict

import aiohttp

from .base import ExternalDataProvider


class TavilyProvider(ExternalDataProvider):
    provider_name = "tavily"
    search_url = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 6.0):
        super().__init__(api_key=api_key or self._env("TAVILY_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing TAVILY_API_KEY",
                "signals": ["TAVILY_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        payload = {
            "api_key": self.api_key,
            "query": f"{symbol} {market} {asset_type} stock news catalysts risks",
            "search_depth": "basic",
            "topic": "news",
            "max_results": 5,
        }
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.search_url, json=payload) as response:
                    if response.status == 429:
                        return self._rate_limited_result(symbol, market, asset_type)
                    data = await response.json()

            items = data.get("results", []) if isinstance(data, dict) else []
            result["news"] = [
                self._news_item(
                    title=item.get("title") or item.get("url", ""),
                    source=item.get("url", "Tavily"),
                    url=item.get("url", ""),
                    summary=item.get("content", ""),
                    polarity="negative" if any(token in str(item.get("content", "")).lower() for token in ["risk", "warn", "fall", "drop", "probe", "lawsuit", "cut", "weak"]) else "neutral",
                    metadata={"score": item.get("score")},
                )
                for item in items[:5]
                if item.get("title") or item.get("url")
            ]
            result["metadata"]["status"] = "ok"
            return result
        except TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

