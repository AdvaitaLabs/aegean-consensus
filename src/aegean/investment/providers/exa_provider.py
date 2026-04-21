from __future__ import annotations

from typing import Any, Dict

import aiohttp

from .base import ExternalDataProvider


class ExaProvider(ExternalDataProvider):
    provider_name = "exa"
    search_url = "https://api.exa.ai/search"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 6.0):
        super().__init__(api_key=api_key or self._env("EXA_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing EXA_API_KEY",
                "signals": ["EXA_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        payload = {
            "query": f"{symbol} {market} {asset_type} latest news catalysts risks",
            "numResults": 5,
            "type": "auto",
        }
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.search_url, json=payload, headers=headers) as response:
                    if response.status == 429:
                        return self._rate_limited_result(symbol, market, asset_type)
                    data = await response.json()
            items = data.get("results", []) if isinstance(data, dict) else []
            result["news"] = [
                self._news_item(
                    title=item.get("title") or item.get("url", ""),
                    source=item.get("url", "Exa"),
                    url=item.get("url", ""),
                    summary=item.get("text", ""),
                    published_at=item.get("publishedDate", ""),
                    polarity="negative" if any(token in str(item.get("text", "")).lower() for token in ["risk", "warn", "fall", "drop", "probe", "lawsuit", "cut", "weak"]) else "neutral",
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

