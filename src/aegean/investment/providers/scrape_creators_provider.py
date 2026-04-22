from __future__ import annotations

from typing import Any, Dict, List

import aiohttp

from .base import ExternalDataProvider


_BEARISH_TOKENS = ("risk", "warn", "fall", "drop", "probe", "lawsuit", "cut", "weak", "short", "bearish")
_BULLISH_TOKENS = ("beat", "raise", "upgrade", "surge", "rally", "buy", "bullish", "growth", "moat")


class ScrapeCreatorsProvider(ExternalDataProvider):
    """Pulls ticker-related social chatter (TikTok / Twitter / Reddit) via ScrapeCreators.

    The ScrapeCreators public search API returns heterogeneous JSON depending on
    the platform; we only read fields common to all (``caption``/``text``,
    ``url``, ``created_at``) and coerce them into our generic news shape so the
    sentiment pipeline can score them alongside finnhub/tushare news.
    """

    provider_name = "scrape_creators"
    search_url = "https://api.scrapecreators.com/v1/search"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 6.0):
        super().__init__(
            api_key=api_key or self._env("SCRAPE_CREATORS_API_KEY"),
            timeout_seconds=timeout_seconds,
        )

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing SCRAPE_CREATORS_API_KEY",
                "signals": ["SCRAPE_CREATORS_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"x-api-key": self.api_key, "accept": "application/json"}
        params = {"query": f"${symbol} {asset_type}", "limit": 8}
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 429:
                        return self._rate_limited_result(symbol, market, asset_type)
                    if response.status >= 400:
                        return self._error_result(
                            symbol, market, asset_type, f"HTTP {response.status}"
                        )
                    data = await response.json()
            items = self._extract_items(data)
            result["news"] = [
                self._news_item(
                    title=self._title_for(item),
                    source=item.get("platform", "ScrapeCreators"),
                    url=item.get("url", ""),
                    summary=item.get("caption") or item.get("text", ""),
                    published_at=item.get("created_at", ""),
                    polarity=self._polarity_for(item),
                )
                for item in items[:5]
                if self._title_for(item)
            ]
            result["metadata"]["status"] = "ok"
            return result
        except TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

    @staticmethod
    def _extract_items(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            for key in ("results", "data", "items", "posts"):
                value = data.get(key)
                if isinstance(value, list):
                    return [v for v in value if isinstance(v, dict)]
        if isinstance(data, list):
            return [v for v in data if isinstance(v, dict)]
        return []

    @staticmethod
    def _title_for(item: Dict[str, Any]) -> str:
        for key in ("title", "caption", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:160]
        return ""

    @staticmethod
    def _polarity_for(item: Dict[str, Any]) -> str:
        text = " ".join(
            str(item.get(key) or "") for key in ("caption", "text", "title")
        ).lower()
        if any(token in text for token in _BEARISH_TOKENS):
            return "negative"
        if any(token in text for token in _BULLISH_TOKENS):
            return "positive"
        return "neutral"
