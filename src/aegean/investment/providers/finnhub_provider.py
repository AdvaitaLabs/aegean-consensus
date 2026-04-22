from __future__ import annotations

from typing import Any, Dict

import asyncio
import aiohttp

from .base import ExternalDataProvider


class FinnhubProvider(ExternalDataProvider):
    provider_name = "finnhub"
    quote_url = "https://finnhub.io/api/v1/quote"
    profile_url = "https://finnhub.io/api/v1/stock/profile2"
    basic_financials_url = "https://finnhub.io/api/v1/stock/metric"
    news_url = "https://finnhub.io/api/v1/company-news"
    insider_url = "https://finnhub.io/api/v1/stock/insider-transactions"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 4.0):
        super().__init__(api_key=api_key or self._env("FINNHUB_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing FINNHUB_API_KEY",
                "signals": ["FINNHUB_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        params = {"symbol": symbol, "token": self.api_key}
        basic_params = {"symbol": symbol, "metric": "all", "token": self.api_key}
        news_params = {"symbol": symbol, "from": "2025-01-01", "to": "2026-12-31", "token": self.api_key}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                quote_req = session.get(self.quote_url, params=params)
                profile_req = session.get(self.profile_url, params=params)
                basic_req = session.get(self.basic_financials_url, params=basic_params)
                news_req = session.get(self.news_url, params=news_params)
                quote_resp, profile_resp, basic_resp, news_resp = await asyncio.gather(
                    quote_req, profile_req, basic_req, news_req
                )
                responses = [quote_resp, profile_resp, basic_resp, news_resp]
                if any(resp.status == 429 for resp in responses):
                    return self._rate_limited_result(symbol, market, asset_type)
                async with quote_resp, profile_resp, basic_resp, news_resp:
                    quote_data, profile_data, basic_data, news_data = await asyncio.gather(
                        quote_resp.json(), profile_resp.json(), basic_resp.json(), news_resp.json()
                    )

            result["market_data"].update(
                {
                    "price": quote_data.get("c"),
                    "change_pct": quote_data.get("dp"),
                    "52w_high": quote_data.get("h"),
                    "52w_low": quote_data.get("l"),
                }
            )
            metric = basic_data.get("metric") if isinstance(basic_data, dict) else {}
            result["fundamentals"].update(
                {
                    "company_name": profile_data.get("name"),
                    "exchange": profile_data.get("exchange"),
                    "industry": profile_data.get("finnhubIndustry"),
                    "market_cap": profile_data.get("marketCapitalization"),
                    "eps": metric.get("epsTTM"),
                    "pb": metric.get("pbAnnual"),
                    "pe_ttm": metric.get("peTTM"),
                    "roe": metric.get("roeTTM"),
                    "net_margin": metric.get("netMargin"),
                    "52w_high": metric.get("52WeekHigh"),
                    "52w_low": metric.get("52WeekLow"),
                }
            )
            result["news"] = [
                self._news_item(
                    title=item.get("headline"),
                    source=item.get("source", "Finnhub"),
                    url=item.get("url", ""),
                    summary=item.get("summary", ""),
                    published_at=str(item.get("datetime", "")),
                    polarity="negative" if any(token in str(item.get("headline", "")).lower() for token in ["risk", "warn", "fall", "drop", "probe", "lawsuit", "cut", "weak"]) else "neutral",
                    metadata={"category": item.get("category", "")},
                )
                for item in news_data[:5]
                if item.get("headline")
            ] if isinstance(news_data, list) else []
            result["metadata"]["status"] = "ok"
            return result
        except asyncio.TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

    async def fetch_insider_transactions(
        self,
        symbol: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch recent insider transactions for ``symbol``.

        Returns a dict shaped as::

            {"status": "ok|unavailable|timeout|rate_limited|error",
             "data": [<raw finnhub rows>],
             "signals": [...]}

        The caller is responsible for turning ``data`` into
        :class:`InsiderTrade` objects via
        :func:`aegean.investment.sentiment.finnhub_insider_to_trades`.
        """
        if not self.api_key:
            return {
                "status": "unavailable",
                "data": [],
                "signals": ["FINNHUB_INSIDER_UNAVAILABLE"],
                "message": "Missing FINNHUB_API_KEY",
            }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        params = {"symbol": symbol, "limit": int(limit), "token": self.api_key}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.insider_url, params=params) as resp:
                    if resp.status == 429:
                        return {
                            "status": "rate_limited",
                            "data": [],
                            "signals": ["FINNHUB_INSIDER_RATE_LIMITED"],
                            "message": "Provider rate limited",
                        }
                    payload = await resp.json()
            rows = payload.get("data") if isinstance(payload, dict) else []
            return {
                "status": "ok",
                "data": rows or [],
                "signals": [],
                "message": "",
            }
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "data": [],
                "signals": ["FINNHUB_INSIDER_TIMEOUT"],
                "message": f"Timed out after {self.timeout_seconds}s",
            }
        except Exception as exc:
            return {
                "status": "error",
                "data": [],
                "signals": ["FINNHUB_INSIDER_FAILED"],
                "message": str(exc),
            }

