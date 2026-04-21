from __future__ import annotations

from typing import Any, Dict

import aiohttp

from .base import ExternalDataProvider


_COIN_SYMBOL_TO_ID = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "doge": "dogecoin",
    "ada": "cardano",
    "trx": "tron",
    "avax": "avalanche-2",
    "dot": "polkadot",
    "link": "chainlink",
    "matic": "matic-network",
}


class CoinGeckoProvider(ExternalDataProvider):
    provider_name = "coingecko"
    markets_url = "https://api.coingecko.com/api/v3/coins/markets"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 4.0):
        super().__init__(api_key=api_key or self._env("COINGECKO_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if asset_type != "crypto":
            result["metadata"] = {
                "status": "unavailable",
                "message": "CoinGecko is only enabled for crypto assets",
                "signals": ["COINGECKO_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                coin = await self._fetch_market_coin(session, symbol)
            if not coin:
                return self._error_result(symbol, market, asset_type, f"Coin not found for symbol={symbol}")

            result["market_data"].update(
                {
                    "symbol": str(coin.get("symbol") or symbol).upper(),
                    "price": coin.get("current_price"),
                    "change_pct": self._to_ratio(coin.get("price_change_percentage_24h")),
                    "market_cap": coin.get("market_cap"),
                    "volume": coin.get("total_volume"),
                    "fully_diluted_valuation": coin.get("fully_diluted_valuation"),
                    "24h_high": coin.get("high_24h"),
                    "24h_low": coin.get("low_24h"),
                }
            )
            result["fundamentals"].update(
                {
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "circulating_supply": coin.get("circulating_supply"),
                    "total_supply": coin.get("total_supply"),
                    "max_supply": coin.get("max_supply"),
                    "ath": coin.get("ath"),
                    "ath_change_pct": self._to_ratio(coin.get("ath_change_percentage")),
                    "atl": coin.get("atl"),
                    "atl_change_pct": self._to_ratio(coin.get("atl_change_percentage")),
                    "price_change_pct_7d": self._to_ratio(coin.get("price_change_percentage_7d_in_currency")),
                    "coingecko_id": coin.get("id"),
                }
            )
            result["metadata"]["status"] = "ok"
            return result
        except TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except aiohttp.ClientResponseError as exc:
            if exc.status == 429:
                return self._rate_limited_result(symbol, market, asset_type)
            return self._error_result(symbol, market, asset_type, str(exc))
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

    async def _fetch_market_coin(self, session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
        normalized_symbol = symbol.strip().lower()
        candidates = []
        mapped_id = _COIN_SYMBOL_TO_ID.get(normalized_symbol)
        if mapped_id:
            candidates.append({"ids": mapped_id})
        candidates.append({"symbols": normalized_symbol})

        seen = set()
        for extra in candidates:
            key = tuple(sorted(extra.items()))
            if key in seen:
                continue
            seen.add(key)
            coin = await self._request_market_coin(session, extra)
            if coin:
                return coin
        return {}

    async def _request_market_coin(self, session: aiohttp.ClientSession, selector: Dict[str, str]) -> Dict[str, Any]:
        params = {
            "vs_currency": "usd",
            "price_change_percentage": "24h,7d",
            "per_page": 1,
            "page": 1,
            "sparkline": "false",
            **selector,
        }
        async with session.get(self.markets_url, params=params) as response:
            if response.status == 429:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message="CoinGecko rate limited",
                    headers=response.headers,
                )
            response.raise_for_status()
            data = await response.json()
        return data[0] if isinstance(data, list) and data else {}

    @staticmethod
    def _to_ratio(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return round(float(value) / 100.0, 6)
        except (TypeError, ValueError):
            return None

