from __future__ import annotations

import asyncio
from typing import Any, Dict

import yfinance as yf

from .base import ExternalDataProvider


class YFinanceProvider(ExternalDataProvider):
    provider_name = "yfinance"

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._fetch_sync, symbol, market, asset_type),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

    def _fetch_sync(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        history = ticker.history(period="5d")
        latest_close = None
        previous_close = info.get("previousClose")
        change_pct = None
        volume = None
        if not history.empty:
            latest_close = float(history["Close"].iloc[-1])
            volume = float(history["Volume"].iloc[-1]) if "Volume" in history else None
        if latest_close is not None and previous_close not in (None, 0):
            change_pct = round((latest_close - float(previous_close)) / float(previous_close), 6)

        result["market_data"].update(
            {
                "price": latest_close,
                "change_pct": change_pct,
                "volume": volume,
                "market_cap": info.get("marketCap"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        )
        result["fundamentals"].update(
            {
                "market_cap": info.get("marketCap"),
                "pe_ttm": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
            }
        )
        result["metadata"]["status"] = "ok"
        return result

