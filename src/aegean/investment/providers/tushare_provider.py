from __future__ import annotations

import asyncio
from typing import Any, Dict

import aiohttp

from .base import ExternalDataProvider


class TushareProvider(ExternalDataProvider):
    provider_name = "tushare"
    pro_url = "https://api.tushare.pro"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 4.0):
        super().__init__(api_key=api_key or self._env("TUSHARE_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if market != "CN":
            result["metadata"] = {
                "status": "unavailable",
                "message": "Tushare is only enabled for CN market",
                "signals": ["TUSHARE_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing TUSHARE_API_KEY",
                "signals": ["TUSHARE_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        ts_code = self._normalize_symbol(symbol)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                basic_task = self._post(
                    session,
                    api_name="daily_basic",
                    ts_code=ts_code,
                    fields="ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
                )
                daily_task = self._post(
                    session,
                    api_name="daily",
                    ts_code=ts_code,
                    fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
                )
                income_task = self._post(
                    session,
                    api_name="income",
                    ts_code=ts_code,
                    fields="ts_code,ann_date,end_date,revenue,n_income,total_profit,basic_eps",
                )
                balancesheet_task = self._post(
                    session,
                    api_name="balancesheet",
                    ts_code=ts_code,
                    fields="ts_code,ann_date,end_date,total_assets,total_liab,money_cap,total_hldr_eqy_exc_min_int",
                )
                cashflow_task = self._post(
                    session,
                    api_name="cashflow",
                    ts_code=ts_code,
                    fields="ts_code,ann_date,end_date,n_cashflow_act,free_cashflow",
                )
                basic_data, daily_data, income_data, balance_data, cashflow_data = await asyncio.gather(
                    basic_task,
                    daily_task,
                    income_task,
                    balancesheet_task,
                    cashflow_task,
                )

            basic = self._first_row(basic_data)
            daily = self._first_row(daily_data)
            income = self._first_row(income_data)
            balance = self._first_row(balance_data)
            cashflow = self._first_row(cashflow_data)
            price = self._to_float(basic.get("close"))
            if price is None:
                price = self._to_float(daily.get("close"))
            change_pct = self._to_ratio(daily.get("pct_chg"))
            if change_pct is None:
                price_change = self._to_float(daily.get("change"))
                pre_close = self._to_float(daily.get("pre_close"))
                if price_change is not None and pre_close not in (None, 0):
                    change_pct = round(price_change / pre_close, 6)

            result["market_data"].update(
                {
                    "symbol": ts_code,
                    "price": price,
                    "change_pct": change_pct,
                    "open": self._to_float(daily.get("open")),
                    "high": self._to_float(daily.get("high")),
                    "low": self._to_float(daily.get("low")),
                    "previous_close": self._to_float(daily.get("pre_close")),
                    "volume": self._scaled_volume(daily.get("vol")),
                    "turnover_rate": self._to_float(basic.get("turnover_rate")),
                    "volume_ratio": self._to_float(basic.get("volume_ratio")),
                    "market_cap": self._scaled_amount(basic.get("total_mv")),
                    "float_market_cap": self._scaled_amount(basic.get("circ_mv")),
                    "trade_date": basic.get("trade_date") or daily.get("trade_date"),
                }
            )
            result["fundamentals"].update(
                {
                    "pe_ttm": self._to_float(basic.get("pe")),
                    "pb": self._to_float(basic.get("pb")),
                    "revenue": self._to_float(income.get("revenue")),
                    "net_income": self._to_float(income.get("n_income")),
                    "total_profit": self._to_float(income.get("total_profit")),
                    "eps": self._to_float(income.get("basic_eps")),
                    "total_assets": self._to_float(balance.get("total_assets")),
                    "total_liabilities": self._to_float(balance.get("total_liab")),
                    "cash": self._to_float(balance.get("money_cap")),
                    "shareholders_equity": self._to_float(balance.get("total_hldr_eqy_exc_min_int")),
                    "operating_cash_flow": self._to_float(cashflow.get("n_cashflow_act")),
                    "free_cash_flow": self._to_float(cashflow.get("free_cashflow")),
                }
            )
            result["metadata"]["status"] = "ok"
            return result
        except asyncio.TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except aiohttp.ClientResponseError as exc:
            if exc.status == 429:
                return self._rate_limited_result(symbol, market, asset_type)
            return self._error_result(symbol, market, asset_type, str(exc))
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

    async def _post(self, session: aiohttp.ClientSession, *, api_name: str, ts_code: str, fields: str) -> Dict[str, Any]:
        payload = {
            "api_name": api_name,
            "token": self.api_key,
            "params": {"ts_code": ts_code, "limit": 1},
            "fields": fields,
        }
        async with session.post(self.pro_url, json=payload) as response:
            if response.status == 429:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message="Tushare rate limited",
                    headers=response.headers,
                )
            response.raise_for_status()
            return await response.json()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if "." in normalized:
            return normalized
        if normalized.startswith(("60", "68", "90")):
            return f"{normalized}.SH"
        return f"{normalized}.SZ"

    @staticmethod
    def _first_row(payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data") if isinstance(payload, dict) else {}
        fields = data.get("fields") if isinstance(data, dict) else []
        items = data.get("items") if isinstance(data, dict) else []
        if not fields or not items:
            return {}
        first = items[0]
        return dict(zip(fields, first)) if isinstance(first, list) else {}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_ratio(value: Any) -> float | None:
        numeric = TushareProvider._to_float(value)
        return round(numeric / 100.0, 6) if numeric is not None else None

    @staticmethod
    def _scaled_volume(value: Any) -> float | None:
        volume = TushareProvider._to_float(value)
        return volume * 100 if volume is not None else None

    @classmethod
    def _scaled_amount(cls, value: Any) -> float | None:
        amount = cls._to_float(value)
        return amount * 10000 if amount is not None else None

