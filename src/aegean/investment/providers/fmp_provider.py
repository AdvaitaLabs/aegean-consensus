from __future__ import annotations

from typing import Any, Dict

import asyncio
import aiohttp

from .base import ExternalDataProvider


class FMPProvider(ExternalDataProvider):
    provider_name = "fmp"
    quote_url = "https://financialmodelingprep.com/api/v3/quote/{symbol}"
    ratios_url = "https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}"
    metrics_url = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
    income_url = "https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
    balance_url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}"
    cashflow_url = "https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
    target_price_url = "https://financialmodelingprep.com/api/v4/price-target"

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 4.0):
        super().__init__(api_key=api_key or self._env("FMP_API_KEY"), timeout_seconds=timeout_seconds)

    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        result = self._base_result(symbol=symbol, market=market, asset_type=asset_type)
        if not self.api_key:
            result["metadata"] = {
                "status": "unavailable",
                "message": "Missing FMP_API_KEY",
                "signals": ["FMP_UNAVAILABLE"],
                "timeout_seconds": self.timeout_seconds,
            }
            return result

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        params = {"apikey": self.api_key}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                quote_req = session.get(self.quote_url.format(symbol=symbol), params=params)
                ratios_req = session.get(self.ratios_url.format(symbol=symbol), params=params)
                metrics_req = session.get(self.metrics_url.format(symbol=symbol), params=params)
                income_req = session.get(self.income_url.format(symbol=symbol), params={**params, "limit": 1})
                balance_req = session.get(self.balance_url.format(symbol=symbol), params={**params, "limit": 1})
                cashflow_req = session.get(self.cashflow_url.format(symbol=symbol), params={**params, "limit": 1})
                target_req = session.get(self.target_price_url, params={"symbol": symbol, **params})
                quote_resp, ratios_resp, metrics_resp, income_resp, balance_resp, cashflow_resp, target_resp = await asyncio.gather(
                    quote_req, ratios_req, metrics_req, income_req, balance_req, cashflow_req, target_req
                )
                responses = [quote_resp, ratios_resp, metrics_resp, income_resp, balance_resp, cashflow_resp, target_resp]
                if any(resp.status == 429 for resp in responses):
                    return self._rate_limited_result(symbol, market, asset_type)
                async with quote_resp, ratios_resp, metrics_resp, income_resp, balance_resp, cashflow_resp, target_resp:
                    quote_data, ratios_data, metrics_data, income_data, balance_data, cashflow_data, target_data = await asyncio.gather(
                        quote_resp.json(),
                        ratios_resp.json(),
                        metrics_resp.json(),
                        income_resp.json(),
                        balance_resp.json(),
                        cashflow_resp.json(),
                        target_resp.json(),
                    )

            quote = quote_data[0] if isinstance(quote_data, list) and quote_data else {}
            ratios = ratios_data[0] if isinstance(ratios_data, list) and ratios_data else {}
            metrics = metrics_data[0] if isinstance(metrics_data, list) and metrics_data else {}
            income = income_data[0] if isinstance(income_data, list) and income_data else {}
            balance = balance_data[0] if isinstance(balance_data, list) and balance_data else {}
            cashflow = cashflow_data[0] if isinstance(cashflow_data, list) and cashflow_data else {}
            targets = target_data if isinstance(target_data, list) else []
            target_price = None
            if targets:
                values = [item.get("priceTarget") for item in targets if item.get("priceTarget") is not None]
                if values:
                    target_price = round(sum(values) / len(values), 4)
            result["market_data"].update(
                {
                    "price": quote.get("price"),
                    "change_pct": quote.get("changesPercentage"),
                    "market_cap": quote.get("marketCap"),
                    "avg_volume": quote.get("avgVolume"),
                }
            )
            result["fundamentals"].update(
                {
                    "pe_ttm": quote.get("pe"),
                    "pb": metrics.get("pbRatioTTM") or ratios.get("priceToBookRatioTTM"),
                    "roe": metrics.get("roeTTM"),
                    "debt_to_equity": metrics.get("debtToEquity") or ratios.get("debtEquityRatioTTM"),
                    "gross_margin": ratios.get("grossProfitMarginTTM"),
                    "operating_margin": ratios.get("operatingProfitMarginTTM"),
                    "revenue": income.get("revenue"),
                    "net_income": income.get("netIncome"),
                    "free_cash_flow": cashflow.get("freeCashFlow"),
                    "cash_and_short_term_investments": balance.get("cashAndShortTermInvestments"),
                    "total_debt": balance.get("totalDebt"),
                    "analyst_target_price": target_price,
                }
            )
            result["metadata"]["status"] = "ok"
            return result
        except asyncio.TimeoutError:
            return self._timeout_result(symbol, market, asset_type)
        except Exception as exc:
            return self._error_result(symbol, market, asset_type, str(exc))

