"""Unit tests for TushareProvider.fetch_insider_transactions + adapter."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from aegean.investment.providers.tushare_provider import TushareProvider
from aegean.investment.sentiment import tushare_insider_to_trades


class _FakeResponse:
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status >= 400 and self.status != 429:
            raise RuntimeError(f"HTTP {self.status}")


class _FakeSession:
    def __init__(self, response: _FakeResponse | None = None, raise_exc: Exception | None = None) -> None:
        self._response = response
        self._raise = raise_exc
        self.last_payload: Dict[str, Any] | None = None

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: Dict[str, Any]):
        self.last_payload = json
        if self._raise is not None:
            raise self._raise
        return self._response


def _install_fake_session(monkeypatch, session: _FakeSession) -> None:
    import aegean.investment.providers.tushare_provider as mod

    class _Factory:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeSession:
            return await session.__aenter__()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            await session.__aexit__(exc_type, exc, tb)

    monkeypatch.setattr(mod.aiohttp, "ClientSession", _Factory)


def test_missing_key_returns_unavailable():
    provider = TushareProvider(api_key=None)
    result = asyncio.run(provider.fetch_insider_transactions("600519"))
    assert result["status"] == "unavailable"
    assert "TUSHARE_INSIDER_UNAVAILABLE" in result["signals"]


def test_ok_payload_parses_fields_items(monkeypatch):
    payload = {
        "data": {
            "fields": ["ts_code", "ann_date", "in_de", "change_vol", "holder_name"],
            "items": [
                ["600519.SH", "20260101", "IN", 10000, "董事长"],
                ["600519.SH", "20260102", "DE", 5000, "高管"],
            ],
        },
        "code": 0,
    }
    session = _FakeSession(response=_FakeResponse(200, payload))
    _install_fake_session(monkeypatch, session)
    provider = TushareProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("600519"))
    assert result["status"] == "ok"
    assert len(result["data"]) == 2
    assert result["data"][0]["in_de"] == "IN"
    assert session.last_payload["api_name"] == "stk_holdertrade"


def test_rate_limited(monkeypatch):
    session = _FakeSession(response=_FakeResponse(429, {}))
    _install_fake_session(monkeypatch, session)
    provider = TushareProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("600519"))
    assert result["status"] == "rate_limited"


def test_timeout(monkeypatch):
    session = _FakeSession(raise_exc=asyncio.TimeoutError())
    _install_fake_session(monkeypatch, session)
    provider = TushareProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("600519"))
    assert result["status"] == "timeout"
    assert "TUSHARE_INSIDER_TIMEOUT" in result["signals"]


def test_tushare_adapter_uses_in_de_sign():
    rows = [
        {"in_de": "IN", "change_vol": 10000, "holder_name": "a", "ann_date": "20260101"},
        {"in_de": "DE", "change_vol": 5000, "holder_name": "b"},
        {"in_de": "DE", "change_vol": -5000, "holder_name": "c"},
        {"in_de": "", "change_vol": -200, "holder_name": "d"},
    ]
    trades = tushare_insider_to_trades(rows)
    assert [t.transaction_shares for t in trades] == [10000.0, -5000.0, -5000.0, -200.0]
    assert trades[0].insider_role == "a"
    assert trades[0].filed_at == "20260101"


def test_tushare_adapter_accepts_fields_items_shape():
    payload = {
        "fields": ["in_de", "change_vol", "holder_name"],
        "items": [["IN", 100, "x"], ["DE", 50, "y"]],
    }
    trades = tushare_insider_to_trades(payload)
    assert [t.transaction_shares for t in trades] == [100.0, -50.0]


def test_tushare_adapter_rejects_bad_inputs():
    assert tushare_insider_to_trades(None) == []
    assert tushare_insider_to_trades("oops") == []
    assert tushare_insider_to_trades({"data": None}) == []
