"""Unit tests for FinnhubProvider.fetch_insider_transactions.

Network calls are stubbed with a light aiohttp monkeypatch so tests run
offline. We only exercise the shape of the wrapper: response parsing,
missing API key short-circuit, rate limits, timeouts.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

import pytest

from aegean.investment.providers.finnhub_provider import FinnhubProvider


class _FakeResponse:
    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    async def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse | None = None, raise_exc: Exception | None = None) -> None:
        self._response = response
        self._raise = raise_exc

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    def get(self, url: str, params: Dict[str, Any]):
        if self._raise is not None:
            raise self._raise
        return self._response


def _install_fake_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    import aegean.investment.providers.finnhub_provider as mod

    class _Factory:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeSession:
            return await session.__aenter__()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            await session.__aexit__(exc_type, exc, tb)

    monkeypatch.setattr(mod.aiohttp, "ClientSession", _Factory)


def test_missing_api_key_returns_unavailable():
    provider = FinnhubProvider(api_key=None)
    result = asyncio.run(provider.fetch_insider_transactions("AAPL"))
    assert result["status"] == "unavailable"
    assert result["signals"] == ["FINNHUB_INSIDER_UNAVAILABLE"]
    assert result["data"] == []


def test_ok_payload_returns_data(monkeypatch):
    fake_rows = [{"name": "CEO", "change": 100}]
    session = _FakeSession(response=_FakeResponse(200, {"data": fake_rows, "symbol": "AAPL"}))
    _install_fake_session(monkeypatch, session)
    provider = FinnhubProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("AAPL"))
    assert result["status"] == "ok"
    assert result["data"] == fake_rows


def test_rate_limited(monkeypatch):
    session = _FakeSession(response=_FakeResponse(429, {}))
    _install_fake_session(monkeypatch, session)
    provider = FinnhubProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("AAPL"))
    assert result["status"] == "rate_limited"
    assert "FINNHUB_INSIDER_RATE_LIMITED" in result["signals"]


def test_timeout(monkeypatch):
    session = _FakeSession(raise_exc=asyncio.TimeoutError())
    _install_fake_session(monkeypatch, session)
    provider = FinnhubProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("AAPL"))
    assert result["status"] == "timeout"
    assert "FINNHUB_INSIDER_TIMEOUT" in result["signals"]


def test_generic_error(monkeypatch):
    session = _FakeSession(raise_exc=RuntimeError("boom"))
    _install_fake_session(monkeypatch, session)
    provider = FinnhubProvider(api_key="token")
    result = asyncio.run(provider.fetch_insider_transactions("AAPL"))
    assert result["status"] == "error"
    assert "FINNHUB_INSIDER_FAILED" in result["signals"]
    assert "boom" in result["message"]
