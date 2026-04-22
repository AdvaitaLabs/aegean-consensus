"""Unit tests for the investment BacktestEngine."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pytest

from aegean.investment.backtest import (
    BacktestEngine,
    BacktestOutcome,
    BacktestRecord,
)
from aegean.investment.backtest.engine import (
    _classify_hit,
    price_lookup_from_series,
)


def _record(
    analysis_id: str,
    symbol: str,
    action: str,
    analysis_date: date,
    role: str = "fundamental_specialist",
    group_id: Optional[str] = "g1",
    situation_text: str = "situation summary",
) -> BacktestRecord:
    return BacktestRecord(
        analysis_id=analysis_id,
        symbol=symbol,
        analysis_date=analysis_date,
        action=action,
        role=role,
        group_id=group_id,
        situation_text=situation_text,
    )


def test_classify_hit_rules():
    assert _classify_hit("buy", 0.02, 0.005, 0.01) is True
    assert _classify_hit("buy", 0.002, 0.005, 0.01) is False
    assert _classify_hit("sell", -0.02, 0.005, 0.01) is True
    assert _classify_hit("sell", -0.001, 0.005, 0.01) is False
    assert _classify_hit("hold", 0.005, 0.005, 0.01) is True
    assert _classify_hit("hold", 0.02, 0.005, 0.01) is False
    assert _classify_hit("watch", -0.008, 0.005, 0.01) is True
    assert _classify_hit("garbage", 0.5, 0.005, 0.01) is False


def test_evaluate_record_hit_and_skip():
    d0 = date(2026, 1, 5)
    series = {
        "AAPL": {d0: 100.0, d0 + timedelta(days=5): 103.0},
        "TSLA": {d0: 200.0},  # missing exit
    }
    price_at = price_lookup_from_series(series)
    engine = BacktestEngine(horizon_days=5, action_threshold=0.005, hold_band=0.01)

    hit_outcome = engine.evaluate_record(_record("a1", "AAPL", "buy", d0), price_at)
    assert hit_outcome.hit is True
    assert hit_outcome.return_pct == pytest.approx(0.03)
    assert hit_outcome.reason == "ok"

    skip_outcome = engine.evaluate_record(_record("a2", "TSLA", "buy", d0), price_at)
    assert skip_outcome.hit is None
    assert skip_outcome.return_pct is None
    assert skip_outcome.reason == "missing_price_data"


def test_evaluate_aggregates_by_action_and_role():
    d0 = date(2026, 1, 5)
    d5 = d0 + timedelta(days=5)
    series = {
        "AAPL": {d0: 100.0, d5: 103.0},   # buy hit
        "MSFT": {d0: 100.0, d5: 100.2},   # buy miss
        "NVDA": {d0: 100.0, d5: 97.0},    # sell hit
        "GOOG": {d0: 100.0, d5: 100.5},   # hold hit
    }
    engine = BacktestEngine(horizon_days=5)
    records = [
        _record("a1", "AAPL", "buy", d0, role="fundamental_specialist"),
        _record("a2", "MSFT", "buy", d0, role="fundamental_specialist"),
        _record("a3", "NVDA", "sell", d0, role="risk_specialist"),
        _record("a4", "GOOG", "hold", d0, role="portfolio_strategist"),
    ]
    report = engine.evaluate(records, price_lookup_from_series(series))

    assert report.total == 4
    assert report.evaluated == 4
    assert report.skipped == 0
    assert report.hits == 3
    assert report.misses == 1
    assert report.hit_rate == pytest.approx(0.75)
    assert report.by_action["buy"]["n"] == 2
    assert report.by_action["buy"]["hit_rate"] == pytest.approx(0.5)
    assert report.by_role["fundamental_specialist"]["n"] == 2
    assert report.by_role["risk_specialist"]["hit_rate"] == pytest.approx(1.0)


def test_apply_to_role_memory_writes_per_hit():
    class _StubRegistry:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def record(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    d0 = date(2026, 1, 5)
    engine = BacktestEngine(horizon_days=3)
    outcomes = [
        BacktestOutcome(
            analysis_id="a1", symbol="AAPL", action="buy", role="fundamental_specialist",
            group_id="g1", horizon_days=3, entry_price=100.0, exit_price=103.0,
            return_pct=0.03, hit=True, reason="ok",
        ),
        BacktestOutcome(
            analysis_id="a2", symbol="TSLA", action="buy", role="fundamental_specialist",
            group_id="g1", horizon_days=3, entry_price=None, exit_price=None,
            return_pct=None, hit=None, reason="missing_price_data",
        ),
    ]
    records_by_id = {
        "a1": _record("a1", "AAPL", "buy", d0),
        "a2": _record("a2", "TSLA", "buy", d0),
    }
    stub = _StubRegistry()
    written = engine.apply_to_role_memory(outcomes, stub, records_by_analysis=records_by_id)

    assert written == 1
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["role"] == "fundamental_specialist"
    assert call["group_id"] == "g1"
    assert call["outcome"]["hit"] is True
    assert call["metadata"]["source"] == "backtest"
    assert call["metadata"]["analysis_id"] == "a1"


def test_price_lookup_weekend_fallback():
    d_fri = date(2026, 1, 2)  # Friday
    d_mon = date(2026, 1, 5)  # Monday
    series = {"AAPL": {d_fri: 100.0, d_mon: 101.0}}
    lookup = price_lookup_from_series(series)
    # Requesting Saturday should fall back to Friday.
    assert lookup("AAPL", date(2026, 1, 3)) == 100.0
    # Requesting Monday directly resolves exactly.
    assert lookup("AAPL", d_mon) == 101.0
    # Missing symbol returns None.
    assert lookup("ZZZ", d_mon) is None
