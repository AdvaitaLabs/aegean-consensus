"""Unit tests for the portfolio risk engine."""

from __future__ import annotations

import math
import random

import pytest

from aegean.investment.risk import PortfolioRiskEngine


def _low_vol_series(n: int = 80, start: float = 100.0, step: float = 0.05) -> list:
    return [start + i * step for i in range(n)]


def _high_vol_series(n: int = 80, start: float = 100.0, seed: int = 1) -> list:
    rng = random.Random(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(max(0.01, prices[-1] * (1 + rng.uniform(-0.06, 0.06))))
    return prices


def test_volatility_adjusted_limit_monotonic():
    eng = PortfolioRiskEngine()
    low = eng.volatility_adjusted_limit(0.10)
    mid = eng.volatility_adjusted_limit(0.25)
    high = eng.volatility_adjusted_limit(0.45)
    extreme = eng.volatility_adjusted_limit(0.80)
    assert low > mid > high
    assert extreme <= high


def test_correlation_multiplier_thresholds():
    eng = PortfolioRiskEngine()
    assert eng.correlation_multiplier(None) == 1.0
    assert eng.correlation_multiplier(0.9) == 0.70
    assert eng.correlation_multiplier(0.65) == 0.85
    assert eng.correlation_multiplier(0.5) == 1.00
    assert eng.correlation_multiplier(0.3) == 1.05
    assert eng.correlation_multiplier(0.1) == 1.10


def test_assess_returns_binding_cap_for_high_vol():
    eng = PortfolioRiskEngine()
    prices = {"AAPL": _high_vol_series(seed=2)}
    result = eng.assess("AAPL", prices, portfolio={"cash": 100_000})
    assert result.volatility.annualized_volatility > 0.30
    # High-vol should be capped at <= 20% base
    assert result.base_limit_pct <= 0.20
    assert result.combined_limit_pct > 0
    assert result.position_limit_value == pytest.approx(
        result.portfolio_value * result.combined_limit_pct
    )


def test_assess_handles_missing_price_series():
    eng = PortfolioRiskEngine()
    result = eng.assess("AAPL", {"AAPL": []}, portfolio=None)
    assert result.volatility.fallback is True
    assert result.correlation.avg_correlation is None
    assert result.combined_limit_pct >= 0.0


def test_correlation_reduces_limit_for_correlated_active_position():
    eng = PortfolioRiskEngine()
    series = _low_vol_series()
    # Same series -> correlation 1.0 with existing active position MSFT
    prices = {"AAPL": series, "MSFT": list(series)}
    portfolio = {
        "cash": 100_000,
        "positions": {"MSFT": {"long": 10, "short": 0}},
    }
    result = eng.assess("AAPL", prices, portfolio=portfolio)
    assert result.correlation.avg_correlation is not None
    assert result.correlation.avg_correlation > 0.9
    assert result.correlation_multiplier <= 0.70
    assert result.combined_limit_pct < result.base_limit_pct


def test_assess_empty_portfolio_uses_all_symbols_for_correlation():
    eng = PortfolioRiskEngine()
    series = _high_vol_series(seed=3)
    # Inverted returns: whenever `series` moves up, `inverted` moves down.
    start = series[0]
    inverted = [start]
    for prev_s, s in zip(series, series[1:]):
        ret = (s - prev_s) / prev_s if prev_s else 0
        inverted.append(max(0.01, inverted[-1] * (1 - ret)))
    prices = {"AAPL": series, "XYZ": inverted}
    result = eng.assess("AAPL", prices, portfolio={"cash": 50_000})
    assert result.correlation.avg_correlation is not None
    assert result.correlation.avg_correlation < 0
    assert result.correlation_multiplier == pytest.approx(1.10)
