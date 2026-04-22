"""Portfolio-level risk adjustments: volatility and correlation sizing.

Ported from ai-hedge-fund's risk_manager.py, re-implemented without
pandas/numpy hard dependency so it degrades gracefully when those
libraries are absent. When ``numpy`` is importable we use it for the
heavy arithmetic; otherwise we fall back to pure-Python stdlib math.

The engine is deliberately pure: it consumes pre-fetched price series
and portfolio snapshots, and returns a deterministic
:class:`PortfolioRiskResult`. It does **not** call any providers — wiring
it into the investment pipeline is the caller's responsibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TRADING_DAYS_PER_YEAR = 252


@dataclass
class VolatilityMetrics:
    daily_volatility: float
    annualized_volatility: float
    data_points: int
    fallback: bool = False


@dataclass
class CorrelationMetrics:
    avg_correlation: Optional[float] = None
    max_correlation: Optional[float] = None
    top_correlated: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "top_correlated": [
                {"ticker": t, "correlation": c} for t, c in self.top_correlated
            ],
        }


@dataclass
class PortfolioRiskResult:
    symbol: str
    volatility: VolatilityMetrics
    correlation: CorrelationMetrics
    base_limit_pct: float
    correlation_multiplier: float
    combined_limit_pct: float
    portfolio_value: float
    position_limit_value: float
    remaining_limit_value: float
    current_position_value: float
    reasoning: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "volatility": {
                "daily_volatility": self.volatility.daily_volatility,
                "annualized_volatility": self.volatility.annualized_volatility,
                "data_points": self.volatility.data_points,
                "fallback": self.volatility.fallback,
            },
            "correlation": self.correlation.to_dict(),
            "base_limit_pct": self.base_limit_pct,
            "correlation_multiplier": self.correlation_multiplier,
            "combined_limit_pct": self.combined_limit_pct,
            "portfolio_value": self.portfolio_value,
            "position_limit_value": self.position_limit_value,
            "remaining_limit_value": self.remaining_limit_value,
            "current_position_value": self.current_position_value,
            "reasoning": self.reasoning,
        }


def _prices_to_returns(prices: Sequence[float]) -> List[float]:
    returns: List[float] = []
    prev: Optional[float] = None
    for p in prices:
        if p is None or p <= 0:
            prev = None
            continue
        if prev is not None and prev > 0:
            returns.append((p - prev) / prev)
        prev = p
    return returns


def _stddev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(variance, 0.0))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return num / (denom_x * denom_y)


def _align_returns(
    series_map: Mapping[str, Sequence[float]]
) -> Dict[str, List[float]]:
    if not series_map:
        return {}
    length = min(len(s) for s in series_map.values())
    if length < 2:
        return {}
    return {k: list(v[-length:]) for k, v in series_map.items()}


class PortfolioRiskEngine:
    """Computes volatility- and correlation-adjusted position limits.

    Args:
        base_limit_pct: baseline exposure cap (fraction of portfolio value).
            Defaults to 0.20 (20%).
        volatility_lookback_days: max number of recent returns used for the
            volatility estimate. Defaults to 60.
    """

    def __init__(
        self,
        base_limit_pct: float = 0.20,
        volatility_lookback_days: int = 60,
    ):
        if not 0.01 <= base_limit_pct <= 1.0:
            raise ValueError("base_limit_pct must be in [0.01, 1.0]")
        if volatility_lookback_days < 5:
            raise ValueError("volatility_lookback_days must be >= 5")
        self.base_limit_pct = base_limit_pct
        self.volatility_lookback_days = volatility_lookback_days

    def compute_volatility(self, prices: Sequence[float]) -> VolatilityMetrics:
        returns = _prices_to_returns(prices)
        if len(returns) < 2:
            return VolatilityMetrics(
                daily_volatility=0.025,
                annualized_volatility=0.025 * math.sqrt(TRADING_DAYS_PER_YEAR),
                data_points=len(returns),
                fallback=True,
            )
        recent = returns[-self.volatility_lookback_days:]
        daily = _stddev(recent)
        if daily == 0.0:
            return VolatilityMetrics(
                daily_volatility=0.025,
                annualized_volatility=0.025 * math.sqrt(TRADING_DAYS_PER_YEAR),
                data_points=len(recent),
                fallback=True,
            )
        return VolatilityMetrics(
            daily_volatility=daily,
            annualized_volatility=daily * math.sqrt(TRADING_DAYS_PER_YEAR),
            data_points=len(recent),
            fallback=False,
        )

    @staticmethod
    def volatility_adjusted_limit(
        annualized_volatility: float,
        base_limit_pct: float = 0.20,
    ) -> float:
        """Map annualized volatility to a base-limit multiplier.

        - < 15% vol: 1.25x  (low vol, allow bigger size)
        - 15–30% vol: linear taper 1.0 -> 0.5
        - 30–50% vol: linear taper 0.75 -> 0.25
        - >= 50% vol: 0.5x floor
        Multiplier clamped to [0.25, 1.25].
        """
        if annualized_volatility < 0.15:
            multiplier = 1.25
        elif annualized_volatility < 0.30:
            multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5
        elif annualized_volatility < 0.50:
            multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5
        else:
            multiplier = 0.50
        multiplier = max(0.25, min(1.25, multiplier))
        return base_limit_pct * multiplier

    @staticmethod
    def correlation_multiplier(avg_correlation: Optional[float]) -> float:
        """Shrink exposure when a new position is highly correlated with existing book."""
        if avg_correlation is None:
            return 1.0
        if avg_correlation >= 0.80:
            return 0.70
        if avg_correlation >= 0.60:
            return 0.85
        if avg_correlation >= 0.40:
            return 1.00
        if avg_correlation >= 0.20:
            return 1.05
        return 1.10

    def compute_correlation(
        self,
        target_symbol: str,
        price_series_by_symbol: Mapping[str, Sequence[float]],
        active_symbols: Optional[Iterable[str]] = None,
    ) -> CorrelationMetrics:
        if target_symbol not in price_series_by_symbol:
            return CorrelationMetrics()
        returns_map = {
            sym: _prices_to_returns(prices)
            for sym, prices in price_series_by_symbol.items()
            if prices and len(prices) >= 2
        }
        aligned = _align_returns(returns_map)
        target_returns = aligned.get(target_symbol)
        if not target_returns:
            return CorrelationMetrics()

        candidates = set(aligned.keys()) - {target_symbol}
        if active_symbols:
            active_set = {s for s in active_symbols if s in aligned and s != target_symbol}
            if active_set:
                candidates = active_set

        corrs: List[Tuple[str, float]] = []
        for sym in candidates:
            c = _pearson(target_returns, aligned[sym])
            if c is not None:
                corrs.append((sym, c))
        if not corrs:
            return CorrelationMetrics()
        values = [c for _, c in corrs]
        avg = sum(values) / len(values)
        mx = max(values)
        top3 = sorted(corrs, key=lambda item: item[1], reverse=True)[:3]
        return CorrelationMetrics(
            avg_correlation=avg,
            max_correlation=mx,
            top_correlated=[(s, round(c, 4)) for s, c in top3],
        )

    def assess(
        self,
        target_symbol: str,
        price_series_by_symbol: Mapping[str, Sequence[float]],
        portfolio: Optional[Mapping[str, object]] = None,
    ) -> PortfolioRiskResult:
        """Run the full volatility+correlation workflow for ``target_symbol``.

        Args:
            target_symbol: symbol being sized.
            price_series_by_symbol: historical close prices keyed by symbol.
                Must include ``target_symbol``. Other symbols are optional but
                enable the correlation adjustment. Series should be ordered
                oldest -> newest.
            portfolio: optional snapshot. Expected shape::

                {
                    "cash": 10000.0,
                    "positions": {
                        "AAPL": {"long": 10, "short": 0},
                        ...
                    }
                }

        Returns:
            :class:`PortfolioRiskResult` describing the adjusted limit.
        """
        target_prices = price_series_by_symbol.get(target_symbol) or []
        volatility = self.compute_volatility(target_prices)
        base_limit = self.volatility_adjusted_limit(
            volatility.annualized_volatility, self.base_limit_pct
        )

        portfolio = portfolio or {}
        cash = float(portfolio.get("cash", 0.0) or 0.0)
        positions = portfolio.get("positions") or {}
        current_prices: Dict[str, float] = {}
        for sym, prices in price_series_by_symbol.items():
            if prices:
                current_prices[sym] = float(prices[-1])

        portfolio_value = cash
        current_position_value = 0.0
        active_symbols: List[str] = []
        if isinstance(positions, Mapping):
            for sym, pos in positions.items():
                if not isinstance(pos, Mapping):
                    continue
                long_qty = float(pos.get("long", 0) or 0)
                short_qty = float(pos.get("short", 0) or 0)
                price = current_prices.get(sym, 0.0)
                portfolio_value += (long_qty - short_qty) * price
                if abs(long_qty - short_qty) > 0:
                    active_symbols.append(sym)
                if sym == target_symbol:
                    current_position_value = abs(long_qty - short_qty) * price

        correlation = self.compute_correlation(
            target_symbol, price_series_by_symbol, active_symbols or None
        )
        corr_mult = self.correlation_multiplier(correlation.avg_correlation)
        combined_pct = max(0.0, base_limit * corr_mult)
        position_limit_value = max(0.0, portfolio_value * combined_pct)
        remaining_limit_value = max(0.0, position_limit_value - current_position_value)

        reasoning = (
            f"ann_vol={volatility.annualized_volatility:.2%} -> base={base_limit:.2%}; "
            f"avg_corr={correlation.avg_correlation if correlation.avg_correlation is not None else 'n/a'} "
            f"-> mult={corr_mult:.2f}; combined={combined_pct:.2%}"
        )

        return PortfolioRiskResult(
            symbol=target_symbol,
            volatility=volatility,
            correlation=correlation,
            base_limit_pct=base_limit,
            correlation_multiplier=corr_mult,
            combined_limit_pct=combined_pct,
            portfolio_value=portfolio_value,
            position_limit_value=position_limit_value,
            remaining_limit_value=remaining_limit_value,
            current_position_value=current_position_value,
            reasoning=reasoning,
        )
