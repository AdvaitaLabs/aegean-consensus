"""Hit-rate backtesting for past investment analyses.

Given a set of past analyses (symbol, action, analysis_date) and a
price lookup function, the engine computes forward-return outcomes,
classifies each as hit/miss against the recommended action, and can
write outcomes back into :class:`RoleMemoryRegistry` so that BM25 recall
surfaces validated cases with higher-quality context next time.

Inspired by daily_stock_analysis' BacktestEngine (1-day window, hit/miss
tagging), generalized here to N-day horizons and role-aware feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional


# Forward-return thresholds (in decimal, e.g. 0.005 == +0.5%).
# BUY needs return >= +threshold to count as hit.
# SELL needs return <= -threshold.
# HOLD/WATCH needs |return| <= hold_band.
DEFAULT_ACTION_THRESHOLD = 0.005  # 0.5% move needed to confirm directional call
DEFAULT_HOLD_BAND = 0.01          # +/-1% band counts as a hit for HOLD/WATCH


@dataclass
class BacktestRecord:
    """A prior analysis to be evaluated."""

    analysis_id: str
    symbol: str
    analysis_date: date
    action: str  # "buy" | "sell" | "hold" | "watch"
    confidence: float = 0.0
    target_exposure_pct: float = 0.0
    role: str = ""
    group_id: Optional[str] = None
    situation_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestOutcome:
    analysis_id: str
    symbol: str
    action: str
    role: str
    group_id: Optional[str]
    horizon_days: int
    entry_price: Optional[float]
    exit_price: Optional[float]
    return_pct: Optional[float]
    hit: Optional[bool]
    reason: str = ""

    def to_memory_outcome(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "hit": self.hit,
            "return_pct": self.return_pct,
            "horizon_days": self.horizon_days,
            "evaluated_at": datetime.utcnow().isoformat(),
        }


@dataclass
class BacktestReport:
    total: int = 0
    evaluated: int = 0
    hits: int = 0
    misses: int = 0
    skipped: int = 0
    hit_rate: float = 0.0
    avg_return_pct: float = 0.0
    by_action: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outcomes: List[BacktestOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "evaluated": self.evaluated,
            "hits": self.hits,
            "misses": self.misses,
            "skipped": self.skipped,
            "hit_rate": self.hit_rate,
            "avg_return_pct": self.avg_return_pct,
            "by_action": self.by_action,
            "by_role": self.by_role,
        }


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    raise TypeError(f"Unsupported date type: {type(value)!r}")


def _classify_hit(
    action: str,
    return_pct: float,
    action_threshold: float,
    hold_band: float,
) -> bool:
    normalized = (action or "").lower()
    if normalized == "buy":
        return return_pct >= action_threshold
    if normalized == "sell":
        return return_pct <= -action_threshold
    if normalized in ("hold", "watch"):
        return abs(return_pct) <= hold_band
    return False


PriceLookup = Callable[[str, date], Optional[float]]


class BacktestEngine:
    """Evaluate prior analyses against actual forward returns.

    Args:
        horizon_days: forward window over which to measure returns.
        action_threshold: directional return (decimal) required for a BUY/SELL
            to count as a hit.
        hold_band: absolute return band within which HOLD/WATCH is a hit.
    """

    def __init__(
        self,
        horizon_days: int = 5,
        action_threshold: float = DEFAULT_ACTION_THRESHOLD,
        hold_band: float = DEFAULT_HOLD_BAND,
    ):
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        self.horizon_days = horizon_days
        self.action_threshold = action_threshold
        self.hold_band = hold_band

    def evaluate_record(
        self,
        record: BacktestRecord,
        price_at: PriceLookup,
    ) -> BacktestOutcome:
        analysis_date = _coerce_date(record.analysis_date)
        target_date = analysis_date + timedelta(days=self.horizon_days)
        entry = price_at(record.symbol, analysis_date)
        exit_ = price_at(record.symbol, target_date)
        if entry is None or entry <= 0 or exit_ is None or exit_ <= 0:
            return BacktestOutcome(
                analysis_id=record.analysis_id,
                symbol=record.symbol,
                action=record.action,
                role=record.role,
                group_id=record.group_id,
                horizon_days=self.horizon_days,
                entry_price=entry,
                exit_price=exit_,
                return_pct=None,
                hit=None,
                reason="missing_price_data",
            )
        return_pct = (exit_ - entry) / entry
        hit = _classify_hit(record.action, return_pct, self.action_threshold, self.hold_band)
        return BacktestOutcome(
            analysis_id=record.analysis_id,
            symbol=record.symbol,
            action=record.action,
            role=record.role,
            group_id=record.group_id,
            horizon_days=self.horizon_days,
            entry_price=entry,
            exit_price=exit_,
            return_pct=return_pct,
            hit=hit,
            reason="ok",
        )

    def evaluate(
        self,
        records: Iterable[BacktestRecord],
        price_at: PriceLookup,
    ) -> BacktestReport:
        outcomes: List[BacktestOutcome] = []
        for record in records:
            outcomes.append(self.evaluate_record(record, price_at))
        return self._summarize(outcomes)

    def apply_to_role_memory(
        self,
        outcomes: Iterable[BacktestOutcome],
        role_memory_registry: Any,
        records_by_analysis: Optional[Dict[str, BacktestRecord]] = None,
    ) -> int:
        """Persist hit/miss outcomes back into per-role BM25 memory.

        For each outcome with a non-null ``hit`` flag, append a new memory
        entry whose ``situation`` is taken from the originating record and
        whose ``outcome`` dict carries the evaluated metrics. Returns the
        number of entries written.

        We intentionally *append* rather than mutate an existing entry:
        BM25 has no stable entry identity, and new entries carrying the
        outcome metadata rank naturally alongside the original recall.
        """
        written = 0
        lookup = records_by_analysis or {}
        for outcome in outcomes:
            if outcome.hit is None:
                continue
            record = lookup.get(outcome.analysis_id)
            if record is None or not record.role or not record.situation_text:
                continue
            try:
                role_memory_registry.record(
                    role=record.role,
                    situation=record.situation_text,
                    recommendation=(
                        f"Retro-eval: action={record.action}, "
                        f"hit={outcome.hit}, return_pct={outcome.return_pct:.4f}, "
                        f"horizon_days={outcome.horizon_days}"
                    ),
                    group_id=record.group_id,
                    outcome=outcome.to_memory_outcome(),
                    metadata={
                        "source": "backtest",
                        "analysis_id": outcome.analysis_id,
                        "symbol": outcome.symbol,
                    },
                )
                written += 1
            except Exception:
                # Never let feedback writes break the pipeline.
                continue
        return written

    def _summarize(self, outcomes: List[BacktestOutcome]) -> BacktestReport:
        report = BacktestReport(total=len(outcomes), outcomes=outcomes)
        eval_returns: List[float] = []
        by_action_acc: Dict[str, Dict[str, float]] = {}
        by_role_acc: Dict[str, Dict[str, float]] = {}

        for o in outcomes:
            if o.hit is None or o.return_pct is None:
                report.skipped += 1
                continue
            report.evaluated += 1
            if o.hit:
                report.hits += 1
            else:
                report.misses += 1
            eval_returns.append(o.return_pct)

            action_key = (o.action or "unknown").lower()
            action_bucket = by_action_acc.setdefault(
                action_key, {"n": 0.0, "hits": 0.0, "return_sum": 0.0}
            )
            action_bucket["n"] += 1
            action_bucket["hits"] += 1 if o.hit else 0
            action_bucket["return_sum"] += o.return_pct

            role_key = o.role or "unknown"
            role_bucket = by_role_acc.setdefault(
                role_key, {"n": 0.0, "hits": 0.0, "return_sum": 0.0}
            )
            role_bucket["n"] += 1
            role_bucket["hits"] += 1 if o.hit else 0
            role_bucket["return_sum"] += o.return_pct

        if report.evaluated:
            report.hit_rate = report.hits / report.evaluated
            report.avg_return_pct = sum(eval_returns) / len(eval_returns)

        report.by_action = {
            k: {
                "n": int(v["n"]),
                "hit_rate": (v["hits"] / v["n"]) if v["n"] else 0.0,
                "avg_return_pct": (v["return_sum"] / v["n"]) if v["n"] else 0.0,
            }
            for k, v in by_action_acc.items()
        }
        report.by_role = {
            k: {
                "n": int(v["n"]),
                "hit_rate": (v["hits"] / v["n"]) if v["n"] else 0.0,
                "avg_return_pct": (v["return_sum"] / v["n"]) if v["n"] else 0.0,
            }
            for k, v in by_role_acc.items()
        }
        return report


def price_lookup_from_series(
    series_by_symbol: Dict[str, Dict[date, float]],
) -> PriceLookup:
    """Build a PriceLookup callable from a nested ``{symbol: {date: close}}`` dict.

    Falls back to the *closest prior date* within 7 calendar days — the same
    behavior you'd expect when the requested date falls on a weekend or
    holiday.
    """

    def _lookup(symbol: str, when: date) -> Optional[float]:
        book = series_by_symbol.get(symbol)
        if not book:
            return None
        if when in book:
            return book[when]
        for offset in range(1, 8):
            prior = when - timedelta(days=offset)
            if prior in book:
                return book[prior]
        return None

    return _lookup
