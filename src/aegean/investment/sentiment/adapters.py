"""Adapters that bridge external providers and :mod:`aegean.investment.sentiment`.

Providers return payloads in their own shapes (finnhub news dicts with a
``polarity`` field, finnhub insider transactions with ``change`` / ``share``
fields, generic normalized news with ``polarity``). The pipeline, on the
other hand, consumes homogeneous :class:`InsiderTrade` / :class:`NewsArticle`
objects. Keeping the conversion here means the pipeline stays free of
provider-specific knowledge and we can swap providers without touching
the scoring logic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from aegean.investment.sentiment.pipeline import InsiderTrade, NewsArticle


_NEWS_POLARITY_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "bullish": "positive",
    "+": "positive",
    "negative": "negative",
    "neg": "negative",
    "bearish": "negative",
    "-": "negative",
    "neutral": "neutral",
    "mixed": "neutral",
    "": "neutral",
}


def _coerce_shares(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def news_items_to_articles(news_items: Iterable[Dict[str, Any]]) -> List[NewsArticle]:
    """Convert normalized provider news dicts into :class:`NewsArticle` list.

    Accepts items produced by :meth:`ExternalDataProvider._news_item` as
    well as raw finnhub entries. The critical field is ``polarity``
    (already normalized by our providers); falls back to ``sentiment``
    if present.
    """
    out: List[NewsArticle] = []
    for item in news_items or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("polarity") or item.get("sentiment") or ""
        label = _NEWS_POLARITY_ALIASES.get(str(raw).strip().lower(), "neutral")
        out.append(
            NewsArticle(
                sentiment=label,
                title=str(item.get("title", "") or ""),
                url=str(item.get("url", "") or ""),
                published_at=str(item.get("published_at", "") or ""),
                metadata={
                    "source": item.get("source", ""),
                    "provider": item.get("provider", ""),
                    **(item.get("metadata") or {}),
                },
            )
        )
    return out


def finnhub_insider_to_trades(payload: Any) -> List[InsiderTrade]:
    """Convert Finnhub ``/stock/insider-transactions`` rows into trades.

    Finnhub returns ``{"data": [{"name", "share", "change", "filingDate",
    "transactionDate", ...}]}``. ``change`` is the signed share delta
    (positive = buy, negative = sell); if absent we fall back to
    ``share * sign(transactionPrice)`` heuristics, but in practice
    ``change`` is populated.
    """
    rows: List[Dict[str, Any]]
    if isinstance(payload, dict):
        rows = list(payload.get("data") or [])
    elif isinstance(payload, list):
        rows = list(payload)
    else:
        return []

    trades: List[InsiderTrade] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        change = row.get("change")
        if change is None:
            change = row.get("transactionShares")
        shares = _coerce_shares(change)
        trades.append(
            InsiderTrade(
                transaction_shares=shares,
                insider_role=str(row.get("name", "") or ""),
                filed_at=str(row.get("filingDate", "") or row.get("transactionDate", "") or ""),
                metadata={
                    "transaction_date": row.get("transactionDate", ""),
                    "transaction_price": row.get("transactionPrice"),
                    "transaction_code": row.get("transactionCode"),
                    "raw_change": change,
                },
            )
        )
    return trades
