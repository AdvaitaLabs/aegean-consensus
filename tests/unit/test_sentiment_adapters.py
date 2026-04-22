"""Unit tests for the sentiment provider adapters."""

from __future__ import annotations

from aegean.investment.sentiment import (
    NewsArticle,
    SentimentPipeline,
    finnhub_insider_to_trades,
    news_items_to_articles,
)


def test_news_items_to_articles_maps_polarity():
    items = [
        {"title": "a", "polarity": "negative", "source": "Finnhub"},
        {"title": "b", "polarity": "positive"},
        {"title": "c", "polarity": "neutral"},
        {"title": "d", "sentiment": "bullish"},
        {"title": "e"},  # missing => neutral
        "garbage",       # dropped
    ]
    articles = news_items_to_articles(items)
    assert [a.sentiment for a in articles] == [
        "negative", "positive", "neutral", "positive", "neutral",
    ]
    assert articles[0].title == "a"
    assert articles[0].metadata["source"] == "Finnhub"


def test_finnhub_insider_to_trades_reads_change_field():
    payload = {
        "symbol": "AAPL",
        "data": [
            {"name": "CEO", "change": 1000, "filingDate": "2026-01-02"},
            {"name": "CFO", "change": -500, "transactionDate": "2026-01-03"},
            {"name": "Director", "change": None, "transactionShares": -200},
            {"name": "Weird", "change": "not-a-number"},
            "garbage",
        ],
    }
    trades = finnhub_insider_to_trades(payload)
    assert [t.transaction_shares for t in trades] == [1000.0, -500.0, -200.0, 0.0]
    assert trades[0].insider_role == "CEO"
    assert trades[1].filed_at == "2026-01-03"


def test_finnhub_insider_handles_missing_or_bad_payload():
    assert finnhub_insider_to_trades(None) == []
    assert finnhub_insider_to_trades("oops") == []
    assert finnhub_insider_to_trades({"data": None}) == []


def test_end_to_end_pipeline_via_adapters():
    # 3 bullish news, 1 bearish insider.
    news_raw = [
        {"title": "beat", "polarity": "positive"},
        {"title": "raise", "polarity": "positive"},
        {"title": "good", "polarity": "positive"},
    ]
    insider_raw = {"data": [{"change": -500, "name": "CFO"}]}
    articles = news_items_to_articles(news_raw)
    trades = finnhub_insider_to_trades(insider_raw)
    result = SentimentPipeline().assess(trades, articles)
    assert result.signal == "bullish"
    assert result.news.bullish == 3
    assert result.insider.bearish == 1


def test_news_adapter_skips_non_dict_entries():
    articles = news_items_to_articles([None, 123, {"title": "ok", "polarity": "positive"}])
    assert len(articles) == 1
    assert articles[0].sentiment == "positive"
