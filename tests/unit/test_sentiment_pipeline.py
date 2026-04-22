"""Unit tests for the three-tier sentiment pipeline."""

from __future__ import annotations

import pytest

from aegean.investment.sentiment import (
    InsiderTrade,
    NewsArticle,
    SentimentPipeline,
    classify_insider_trade,
    classify_news_article,
)


def test_classify_insider_trade_direction():
    assert classify_insider_trade(InsiderTrade(transaction_shares=100)) == "bullish"
    assert classify_insider_trade(InsiderTrade(transaction_shares=-100)) == "bearish"
    assert classify_insider_trade(InsiderTrade(transaction_shares=0)) == "neutral"
    assert classify_insider_trade(InsiderTrade(transaction_shares=float("nan"))) == "neutral"


def test_classify_news_article_labels():
    assert classify_news_article(NewsArticle(sentiment="positive")) == "bullish"
    assert classify_news_article(NewsArticle(sentiment="Negative")) == "bearish"
    assert classify_news_article(NewsArticle(sentiment="neutral")) == "neutral"
    assert classify_news_article(NewsArticle(sentiment="")) == "neutral"
    assert classify_news_article(NewsArticle(sentiment="bullish")) == "bullish"


def test_pipeline_rejects_bad_weights():
    with pytest.raises(ValueError):
        SentimentPipeline(insider_weight=-0.1)
    with pytest.raises(ValueError):
        SentimentPipeline(insider_weight=0, news_weight=0)
    with pytest.raises(ValueError):
        SentimentPipeline(threshold=1.0)


def test_default_weights_match_reference():
    p = SentimentPipeline()
    assert p.insider_weight == 0.3
    assert p.news_weight == 0.7


def test_news_heavy_overrides_insider_minority():
    # 1 bearish insider (weight 0.3 = 0.3) vs 3 bullish news (weight 0.7 = 2.1).
    pipeline = SentimentPipeline()
    result = pipeline.assess(
        insider_trades=[InsiderTrade(transaction_shares=-50)],
        news_articles=[
            NewsArticle(sentiment="positive"),
            NewsArticle(sentiment="positive"),
            NewsArticle(sentiment="positive"),
        ],
    )
    assert result.signal == "bullish"
    assert result.insider.signal == "bearish"
    assert result.news.signal == "bullish"
    assert result.total_weighted_bullish == pytest.approx(2.1)
    assert result.total_weighted_bearish == pytest.approx(0.3)


def test_empty_inputs_are_neutral_zero_confidence():
    pipeline = SentimentPipeline()
    result = pipeline.assess()
    assert result.signal == "neutral"
    assert result.confidence == 0.0
    assert result.insider.total == 0
    assert result.news.total == 0


def test_threshold_forces_neutral_when_close():
    pipeline = SentimentPipeline(threshold=0.3)
    # weighted_bull = 0.7, weighted_bear = 0.7 + 0.3 = 1.0 => margin ~0.176 < 0.3
    result = pipeline.assess(
        insider_trades=[InsiderTrade(transaction_shares=-1)],
        news_articles=[NewsArticle(sentiment="positive"), NewsArticle(sentiment="negative")],
    )
    assert result.signal == "neutral"


def test_to_dict_contains_three_tiers():
    pipeline = SentimentPipeline()
    result = pipeline.assess(
        insider_trades=[InsiderTrade(transaction_shares=10)],
        news_articles=[NewsArticle(sentiment="negative")],
    )
    payload = result.to_dict()
    assert set(payload) >= {
        "signal",
        "confidence",
        "insider_trading",
        "news_sentiment",
        "combined_analysis",
    }
    assert payload["insider_trading"]["metrics"]["weight"] == 0.3
    assert payload["news_sentiment"]["metrics"]["weight"] == 0.7


def test_all_neutral_inputs_yield_neutral_zero_confidence():
    pipeline = SentimentPipeline()
    result = pipeline.assess(
        insider_trades=[InsiderTrade(transaction_shares=0)],
        news_articles=[NewsArticle(sentiment="neutral"), NewsArticle(sentiment="")],
    )
    assert result.signal == "neutral"
    assert result.confidence == 0.0
    assert result.total_weighted_bullish == 0.0
    assert result.total_weighted_bearish == 0.0


def test_confidence_reflects_weighted_proportion():
    # 2 bullish news, 0 bearish => total_weight_capacity = 2 * 0.7 = 1.4
    # max weighted = 1.4. confidence = 1.0.
    pipeline = SentimentPipeline()
    result = pipeline.assess(
        news_articles=[NewsArticle(sentiment="positive"), NewsArticle(sentiment="positive")],
    )
    assert result.signal == "bullish"
    assert result.confidence == pytest.approx(1.0)
