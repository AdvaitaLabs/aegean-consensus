"""Three-tier sentiment pipeline (insider trades + company news)."""

from aegean.investment.sentiment.adapters import (
    finnhub_insider_to_trades,
    news_items_to_articles,
)
from aegean.investment.sentiment.pipeline import (
    InsiderTrade,
    NewsArticle,
    SentimentPipeline,
    SentimentResult,
    SentimentTier,
    classify_insider_trade,
    classify_news_article,
)

__all__ = [
    "InsiderTrade",
    "NewsArticle",
    "SentimentPipeline",
    "SentimentResult",
    "SentimentTier",
    "classify_insider_trade",
    "classify_news_article",
    "finnhub_insider_to_trades",
    "news_items_to_articles",
]
