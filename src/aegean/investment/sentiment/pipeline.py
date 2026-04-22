"""三级 sentiment 管道：insider 交易 + 公司新闻 → 综合信号。

参考 ai-hedge-fund/src/agents/sentiment.py：
  insider 权重 0.3，news 权重 0.7。
  三个层级分别产出单独结论再按权重合并，
  并在总体输出里保留明细，便于 panel 引用。

此模块只依赖标准库，不绑定任何具体 provider——调用方负责把
yfinance / finnhub / fmp 等返回的数据转换成 ``InsiderTrade`` /
``NewsArticle``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional


DEFAULT_INSIDER_WEIGHT = 0.3
DEFAULT_NEWS_WEIGHT = 0.7

SentimentTier = Literal["bullish", "bearish", "neutral"]


@dataclass
class InsiderTrade:
    transaction_shares: float
    insider_role: str = ""
    filed_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsArticle:
    sentiment: str = ""          # 原始标签：positive | negative | neutral | ""
    title: str = ""
    url: str = ""
    published_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TierBreakdown:
    signal: SentimentTier
    confidence: float            # 0-1
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0
    total: int = 0
    weight: float = 0.0
    weighted_bullish: float = 0.0
    weighted_bearish: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "metrics": {
                "total": self.total,
                "bullish": self.bullish,
                "bearish": self.bearish,
                "neutral": self.neutral,
                "weight": self.weight,
                "weighted_bullish": round(self.weighted_bullish, 4),
                "weighted_bearish": round(self.weighted_bearish, 4),
            },
        }


@dataclass
class SentimentResult:
    signal: SentimentTier
    confidence: float
    insider: TierBreakdown
    news: TierBreakdown
    total_weighted_bullish: float
    total_weighted_bearish: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "insider_trading": self.insider.to_dict(),
            "news_sentiment": self.news.to_dict(),
            "combined_analysis": {
                "total_weighted_bullish": round(self.total_weighted_bullish, 4),
                "total_weighted_bearish": round(self.total_weighted_bearish, 4),
                "signal_determination": (
                    f"{self.signal} based on weighted signal comparison"
                ),
            },
        }


def classify_insider_trade(trade: InsiderTrade) -> SentimentTier:
    """shares > 0 视为 bullish，< 0 视为 bearish，= 0 / NaN 视为 neutral。"""
    shares = trade.transaction_shares
    if shares is None:
        return "neutral"
    try:
        shares_f = float(shares)
    except (TypeError, ValueError):
        return "neutral"
    if shares_f > 0:
        return "bullish"
    if shares_f < 0:
        return "bearish"
    return "neutral"


def classify_news_article(article: NewsArticle) -> SentimentTier:
    label = (article.sentiment or "").strip().lower()
    if label in ("positive", "bullish", "pos", "+"):
        return "bullish"
    if label in ("negative", "bearish", "neg", "-"):
        return "bearish"
    return "neutral"


def _tier_signal(bullish: int, bearish: int) -> SentimentTier:
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def _tier_confidence(bullish: int, bearish: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(bullish, bearish) / total


class SentimentPipeline:
    """把 insider + news 两路信号合并成最终 sentiment 结果。

    Args:
        insider_weight: insider 这一级在最终加权里的权重。
        news_weight: news 这一级的权重。
        threshold: 最终 bullish vs bearish 权重需要达到的最小差值
            （相对两者之和）才判定为方向性信号，否则返回 neutral。
            默认 0，表示严格多数即可。
    """

    def __init__(
        self,
        insider_weight: float = DEFAULT_INSIDER_WEIGHT,
        news_weight: float = DEFAULT_NEWS_WEIGHT,
        threshold: float = 0.0,
    ) -> None:
        if insider_weight < 0 or news_weight < 0:
            raise ValueError("weights must be >= 0")
        total = insider_weight + news_weight
        if total <= 0:
            raise ValueError("insider_weight + news_weight must be > 0")
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1)")
        self.insider_weight = insider_weight
        self.news_weight = news_weight
        self.threshold = threshold

    def _summarize_insider(self, trades: Iterable[InsiderTrade]) -> TierBreakdown:
        bull = bear = neu = 0
        total = 0
        for trade in trades:
            total += 1
            tier = classify_insider_trade(trade)
            if tier == "bullish":
                bull += 1
            elif tier == "bearish":
                bear += 1
            else:
                neu += 1
        return TierBreakdown(
            signal=_tier_signal(bull, bear),
            confidence=_tier_confidence(bull, bear, total),
            bullish=bull,
            bearish=bear,
            neutral=neu,
            total=total,
            weight=self.insider_weight,
            weighted_bullish=bull * self.insider_weight,
            weighted_bearish=bear * self.insider_weight,
        )

    def _summarize_news(self, articles: Iterable[NewsArticle]) -> TierBreakdown:
        bull = bear = neu = 0
        total = 0
        for article in articles:
            total += 1
            tier = classify_news_article(article)
            if tier == "bullish":
                bull += 1
            elif tier == "bearish":
                bear += 1
            else:
                neu += 1
        return TierBreakdown(
            signal=_tier_signal(bull, bear),
            confidence=_tier_confidence(bull, bear, total),
            bullish=bull,
            bearish=bear,
            neutral=neu,
            total=total,
            weight=self.news_weight,
            weighted_bullish=bull * self.news_weight,
            weighted_bearish=bear * self.news_weight,
        )

    def assess(
        self,
        insider_trades: Optional[Iterable[InsiderTrade]] = None,
        news_articles: Optional[Iterable[NewsArticle]] = None,
    ) -> SentimentResult:
        insider = self._summarize_insider(insider_trades or [])
        news = self._summarize_news(news_articles or [])

        total_bull = insider.weighted_bullish + news.weighted_bullish
        total_bear = insider.weighted_bearish + news.weighted_bearish
        directional_sum = total_bull + total_bear

        if directional_sum <= 0:
            overall: SentimentTier = "neutral"
            confidence = 0.0
        else:
            margin = abs(total_bull - total_bear) / directional_sum
            if margin < self.threshold:
                overall = "neutral"
            elif total_bull > total_bear:
                overall = "bullish"
            elif total_bear > total_bull:
                overall = "bearish"
            else:
                overall = "neutral"
            total_weight_capacity = (
                insider.total * self.insider_weight + news.total * self.news_weight
            )
            confidence = (
                max(total_bull, total_bear) / total_weight_capacity
                if total_weight_capacity > 0
                else 0.0
            )

        return SentimentResult(
            signal=overall,
            confidence=confidence,
            insider=insider,
            news=news,
            total_weighted_bullish=total_bull,
            total_weighted_bearish=total_bear,
        )
