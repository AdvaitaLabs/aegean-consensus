from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ProviderResult(dict):
    """Thin dict container for normalized provider payloads."""


class ExternalDataProvider(ABC):
    """Base class for external investment data providers."""

    provider_name: str = "provider"
    timeout_seconds: float = 4.0

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 4.0):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def fetch(self, symbol: str, market: str, asset_type: str) -> Dict[str, Any]:
        """Fetch normalized data for one asset."""
        raise NotImplementedError

    def _base_result(
        self,
        *,
        symbol: str,
        market: str,
        asset_type: str,
        status: str = "ok",
        message: str = "",
        signals: Optional[list[str]] = None,
    ) -> ProviderResult:
        return ProviderResult(
            provider=self.provider_name,
            market_data={
                "symbol": symbol,
                "market": market,
                "asset_type": asset_type,
            },
            fundamentals={},
            news=[],
            metadata={
                "status": status,
                "message": message,
                "signals": signals or [],
                "timeout_seconds": self.timeout_seconds,
            },
        )

    def _news_item(
        self,
        *,
        title: str,
        source: str = "",
        url: str = "",
        summary: str = "",
        polarity: str = "neutral",
        published_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "title": title,
            "source": source,
            "provider": self.provider_name,
            "url": url,
            "summary": summary,
            "polarity": polarity,
            "published_at": published_at,
            "metadata": metadata or {},
        }

    def _timeout_result(self, symbol: str, market: str, asset_type: str) -> ProviderResult:
        return self._base_result(
            symbol=symbol,
            market=market,
            asset_type=asset_type,
            status="timeout",
            message=f"Timed out after {self.timeout_seconds}s",
            signals=[f"{self.provider_name.upper()}_TIMEOUT"],
        )

    def _rate_limited_result(self, symbol: str, market: str, asset_type: str) -> ProviderResult:
        return self._base_result(
            symbol=symbol,
            market=market,
            asset_type=asset_type,
            status="rate_limited",
            message="Provider rate limited",
            signals=[f"{self.provider_name.upper()}_RATE_LIMITED"],
        )

    def _error_result(self, symbol: str, market: str, asset_type: str, message: str) -> ProviderResult:
        return self._base_result(
            symbol=symbol,
            market=market,
            asset_type=asset_type,
            status="error",
            message=message,
            signals=[f"{self.provider_name.upper()}_FAILED"],
        )

    @staticmethod
    def _env(name: str) -> Optional[str]:
        value = os.getenv(name)
        return value.strip() if value else None

