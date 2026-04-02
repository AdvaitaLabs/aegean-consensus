"""Pydantic models for investment analysis workflows."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class InvestmentMode(str, Enum):
    FAST = "fast"
    AUTO = "auto"
    COLLABORATE = "collaborate"
    ROUNDTABLE = "roundtable"


class MarketCode(str, Enum):
    CN = "CN"
    HK = "HK"
    US = "US"


class AssetType(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FUND = "fund"
    CONVERTIBLE_BOND = "convertible_bond"
    FUTURES = "futures"
    OPTIONS = "options"
    CRYPTO = "crypto"


class RecommendationAction(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    WATCH = "watch"


class InvestmentAsset(BaseModel):
    symbol: str = Field(..., description="Asset symbol, e.g. AAPL / 600519.SH")
    market: MarketCode
    asset_type: AssetType


class InvestmentTimeframe(BaseModel):
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    lookback_window_days: int = Field(90, ge=1, le=3650)
    horizon: str = Field("1m", description="Expected holding horizon, e.g. 1w/1m/3m")


class InvestmentRecommendation(BaseModel):
    action: RecommendationAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    position_suggestion: Dict[str, float] = Field(default_factory=dict)


class InvestmentSummary(BaseModel):
    thesis: str
    key_drivers: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    agent_id: str
    signal: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class RiskGateResult(BaseModel):
    status: str = Field("pass", description="pass | challenge | reject")
    risk_level: str = Field("low")
    risk_indicators: List[str] = Field(default_factory=list)


class ConsensusResultView(BaseModel):
    enabled: bool = False
    rounds_used: int = 0
    consensus_reached: bool = False
    weighted_votes: Dict[str, float] = Field(default_factory=dict)


class InvestmentMetadata(BaseModel):
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0}
    )
    latency_ms: int = 0
    data_sources: List[str] = Field(default_factory=list)
    selected_skills: List[str] = Field(default_factory=list)
    task_type: str = ""
    constraints_applied_summary: Dict[str, Any] = Field(default_factory=dict)


class InvestmentAnalysisRequest(BaseModel):
    mode: InvestmentMode = InvestmentMode.AUTO
    asset: InvestmentAsset
    timeframe: InvestmentTimeframe = Field(default_factory=InvestmentTimeframe)

    risk_profile: str = Field("balanced", description="conservative|balanced|aggressive")
    objective: str = Field("balanced", description="alpha|defensive|income|balanced")

    market_snapshot: Optional[str] = None
    public_facts: List[str] = Field(default_factory=list)
    custom_question: Optional[str] = None

    portfolio_context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    private_context_refs: List[str] = Field(default_factory=list)

    user_id: str = Field("anonymous")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvestmentAnalysisResponse(BaseModel):
    request_id: str
    mode: InvestmentMode
    asset: InvestmentAsset
    timeframe: InvestmentTimeframe

    recommendation: InvestmentRecommendation
    summary: InvestmentSummary
    agent_outputs: List[AgentOutput] = Field(default_factory=list)

    risk_gate: RiskGateResult = Field(default_factory=RiskGateResult)
    consensus: ConsensusResultView = Field(default_factory=ConsensusResultView)

    report_markdown: str = ""
    metadata: InvestmentMetadata = Field(default_factory=InvestmentMetadata)

