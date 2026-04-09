"""Pydantic models for investment analysis workflows."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
    display_name: str = ""
    exchange: str = ""


class InvestmentTimeframe(BaseModel):
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    lookback_window_days: int = Field(90, ge=1, le=3650)
    horizon: str = Field("1m", description="Expected holding horizon, e.g. 1w/1m/3m")


class InvestmentRecommendation(BaseModel):
    action: RecommendationAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    position_suggestion: Dict[str, float] = Field(default_factory=dict)
    decision_rationale: str = ""


class InvestmentSummary(BaseModel):
    thesis: str
    key_drivers: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)


class AnalysisFramework(BaseModel):
    style: str = "multi_agent_investment_review"
    task_type: str = ""
    selected_skills: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    why_selected: List[str] = Field(default_factory=list)


class CatalystItem(BaseModel):
    name: str
    direction: str = "two_way"
    importance: str = "medium"
    time_horizon: str = ""


class ScenarioItem(BaseModel):
    name: str
    probability: float = Field(..., ge=0.0, le=1.0)
    view: str
    description: str = ""


class AgentOutput(BaseModel):
    agent_id: str
    role: str = ""
    title: str = ""
    signal: str
    stance: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = ""
    evidence: List[str] = Field(default_factory=list)
    risks_flagged: List[str] = Field(default_factory=list)


class DisagreementSummary(BaseModel):
    main_conflict: str = ""
    agreement_points: List[str] = Field(default_factory=list)
    disagreement_points: List[str] = Field(default_factory=list)


class DiscussionAgentEntry(BaseModel):
    agent_id: str
    role: str = ""
    stance: str = ""
    previous_signal: Optional[str] = None
    current_signal: str = ""
    changed_position: bool = False
    summary: str = ""
    message: str = ""
    evidence: List[str] = Field(default_factory=list)


class DiscussionRound(BaseModel):
    round_number: int
    candidate_action: str = ""
    candidate_confidence: float = Field(0.0, ge=0.0, le=1.0)
    agents: List[DiscussionAgentEntry] = Field(default_factory=list)
    agreement_points: List[str] = Field(default_factory=list)
    disagreement_points: List[str] = Field(default_factory=list)


class ConsensusTrace(BaseModel):
    discussion_enabled: bool = False
    final_summary: str = ""
    rounds: List[DiscussionRound] = Field(default_factory=list)


class PolicyOverrides(BaseModel):
    input_action: str = ""
    output_action: str = ""
    input_target_exposure_pct: float = 0.0
    output_target_exposure_pct: float = 0.0
    binding_cap: str = "none"
    effective_caps: Dict[str, float] = Field(default_factory=dict)
    triggered_rules: List[str] = Field(default_factory=list)
    human_readable_explanation: str = ""


class RiskGateResult(BaseModel):
    status: str = Field("pass", description="pass | challenge | reject")
    risk_level: str = Field("low")
    risk_indicators: List[str] = Field(default_factory=list)
    review_summary: str = ""


class ConsensusResultView(BaseModel):
    enabled: bool = False
    rounds_used: int = 0
    consensus_reached: bool = False
    final_action: str = ""
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
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    event_count: int = 0
    schema_version: str = "investment_analysis.v2"
    debug_flags: Dict[str, bool] = Field(default_factory=dict)


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
    status: str = "completed"
    mode: InvestmentMode
    asset: InvestmentAsset
    timeframe: InvestmentTimeframe

    analysis_framework: AnalysisFramework = Field(default_factory=AnalysisFramework)
    recommendation: InvestmentRecommendation
    summary: InvestmentSummary
    bull_case: List[str] = Field(default_factory=list)
    bear_case: List[str] = Field(default_factory=list)
    catalysts: List[CatalystItem] = Field(default_factory=list)
    scenario_analysis: List[ScenarioItem] = Field(default_factory=list)
    agent_outputs: List[AgentOutput] = Field(default_factory=list)
    disagreement_summary: DisagreementSummary = Field(default_factory=DisagreementSummary)

    risk_gate: RiskGateResult = Field(default_factory=RiskGateResult)
    consensus: ConsensusResultView = Field(default_factory=ConsensusResultView)
    consensus_trace: ConsensusTrace = Field(default_factory=ConsensusTrace)
    policy_overrides: PolicyOverrides = Field(default_factory=PolicyOverrides)

    report_markdown: str = ""
    metadata: InvestmentMetadata = Field(default_factory=InvestmentMetadata)
