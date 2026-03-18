"""
Amount Validator - Transaction amount, frequency & velocity committee.

Responsibilities:
- Absolute amount threshold checks
- Per-period spending limit enforcement
- Frequency / velocity anomalies
- Round-number detection (common in fraud)
- Amount vs. subject profile consistency
"""

from typing import Optional, Dict, Any
import logging

from aegean.risk.validators.base_validator import BaseValidator
from aegean.risk.models import RiskRequest, RiskLevel, ValidatorType

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden per project via config)
DEFAULT_SINGLE_LIMIT = 50_000   # Single transaction hard limit
DEFAULT_HOURLY_LIMIT = 20_000   # Hourly aggregate limit
ROUND_NUMBER_MULTIPLES = [1000, 5000, 10000]  # Suspiciously round amounts


class AmountValidator(BaseValidator):
    """
    Specialist in amount and velocity risk assessment.

    Pre-screening rules:
    - Single transaction exceeds hard limit
    - Hourly aggregate exceeds limit
    - Suspiciously round number (common fraud signal)
    - Amount inconsistent with account history

    LLM analysis focus:
    - Whether the amount is contextually justified
    - Amount-to-income/trust ratio assessment
    - Pattern comparison with similar subjects
    """

    validator_type = ValidatorType.AMOUNT
    base_capability_weight = 0.88

    def __init__(self, validator_id: str = "amount-v1", **kwargs):
        super().__init__(validator_id=validator_id, **kwargs)
        self.single_limit = self.config.get("single_limit", DEFAULT_SINGLE_LIMIT)
        self.hourly_limit = self.config.get("hourly_limit", DEFAULT_HOURLY_LIMIT)

    def _pre_screen(self, request: RiskRequest) -> Optional[Dict[str, Any]]:
        """Fast amount and velocity rule checks."""
        ctx = request.context
        subject = request.subject
        amount = ctx.amount or 0
        indicators = []

        # Rule 1: Hard single-transaction limit
        if amount > self.single_limit:
            return {
                "risk_level": RiskLevel.CRITICAL,
                "confidence": 0.97,
                "reasoning": (
                    f"Single transaction amount {amount:,.0f} exceeds hard limit "
                    f"{self.single_limit:,.0f}."
                ),
                "indicators": [f"exceeds_single_limit_{amount:.0f}"],
            }

        # Rule 2: Hourly aggregate limit
        projected_hourly = ctx.recent_transaction_amount + amount
        if projected_hourly > self.hourly_limit:
            indicators.append(
                f"hourly_limit_breach_{projected_hourly:.0f}_vs_{self.hourly_limit:.0f}"
            )

        # Rule 3: Round number detection
        for multiple in ROUND_NUMBER_MULTIPLES:
            if amount > 0 and amount % multiple == 0:
                indicators.append(f"round_number_{amount:.0f}")
                break

        # Rule 4: Amount vs trust score inconsistency
        # Low-trust subjects attempting high-value transactions
        if subject.trust_score < 0.5 and amount > 10_000:
            indicators.append(
                f"high_amount_low_trust_{subject.trust_score:.2f}"
            )

        # Rule 5: First-time large amount (no history)
        if subject.total_transactions == 0 and amount > 1_000:
            indicators.append("first_transaction_high_value")

        if len(indicators) >= 2:
            return {
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.80,
                "reasoning": f"Multiple amount risk signals: {', '.join(indicators)}",
                "indicators": indicators,
            }

        if indicators:
            return {
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.58,
                "reasoning": f"Amount signal detected: {indicators[0]}",
                "indicators": indicators,
            }

        return None

    async def _analyze_with_llm(
        self,
        request: RiskRequest,
        rag_context: str,
        pre_result: Optional[Dict[str, Any]],
        context_hint: Optional[str],
    ) -> Dict[str, Any]:
        """LLM-based amount/velocity analysis."""
        if not self.llm_client:
            return self._heuristic_fallback(request)

        subject = request.subject
        ctx = request.context
        pre_notes = f"\n【规则预筛结果】\n{pre_result['reasoning']}" if pre_result else ""

        prompt = f"""你是一个专业的交易金额与频率风险分析师。

{rag_context}

【主体画像】
- ID: {subject.subject_id} (信任评分: {subject.trust_score:.2f})
- 历史交易总数: {subject.total_transactions}
- 历史标记次数: {subject.flagged_count}

【本次交易】
- 操作: {ctx.action_type}
- 金额: {ctx.amount:,.2f} {ctx.currency or ''}
- 描述: {ctx.description}

【近期行为】
- 近1小时交易次数: {ctx.recent_transaction_count}
- 近1小时累计金额: {ctx.recent_transaction_amount:,.2f}
- 单笔限额: {self.single_limit:,.0f}
- 小时限额: {self.hourly_limit:,.0f}
{pre_notes}

请从金额与频率风险角度评估，判断金额是否与主体背景相符，输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体风险信号]
分析: [详细分析，2-4句话]"""

        try:
            response = await self.llm_client.complete(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"Amount LLM call failed: {e}")
            return self._heuristic_fallback(request)

    def _get_rag_query(self, request: RiskRequest) -> str:
        return (
            f"transaction amount limit velocity fraud detection "
            f"{request.context.action_type} {request.context.amount or 0:.0f} threshold"
        )

    def _get_rag_category(self) -> Optional[str]:
        return "risk_indicators"

    @staticmethod
    def _parse_llm_response(response: str) -> Dict[str, Any]:
        lines = response.strip().split("\n")
        result = {
            "risk_level": RiskLevel.MEDIUM,
            "confidence": 0.5,
            "reasoning": response,
            "indicators": [],
        }
        for line in lines:
            line = line.strip()
            if line.startswith("风险等级:"):
                val = line.split(":", 1)[1].strip().lower()
                level_map = {"low": RiskLevel.LOW, "medium": RiskLevel.MEDIUM,
                             "high": RiskLevel.HIGH, "critical": RiskLevel.CRITICAL}
                result["risk_level"] = level_map.get(val, RiskLevel.MEDIUM)
            elif line.startswith("置信度:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("风险指标:"):
                raw = line.split(":", 1)[1].strip()
                result["indicators"] = [i.strip() for i in raw.split(",") if i.strip()]
            elif line.startswith("分析:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
        return result

