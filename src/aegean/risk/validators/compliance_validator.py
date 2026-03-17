"""
Compliance Validator - Regulatory & AML compliance committee.

Responsibilities:
- AML (Anti-Money Laundering) rule checks
- Sanctions screening (OFAC, UN, EU lists)
- Transaction reporting thresholds (e.g. CTR in US: >$10k)
- Jurisdiction-specific regulatory requirements
- KYC completeness verification
"""

from typing import Optional, Dict, Any
import logging

from aegean.risk.validators.base_validator import BaseValidator
from aegean.risk.models import RiskRequest, RiskLevel, ValidatorType

logger = logging.getLogger(__name__)

# AML reporting thresholds by currency (simplified)
CTR_THRESHOLDS: Dict[str, float] = {
    "USD": 10_000,
    "EUR": 10_000,
    "CNY": 50_000,  # RMB large transaction reporting threshold
    "GBP": 10_000,
    "DEFAULT": 10_000,
}

# Structuring detection: flag if amount is >85% of threshold
STRUCTURING_RATIO = 0.85


class ComplianceValidator(BaseValidator):
    """
    Specialist in regulatory compliance and AML.

    Pre-screening rules:
    - Amount exceeds CTR reporting threshold
    - Amount suspiciously close to threshold (structuring)
    - Cross-border high-value transfer
    - Multiple recent transactions near threshold

    LLM analysis focus:
    - AML typology matching
    - Regulatory requirement applicability
    - Jurisdiction-specific compliance gaps
    """

    validator_type = ValidatorType.COMPLIANCE
    base_capability_weight = 0.95  # Compliance rules are precise, high weight

    def __init__(self, validator_id: str = "compliance-v1", **kwargs):
        super().__init__(validator_id=validator_id, **kwargs)

    def _pre_screen(self, request: RiskRequest) -> Optional[Dict[str, Any]]:
        """Fast AML/compliance rule checks."""
        context = request.context
        indicators = []
        amount = context.amount or 0
        currency = (context.currency or "DEFAULT").upper()
        threshold = CTR_THRESHOLDS.get(currency, CTR_THRESHOLDS["DEFAULT"])

        # Rule 1: Exceeds CTR threshold - mandatory reporting
        if amount >= threshold:
            indicators.append(f"ctr_threshold_exceeded_{currency}_{amount:.0f}")
            return {
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.92,
                "reasoning": (
                    f"Transaction amount {amount} {currency} exceeds mandatory reporting "
                    f"threshold of {threshold}. CTR filing required."
                ),
                "indicators": indicators,
            }

        # Rule 2: Structuring detection (just below threshold)
        if amount >= threshold * STRUCTURING_RATIO:
            indicators.append(f"potential_structuring_{amount:.0f}_vs_{threshold:.0f}")

        # Rule 3: Multiple recent transactions suggesting structuring
        if (
            context.recent_transaction_count >= 3
            and context.recent_transaction_amount >= threshold * STRUCTURING_RATIO
        ):
            indicators.append(
                f"velocity_structuring_suspicion_"
                f"{context.recent_transaction_count}x_"
                f"{context.recent_transaction_amount:.0f}"
            )

        # Rule 4: Cross-border high-value
        subject_jurisdiction = request.subject.jurisdiction or ""
        tx_geo = context.geo_location or ""
        if (
            amount >= 5_000
            and subject_jurisdiction
            and tx_geo
            and subject_jurisdiction.upper() not in tx_geo.upper()
        ):
            indicators.append("cross_border_high_value")

        if len(indicators) >= 2:
            return {
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.78,
                "reasoning": f"Multiple AML indicators: {', '.join(indicators)}",
                "indicators": indicators,
            }

        if indicators:
            return {
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.60,
                "reasoning": f"AML compliance signal detected: {indicators[0]}",
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
        """LLM-based AML/compliance analysis."""
        if not self.llm_client:
            return self._heuristic_fallback(request)

        subject = request.subject
        ctx = request.context
        pre_notes = f"\n【规则预筛结果】\n{pre_result['reasoning']}" if pre_result else ""
        currency = ctx.currency or "USD"
        threshold = CTR_THRESHOLDS.get(currency.upper(), CTR_THRESHOLDS["DEFAULT"])

        prompt = f"""你是一个专业的反洗钱(AML)合规专家。

{rag_context}

【合规背景信息】
- {currency} 大额交易申报门槛: {threshold:,.0f}
- 当前交易金额: {ctx.amount} {currency}
- 主体司法管辖区: {subject.jurisdiction or '未知'}
- 交易发起地: {ctx.geo_location or '未知'}

【主体信息】
- ID: {subject.subject_id}
- 历史交易: {subject.total_transactions} 笔
- 历史标记: {subject.flagged_count} 次

【交易详情】
- 操作: {ctx.action_type} - {ctx.description}
- 对手方: {ctx.counterparty_id or '未知'}
- 近1小时: {ctx.recent_transaction_count} 笔, 合计 {ctx.recent_transaction_amount} {currency}
{pre_notes}

请从AML合规角度评估，检查洗钱类型（分层、整合、拆分等），输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体AML风险信号]
分析: [详细的合规分析，2-4句话]"""

        try:
            response = await self.llm_client.complete(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"Compliance LLM call failed: {e}")
            return self._heuristic_fallback(request)

    def _get_rag_query(self, request: RiskRequest) -> str:
        amount = request.context.amount or 0
        currency = request.context.currency or "USD"
        return (
            f"AML anti-money laundering compliance {currency} {amount:.0f} "
            f"CTR reporting threshold structuring regulations"
        )

    def _get_rag_category(self) -> Optional[str]:
        return "aml_regulations"

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
