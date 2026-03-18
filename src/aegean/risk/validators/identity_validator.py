"""
Identity Validator - KYA (Know Your Agent/Account) committee.

Responsibilities:
- Verify subject identity and credentials
- Check registration legitimacy
- Assess trust score history
- Detect identity anomalies (age of account, profile completeness)
"""

from typing import Optional, Dict, Any
import re
import logging

from aegean.risk.validators.base_validator import BaseValidator, score_to_risk_level
from aegean.risk.models import RiskRequest, RiskLevel, ValidatorType
from aegean.memory.global_memory import GlobalMemorySystem

logger = logging.getLogger(__name__)


class IdentityValidator(BaseValidator):
    """
    Specialist in identity verification and KYA (Know Your Agent).

    Pre-screening rules:
    - Trust score below threshold → elevated risk
    - New account (< 7 days) + high amount → high risk
    - Previously flagged multiple times → high risk

    LLM analysis focus:
    - Identity plausibility given context
    - Account history consistency
    - Jurisdiction-specific requirements
    """

    validator_type = ValidatorType.IDENTITY
    base_capability_weight = 0.9  # Identity is well-defined, high confidence domain

    def __init__(self, validator_id: str = "identity-v1", **kwargs):
        super().__init__(validator_id=validator_id, **kwargs)
        # Thresholds
        self.min_trust_score = self.config.get("min_trust_score", 0.3)
        self.new_account_days = self.config.get("new_account_days", 7)
        self.max_flag_count = self.config.get("max_flag_count", 3)

    def _pre_screen(self, request: RiskRequest) -> Optional[Dict[str, Any]]:
        """Fast identity rule checks."""
        subject = request.subject
        context = request.context
        indicators = []

        # Rule 1: Very low trust score
        if subject.trust_score < 0.1:
            return {
                "risk_level": RiskLevel.CRITICAL,
                "confidence": 0.95,
                "reasoning": f"Subject trust score critically low: {subject.trust_score:.2f}. "
                             f"Flagged {subject.flagged_count} times previously.",
                "indicators": ["critical_low_trust_score", "repeat_offender"],
            }

        # Rule 2: High flag history
        if subject.flagged_count >= self.max_flag_count:
            indicators.append("high_flag_count")

        # Rule 3: Low trust + large amount
        amount = context.amount or 0
        if subject.trust_score < self.min_trust_score and amount > 1000:
            indicators.append("low_trust_high_amount")

        # Rule 4: New account with significant transaction
        if subject.registered_at and amount > 500:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            registered = subject.registered_at
            if registered.tzinfo is None:
                from datetime import timezone
                registered = registered.replace(tzinfo=timezone.utc)
            age_days = (now - registered).days
            if age_days < self.new_account_days:
                indicators.append(f"new_account_{age_days}d_old")

        if len(indicators) >= 2:
            return {
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.80,
                "reasoning": f"Multiple identity risk signals detected: {', '.join(indicators)}",
                "indicators": indicators,
            }

        return None  # No strong signal, defer to LLM

    async def _analyze_with_llm(
        self,
        request: RiskRequest,
        rag_context: str,
        pre_result: Optional[Dict[str, Any]],
        context_hint: Optional[str],
    ) -> Dict[str, Any]:
        """LLM-based identity analysis."""
        if not self.llm_client:
            return self._heuristic_fallback(request)

        subject = request.subject
        context = request.context

        pre_notes = ""
        if pre_result:
            pre_notes = f"\n【规则预筛结果】\n{pre_result['reasoning']}"

        prompt = f"""你是一个专业的身份验证专家（KYA - Know Your Agent/Account）。

{rag_context}

【待评估主体】
- ID: {subject.subject_id}
- 类型: {subject.subject_type}
- 信任评分: {subject.trust_score:.2f}
- 历史交易次数: {subject.total_transactions}
- 历史标记次数: {subject.flagged_count}
- 注册时间: {subject.registered_at or '未知'}
- 司法管辖区: {subject.jurisdiction or '未知'}

【当前行为】
- 操作类型: {context.action_type}
- 描述: {context.description}
- 金额: {context.amount} {context.currency or ''}
- 地理位置: {context.geo_location or '未知'}
{pre_notes}

请从身份验证角度评估该主体的风险，输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体风险信号]
分析: [详细的身份验证分析，2-4句话]"""

        try:
            response = await self.llm_client.complete(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"Identity LLM call failed: {e}")
            return self._heuristic_fallback(request)

    def _get_rag_query(self, request: RiskRequest) -> str:
        return (
            f"identity verification {request.subject.subject_type} "
            f"trust score risk assessment KYA KYC"
        )

    def _get_rag_category(self) -> Optional[str]:
        return "identity_verification"

    @staticmethod
    def _parse_llm_response(response: str) -> Dict[str, Any]:
        """Parse structured LLM output."""
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

