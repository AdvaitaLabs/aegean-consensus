"""
Anomaly Validator - Behavioral and contextual anomaly detection committee.

Responsibilities:
- Detect unusual behavioral patterns vs baseline
- Geographic anomalies (impossible travel, high-risk regions)
- Device/IP anomalies
- Transaction velocity spikes
- Time-of-day anomalies
"""

from typing import Optional, Dict, Any
import logging

from aegean.risk.validators.base_validator import BaseValidator
from aegean.risk.models import RiskRequest, RiskLevel, ValidatorType
from aegean.memory.global_memory import GlobalMemorySystem

logger = logging.getLogger(__name__)

# High-risk regions (simplified example set)
HIGH_RISK_REGIONS = {
    "KP", "IR", "SY", "CU",  # OFAC sanctioned
    "NG", "VN",               # High fraud regions (illustrative)
}


class AnomalyValidator(BaseValidator):
    """
    Specialist in behavioral and contextual anomaly detection.

    Pre-screening rules:
    - Velocity: too many transactions in short window
    - Geographic: high-risk regions or impossible travel
    - Time: unusual hours for the account type

    LLM analysis focus:
    - Behavioral pattern deviation from historical baseline
    - Context consistency (does the action make sense given history?)
    - Environmental signals (IP, device, geo correlation)
    """

    validator_type = ValidatorType.ANOMALY
    base_capability_weight = 0.85

    def __init__(self, validator_id: str = "anomaly-v1", **kwargs):
        super().__init__(validator_id=validator_id, **kwargs)
        self.velocity_threshold = self.config.get("velocity_threshold", 10)
        self.velocity_amount_threshold = self.config.get("velocity_amount_threshold", 5000.0)

    def _pre_screen(self, request: RiskRequest) -> Optional[Dict[str, Any]]:
        """Fast anomaly rule checks."""
        context = request.context
        indicators = []

        # Rule 1: Transaction velocity spike
        if context.recent_transaction_count >= self.velocity_threshold:
            indicators.append(f"velocity_spike_{context.recent_transaction_count}_in_1h")

        # Rule 2: High cumulative amount in short window
        if context.recent_transaction_amount >= self.velocity_amount_threshold:
            indicators.append(
                f"high_velocity_amount_{context.recent_transaction_amount:.0f}"
            )

        # Rule 3: High-risk geographic region
        if context.geo_location:
            country = context.geo_location.upper().split(",")[-1].strip()
            if country in HIGH_RISK_REGIONS:
                indicators.append(f"high_risk_region_{country}")
                return {
                    "risk_level": RiskLevel.CRITICAL,
                    "confidence": 0.96,
                    "reasoning": f"Transaction from OFAC/high-risk region: {country}.",
                    "indicators": [f"sanctioned_region_{country}"],
                }

        # Rule 4: Combined velocity + amount signal
        if len(indicators) >= 2:
            return {
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.82,
                "reasoning": f"Multiple velocity anomalies detected: {', '.join(indicators)}",
                "indicators": indicators,
            }

        if indicators:
            # Single weak signal — note it but defer to LLM
            return {
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.55,
                "reasoning": f"Mild anomaly signal: {indicators[0]}",
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
        """LLM-based anomaly analysis."""
        if not self.llm_client:
            return self._heuristic_fallback(request)

        subject = request.subject
        ctx = request.context
        pre_notes = f"\n【规则预筛结果】\n{pre_result['reasoning']}" if pre_result else ""

        prompt = f"""你是一个专业的异常行为检测专家。

{rag_context}

【主体背景】
- ID: {subject.subject_id} (类型: {subject.subject_type})
- 历史交易总数: {subject.total_transactions}
- 信任评分: {subject.trust_score:.2f}

【当前行为环境】
- 操作: {ctx.action_type} - {ctx.description}
- 金额: {ctx.amount} {ctx.currency or ''}
- 地理位置: {ctx.geo_location or '未知'}
- 设备ID: {ctx.device_id or '未知'}
- IP: {ctx.ip_address or '未知'}
- 渠道: {ctx.channel or '未知'}
- 近1小时交易次数: {ctx.recent_transaction_count}
- 近1小时交易总额: {ctx.recent_transaction_amount}
{pre_notes}

请从行为异常检测角度评估风险，输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体异常信号]
分析: [详细的行为异常分析，2-4句话]"""

        try:
            response = await self.llm_client.complete(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"Anomaly LLM call failed: {e}")
            return self._heuristic_fallback(request)

    def _get_rag_query(self, request: RiskRequest) -> str:
        return (
            f"behavioral anomaly detection transaction velocity "
            f"{request.context.action_type} fraud pattern unusual activity"
        )

    def _get_rag_category(self) -> Optional[str]:
        return "fraud_patterns"

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

