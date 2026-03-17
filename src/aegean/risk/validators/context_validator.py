"""
Context Validator - Reasoning trace & contextual analysis committee.

Responsibilities:
- Analyze the agent/user reasoning trace (trace_context)
- Verify action consistency with stated purpose
- Detect prompt injection or reasoning manipulation
- Assess overall context plausibility
- Cross-reference action against prior behavior patterns

This validator is unique: it treats the trace_context field as a
first-class signal, mirroring Trustline's use of agent reasoning
traces for contextual legitimacy verification.
"""

from typing import Optional, Dict, Any
import logging

from aegean.risk.validators.base_validator import BaseValidator
from aegean.risk.models import RiskRequest, RiskLevel, ValidatorType

logger = logging.getLogger(__name__)

# Keywords suggesting suspicious or injected reasoning
SUSPICIOUS_TRACE_KEYWORDS = [
    "ignore previous", "ignore instructions", "jailbreak",
    "bypass", "override security", "disregard", "pretend you are",
    "act as if", "forget your", "new instruction",
]


class ContextValidator(BaseValidator):
    """
    Specialist in reasoning trace and contextual legitimacy analysis.

    This is the most "AI-native" validator - it evaluates whether
    the reasoning that led to a request makes sense, and whether
    the action is consistent with the stated context.

    Pre-screening rules:
    - Prompt injection keywords in trace_context
    - No trace provided for high-value actions
    - Stated purpose contradicts action type

    LLM analysis focus:
    - Is the reasoning chain internally consistent?
    - Does the action logically follow from the stated purpose?
    - Are there signs of manipulation or deception?
    """

    validator_type = ValidatorType.CONTEXT
    base_capability_weight = 0.80  # Context analysis is nuanced, moderate weight

    def __init__(self, validator_id: str = "context-v1", **kwargs):
        super().__init__(validator_id=validator_id, **kwargs)
        self.require_trace_above_amount = self.config.get(
            "require_trace_above_amount", 5_000
        )

    def _pre_screen(self, request: RiskRequest) -> Optional[Dict[str, Any]]:
        """Fast context consistency checks."""
        ctx = request.context
        indicators = []
        trace = (ctx.trace_context or "").lower()

        # Rule 1: Prompt injection detection in trace
        for keyword in SUSPICIOUS_TRACE_KEYWORDS:
            if keyword in trace:
                return {
                    "risk_level": RiskLevel.CRITICAL,
                    "confidence": 0.97,
                    "reasoning": (
                        f"Potential prompt injection detected in reasoning trace. "
                        f"Suspicious keyword: '{keyword}'"
                    ),
                    "indicators": ["prompt_injection_detected", f"keyword_{keyword.replace(' ', '_')}"],
                }

        # Rule 2: High-value action with no trace context
        amount = ctx.amount or 0
        if amount >= self.require_trace_above_amount and not ctx.trace_context:
            indicators.append("missing_trace_high_value_action")

        # Rule 3: Action type vs description mismatch (simple keyword check)
        description_lower = ctx.description.lower()
        if ctx.action_type == "payment" and not any(
            kw in description_lower
            for kw in ["pay", "transfer", "send", "purchase", "buy", "支付", "转账", "购买"]
        ):
            indicators.append("action_description_mismatch")

        if len(indicators) >= 2:
            return {
                "risk_level": RiskLevel.HIGH,
                "confidence": 0.75,
                "reasoning": f"Context consistency issues: {', '.join(indicators)}",
                "indicators": indicators,
            }

        if indicators:
            return {
                "risk_level": RiskLevel.MEDIUM,
                "confidence": 0.55,
                "reasoning": f"Context signal: {indicators[0]}",
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
        """LLM-based context and reasoning trace analysis."""
        if not self.llm_client:
            return self._heuristic_fallback(request)

        subject = request.subject
        ctx = request.context
        pre_notes = f"\n【规则预筛结果】\n{pre_result['reasoning']}" if pre_result else ""
        trace_section = (
            f"\n【推理轨迹 (trace_context)】\n{ctx.trace_context}"
            if ctx.trace_context
            else "\n【推理轨迹】未提供"
        )

        prompt = f"""你是一个专业的上下文与推理链分析专家。

{rag_context}

【请求主体】
- ID: {subject.subject_id} (类型: {subject.subject_type})
- 信任评分: {subject.trust_score:.2f}

【请求详情】
- 操作类型: {ctx.action_type}
- 描述: {ctx.description}
- 金额: {ctx.amount} {ctx.currency or ''}
- 渠道: {ctx.channel or '未知'}
{trace_section}
{pre_notes}

请从上下文一致性和推理合理性角度评估：
1. 推理链是否内部一致、逻辑通顺？
2. 当前操作是否与描述的目的相符？
3. 是否存在推理操纵或欺骗迹象？

输出以下格式：
风险等级: [low/medium/high/critical]
置信度: [0.0-1.0]
风险指标: [用逗号分隔的具体上下文风险信号]
分析: [详细分析，2-4句话]"""

        try:
            response = await self.llm_client.complete(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"Context LLM call failed: {e}")
            return self._heuristic_fallback(request)

    def _get_rag_query(self, request: RiskRequest) -> str:
        return (
            f"context verification reasoning trace consistency "
            f"{request.context.action_type} agent behavior legitimacy"
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
