#!/usr/bin/env python3
"""
Script B: Generate synthetic risk cases using LLM and seed into ExperienceBase.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --count 50 --model gpt-4o
    python scripts/generate_synthetic.py --dry-run
"""

import asyncio
import argparse
import json
import sys
import os
import logging
from typing import List, Dict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_synthetic")


DETERMINISTIC_CASES: List[Dict] = [
    {
        "task": "用户在3分钟内从上海和北京分别发起支付请求，金额各5000元",
        "final_answer": "reject:high:0.96",
        "risk_level": "high", "decision": "reject",
        "reasoning": "地理位置不可能：上海到北京1200km，3分钟内无法到达，疑似账号被盗。",
        "indicators": ["impossible_travel", "geo_anomaly", "velocity_anomaly"],
        "outcome": "fraud_confirmed", "rounds_used": 1,
    },
    {
        "task": "新注册账号（注册时间2小时前），首笔交易即为向境外账户转账8000美元",
        "final_answer": "reject:high:0.91",
        "risk_level": "high", "decision": "reject",
        "reasoning": "新账号高价值跨境转账：账号年龄2小时，首笔即大额跨境，符合欺诈账号特征。",
        "indicators": ["new_account_2h_old", "cross_border_high_value", "first_transaction_high_value"],
        "outcome": "fraud_confirmed", "rounds_used": 1,
    },
    {
        "task": "凌晨3:17分，用户连续发起9笔499元转账，收款方为9个不同账户",
        "final_answer": "reject:high:0.89",
        "risk_level": "high", "decision": "reject",
        "reasoning": "典型拆分交易（Structuring）：金额低于门槛，凌晨时段，多收款方，疑似洗钱。",
        "indicators": ["structuring_pattern", "unusual_time_3am", "velocity_spike_9x", "multiple_recipients"],
        "outcome": "aml_violation_confirmed", "rounds_used": 2,
    },
    {
        "task": "企业账户向供应商支付月度货款150万元，附有完整采购订单和发票，历史每月有类似记录",
        "final_answer": "approve:low:0.94",
        "risk_level": "low", "decision": "approve",
        "reasoning": "正常业务支付：有完整文件佐证，历史模式一致，无异常信号。",
        "indicators": [],
        "outcome": "legitimate_confirmed", "rounds_used": 1,
    },
    {
        "task": "用户申请信用卡，年收入声称50万，但过去3个月频繁更换地址，有4个未结清小额贷款",
        "final_answer": "review:medium:0.72",
        "risk_level": "medium", "decision": "review",
        "reasoning": "信用风险中等：收入无法核实，地址不稳定，多笔小额贷款未还，建议人工审核。",
        "indicators": ["address_instability", "multiple_open_loans", "unverified_income"],
        "outcome": "requires_review", "rounds_used": 2,
    },
    {
        "task": "来自朝鲜IP地址的支付请求，金额2000美元，收款方为加密货币交易所",
        "final_answer": "reject:critical:0.99",
        "risk_level": "critical", "decision": "reject",
        "reasoning": "OFAC制裁国家：朝鲜在制裁名单，任何支付均违反美国法律，必须拒绝并上报。",
        "indicators": ["sanctioned_region_KP", "crypto_exchange_recipient", "ofac_violation"],
        "outcome": "blocked_compliance", "rounds_used": 1,
    },
    {
        "task": "用户在1小时内发起12笔转账，总额18000元，均为9800-9900元（略低于10000报告门槛）",
        "final_answer": "reject:high:0.93",
        "risk_level": "high", "decision": "reject",
        "reasoning": "拆分交易：金额集中在报告门槛85%-99%，高频，高度疑似规避大额申报。",
        "indicators": ["structuring_below_threshold", "velocity_spike_12x", "ctr_avoidance"],
        "outcome": "aml_violation_confirmed", "rounds_used": 1,
    },
    {
        "task": "退休老人账户突然收到50万元转入，随即全额转出至境外，老人表示不知情",
        "final_answer": "reject:critical:0.97",
        "risk_level": "critical", "decision": "reject",
        "reasoning": "洗钱分层：利用老人账户中转，资金快速流转至境外，账户所有人不知情，疑似money mule。",
        "indicators": ["money_mule_suspected", "pass_through_account", "cross_border"],
        "outcome": "fraud_confirmed", "rounds_used": 2,
    },
    {
        "task": "电商商户申请提现30万，近30天平均日销售额5000元，本次约为60天收入",
        "final_answer": "challenge:medium:0.68",
        "risk_level": "medium", "decision": "challenge",
        "reasoning": "提现与销售不匹配：30万相当于60天收入，需要商户提供大额订单或促销证明。",
        "indicators": ["withdrawal_exceeds_sales_baseline", "amount_profile_mismatch"],
        "outcome": "legitimate_after_evidence", "rounds_used": 2,
    },
    {
        "task": "用户从VPN节点（出口IP美国）发起转账，账户注册地中国，转账目标境内账户，金额500元",
        "final_answer": "approve:low:0.78",
        "risk_level": "low", "decision": "approve",
        "reasoning": "VPN常见于合法用途，金额小，目标境内，无其他异常信号，低风险批准。",
        "indicators": ["vpn_detected"],
        "outcome": "legitimate_confirmed", "rounds_used": 1,
    },
]


LLM_GENERATION_PROMPT = """你是一个金融风险案例生成专家。请生成{count}个真实的金融风险评估案例。

要求：
1. 每个案例必须有明确的风险场景描述
2. 包含多样化风险等级：low(20%) / medium(30%) / high(35%) / critical(15%)
3. 场景多样：支付欺诈、信用风险、洗钱、账号盗用、合规违规等
4. 每个案例包含：scenario, risk_level, decision, confidence, reasoning, indicators, outcome

输出严格JSON数组，每个元素：
{{"scenario":"场景","risk_level":"low|medium|high|critical","decision":"approve|reject|challenge|review","confidence":0.85,"reasoning":"分析","indicators":["i1"],"outcome":"fraud_confirmed|legitimate_confirmed|aml_violation|requires_review"}}

只输出JSON数组："""


async def generate_with_llm(count: int, api_key: str, model: str) -> List[Dict]:
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        prompt = LLM_GENERATION_PROMPT.format(count=count)
        logger.info(f"Generating {count} cases with {model}...")
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000,
        )
        content = resp.choices[0].message.content or ""
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            cases = json.loads(content[start:end])
            logger.info(f"LLM generated {len(cases)} cases")
            return cases
        logger.warning("LLM output did not contain valid JSON array")
        return []
    except ImportError:
        logger.error("openai not installed: pip install openai")
        return []
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return []


async def store_cases(cases: List[Dict], dry_run: bool = False) -> int:
    if dry_run:
        logger.info(f"[DRY RUN] Would store {len(cases)} cases")
        for c in cases[:3]:
            task = c.get("task") or c.get("scenario", "")
            print(f"  [{c.get('risk_level','?')}] {task[:100]}")
        return len(cases)

    from aegean.memory.global_memory import GlobalMemorySystem
    memory = GlobalMemorySystem()
    count = 0
    for i, case in enumerate(cases):
        try:
            task = case.get("task") or case.get("scenario", f"Synthetic case {i}")
            risk_level = case.get("risk_level", "medium")
            decision = case.get("decision", "review")
            confidence = case.get("confidence", 0.8)
            final_answer = case.get("final_answer") or f"{decision}:{risk_level}:{confidence:.2f}"
            await memory.store_consensus_result(
                consensus_id=f"synthetic-{i:04d}-{int(datetime.now().timestamp())}",
                task=task,
                final_answer=final_answer,
                rounds_used=case.get("rounds_used", 1),
                consensus_reached=decision != "challenge",
                participating_agents=["synthetic_generator"],
                execution_time=0.0,
                metadata={
                    "source": "synthetic",
                    "risk_level": risk_level,
                    "decision": decision,
                    "indicators": case.get("indicators", []),
                    "reasoning": case.get("reasoning", ""),
                    "outcome": case.get("outcome", "unknown"),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to store case {i}: {e}")
    logger.info(f"Stored {count}/{len(cases)} synthetic cases")
    return count


async def main(llm_count: int = 30, dry_run: bool = False, model: str = "gpt-4o"):
    logger.info("=== Synthetic Risk Case Generator ===")
    all_cases = list(DETERMINISTIC_CASES)
    logger.info(f"Deterministic seed cases: {len(all_cases)}")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-your") and llm_count > 0:
        llm_cases = await generate_with_llm(llm_count, api_key, model)
        all_cases.extend(llm_cases)
        logger.info(f"Total: {len(all_cases)} ({len(DETERMINISTIC_CASES)} deterministic + {len(llm_cases)} LLM)")
    else:
        logger.info("No OPENAI_API_KEY — using deterministic cases only")

    count = await store_cases(all_cases, dry_run=dry_run)
    logger.info(f"Done. {count} synthetic cases ready for RAG retrieval.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic risk cases")
    parser.add_argument("--count", type=int, default=30, help="LLM-generated case count")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model")
    parser.add_argument("--dry-run", action="store_true", help="Print, don't store")
    args = parser.parse_args()
    asyncio.run(main(llm_count=args.count, dry_run=args.dry_run, model=args.model))
