#!/usr/bin/env python3
"""
Script C: Import Kaggle fraud dataset features into KnowledgeBase.

How to get the data:
  Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  IEEE-CIS:    https://www.kaggle.com/competitions/ieee-fraud-detection

Usage:
    python scripts/import_kaggle.py --file data/creditcard.csv
    python scripts/import_kaggle.py --file data/train_transaction.csv --type ieee
    python scripts/import_kaggle.py --file data/creditcard.csv --dry-run
"""

import asyncio
import argparse
import sys
import os
import logging
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("import_kaggle")


def analyze_creditcard_csv(filepath: str) -> List[Dict]:
    """
    Analyze Kaggle Credit Card Fraud dataset.
    Extracts statistical patterns as knowledge documents.
    """
    import csv

    fraud_amounts = []
    legit_amounts = []
    total_count = 0

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                amount = float(row.get("Amount", 0))
                is_fraud = int(row.get("Class", 0)) == 1
                if is_fraud:
                    fraud_amounts.append(amount)
                else:
                    legit_amounts.append(amount)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return []

    if not fraud_amounts:
        logger.warning("No fraud cases found in dataset")
        return []

    fraud_count = len(fraud_amounts)
    fraud_rate = fraud_count / total_count
    avg_fraud_amt = sum(fraud_amounts) / fraud_count
    avg_legit_amt = sum(legit_amounts) / len(legit_amounts) if legit_amounts else 0
    max_fraud = max(fraud_amounts)
    min_fraud = min(fraud_amounts)

    buckets = {"0-10": 0, "10-100": 0, "100-1000": 0, "1000+": 0}
    for a in fraud_amounts:
        if a < 10:
            buckets["0-10"] += 1
        elif a < 100:
            buckets["10-100"] += 1
        elif a < 1000:
            buckets["100-1000"] += 1
        else:
            buckets["1000+"] += 1

    bucket_pct = {k: f"{v / fraud_count * 100:.1f}%" for k, v in buckets.items()}

    return [
        {
            "content": (
                f"Credit Card Fraud Statistical Analysis (Kaggle, {total_count:,} transactions):\n"
                f"Fraud rate: {fraud_rate:.3%} ({fraud_count:,} fraud / {total_count:,} total)\n"
                f"Fraud amount: avg=${avg_fraud_amt:.2f}, min=${min_fraud:.2f}, max=${max_fraud:.2f}\n"
                f"Legitimate amount avg: ${avg_legit_amt:.2f}\n"
                f"Fraud by amount range: {bucket_pct}\n"
                f"Insight: {bucket_pct.get('0-10','?')} of fraud is under $10 "
                f"(test transactions to verify card validity)."
            ),
            "category": "risk_indicators",
            "metadata": {"source": "kaggle_creditcard", "doc_type": "statistics"},
        },
        {
            "content": (
                "Credit Card Fraud Patterns from Kaggle Dataset:\n"
                "1. Small test transactions: Fraudsters often start with tiny amounts ($0.01-$10) "
                "to verify card validity before larger fraud.\n"
                f"2. Amount distribution across fraud cases: {bucket_pct}\n"
                "3. True fraud rate ~0.17%: 99.83% of transactions are legitimate. "
                "High precision is critical to avoid false positives.\n"
                "4. Amount alone is insufficient: Fraud occurs at all amount ranges. "
                "Behavioral and contextual signals are more predictive.\n"
                "5. Velocity matters: Multiple rapid small transactions often precede large fraud."
            ),
            "category": "fraud_patterns",
            "metadata": {"source": "kaggle_creditcard", "doc_type": "patterns"},
        },
        {
            "content": (
                "Risk Assessment Guidelines from Kaggle Fraud Analysis:\n"
                "LOW risk: Consistent spending patterns, known merchant, amount within history.\n"
                "MEDIUM risk: Slightly elevated amount, new merchant category, minor velocity spike.\n"
                "HIGH risk: Amount far above average, multiple rapid transactions, new geography.\n"
                "CRITICAL risk: Test transaction + immediate large transaction, "
                "impossible travel, known fraud merchant, OFAC-sanctioned entity."
            ),
            "category": "risk_indicators",
            "metadata": {"source": "kaggle_creditcard", "doc_type": "guidelines"},
        },
    ]


def analyze_ieee_csv(filepath: str) -> List[Dict]:
    """
    Analyze IEEE-CIS Fraud Detection dataset.
    Samples first 100k rows and extracts key fraud features.
    """
    import csv

    fraud_product_types: Dict[str, int] = {}
    total = 0
    fraud_total = 0

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                is_fraud = int(row.get("isFraud", 0)) == 1
                if is_fraud:
                    fraud_total += 1
                    prod = row.get("ProductCD", "unknown")
                    fraud_product_types[prod] = fraud_product_types.get(prod, 0) + 1
                if total >= 100000:
                    break
    except Exception as e:
        logger.error(f"Failed to read IEEE CSV: {e}")
        return []

    if total == 0:
        return []

    top_products = sorted(fraud_product_types.items(), key=lambda x: -x[1])[:5]

    return [
        {
            "content": (
                f"IEEE-CIS Fraud Detection Dataset Analysis ({total:,} transactions sampled):\n"
                f"Fraud rate: {fraud_total / total:.3%}\n"
                f"Top fraud product categories: {top_products}\n"
                "Key predictive features:\n"
                "- TransactionAmt: distribution differs between fraud and legitimate\n"
                "- ProductCD: W (web) transactions have higher fraud rates\n"
                "- addr1/addr2: billing address mismatch signals fraud\n"
                "- D-features: time deltas between transactions (velocity signals)\n"
                "- V-features: Vesta-engineered behavioral features (most predictive)\n"
                "- card1-6: card metadata helps identify fraud clusters"
            ),
            "category": "fraud_patterns",
            "metadata": {"source": "kaggle_ieee_cis", "doc_type": "analysis"},
        }
    ]


async def store_docs(docs: List[Dict], dry_run: bool = False) -> int:
    if dry_run:
        logger.info(f"[DRY RUN] Would store {len(docs)} documents")
        for d in docs:
            print(f"  [{d['category']}] {d['content'][:120]}...")
        return len(docs)

    from aegean.memory.global_memory import GlobalMemorySystem
    memory = GlobalMemorySystem()
    count = 0
    for doc in docs:
        try:
            await memory.add_knowledge(
                content=doc["content"],
                category=doc["category"],
                metadata=doc["metadata"],
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to store: {e}")
    logger.info(f"Stored {count}/{len(docs)} documents")
    return count


async def main(filepath: str, dataset_type: str = "creditcard", dry_run: bool = False):
    logger.info(f"=== Kaggle Dataset Importer: {filepath} ===")

    if not os.path.exists(filepath):
        logger.error(
            f"File not found: {filepath}\n"
            "Download from:\n"
            "  Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "  IEEE-CIS:    https://www.kaggle.com/competitions/ieee-fraud-detection"
        )
        return 0

    docs = analyze_ieee_csv(filepath) if dataset_type == "ieee" else analyze_creditcard_csv(filepath)
    logger.info(f"Generated {len(docs)} knowledge documents")
    count = await store_docs(docs, dry_run=dry_run)
    logger.info(f"Done. {count} documents ready for RAG retrieval.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle fraud dataset into knowledge base")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--type", choices=["creditcard", "ieee"], default="creditcard",
                        help="Dataset type")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't store")
    args = parser.parse_args()
    asyncio.run(main(filepath=args.file, dataset_type=args.type, dry_run=args.dry_run))
