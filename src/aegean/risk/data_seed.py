"""
Risk Knowledge Base Seeder.

Populates the GlobalMemorySystem knowledge base with publicly available
financial risk data to enable RAG-powered validator analysis from day one.

Data sources used (all public domain):
- FATF AML typologies
- OFAC/FinCEN public guidelines
- Common fraud patterns (public research)
- PBC (People's Bank of China) regulations
- Generic risk indicators from academic literature

Usage::
    from aegean.risk.data_seed import RiskKnowledgeSeeder
    seeder = RiskKnowledgeSeeder(memory_system)
    count = await seeder.seed_all()
    print(f"Seeded {count} documents")
"""

from typing import List, Tuple
import logging

from aegean.memory.global_memory import GlobalMemorySystem

logger = logging.getLogger(__name__)

# (content, category) pairs
SEED_DATA: List[Tuple[str, str]] = [
    # ── AML Regulations ───────────────────────────────────────────────────────
    (
        "FATF Recommendation 16 (Wire Transfers): Financial institutions must "
        "include originator and beneficiary information with wire transfers. "
        "For transfers above USD 1,000 threshold, full KYC information is required. "
        "Correspondent banks must verify this information and apply enhanced due "
        "diligence for transactions from high-risk jurisdictions.",
        "aml_regulations",
    ),
    (
        "US Bank Secrecy Act (BSA) - Currency Transaction Reports (CTR): "
        "Financial institutions must file CTRs for cash transactions exceeding "
        "USD 10,000 in a single business day. Multiple transactions by the same "
        "person totaling more than $10,000 must also be reported (aggregation rule). "
        "Structuring transactions to avoid CTR filing is a federal crime.",
        "aml_regulations",
    ),
    (
        "China AML Regulations - Large Transaction Reporting: "
        "Per PBOC regulations, RMB cash transactions above 50,000 CNY or "
        "foreign currency transactions above USD 10,000 must be reported. "
        "Cross-border transfers above USD 200,000 require enhanced documentation. "
        "Suspicious activity reports (SARs) must be filed within 5 business days.",
        "aml_regulations",
    ),
    (
        "FATF High-Risk Jurisdictions: Countries subject to FATF enhanced monitoring "
        "include those with strategic AML/CFT deficiencies. Transactions involving "
        "counterparties in these jurisdictions require enhanced due diligence (EDD). "
        "Currently monitored jurisdictions include: Myanmar, Iran, North Korea (call-for-action). "
        "OFAC maintains separate sanctions lists requiring screening for all transactions.",
        "aml_regulations",
    ),
    (
        "EU 6AMLD - Anti-Money Laundering Directive: Extends criminal liability for "
        "money laundering to legal persons. Predicate offenses include cybercrime, "
        "environmental crime, and tax crimes. Member states required to impose minimum "
        "4-year imprisonment. Enhanced beneficial ownership transparency requirements "
        "for companies and trusts.",
        "aml_regulations",
    ),
    # ── Fraud Patterns ────────────────────────────────────────────────────────
    (
        "Account Takeover (ATO) Fraud Indicators: Sudden change in device fingerprint "
        "or IP address, login from new geographic location, immediate high-value "
        "transaction after password change, multiple failed login attempts followed "
        "by success, changes to contact information before transaction, unusual "
        "transaction timing (3am-5am local time), first transaction to new payee "
        "is high value.",
        "fraud_patterns",
    ),
    (
        "Synthetic Identity Fraud: Fraudsters combine real and fictitious information "
        "to create a new identity. Common signals: SSN/ID number belongs to minor or "
        "deceased person, credit file 'thin' with sudden large credit applications, "
        "multiple accounts sharing the same address or phone number, "
        "no prior credit history followed by immediate large credit requests, "
        "inconsistencies between stated income and transaction behavior.",
        "fraud_patterns",
    ),
    (
        "Card-Not-Present (CNP) Fraud Patterns: High velocity of small test transactions "
        "followed by large purchase, billing and shipping address mismatch, "
        "multiple cards used from same device/IP, orders to freight forwarders, "
        "high-value electronics or gift cards, rush/overnight shipping for expensive items, "
        "multiple failed card attempts before success.",
        "fraud_patterns",
    ),
    (
        "Money Mule Detection: Third-party receives funds and immediately transfers out, "
        "account opened recently with no prior history, funds received in round numbers, "
        "multiple small incoming transfers aggregating to large outgoing, "
        "account holder unable to explain source of funds, "
        "transactions inconsistent with stated account purpose, "
        "IP/device located in different country than account registration.",
        "fraud_patterns",
    ),
    (
        "Business Email Compromise (BEC) Fraud: Fraudster impersonates executive or vendor "
        "to redirect payments. Key indicators: urgent wire transfer request via email only, "
        "slight variation in email domain (homograph attack), "
        "request to change existing vendor bank account details, "
        "transaction requested outside normal business hours, "
        "payee account recently opened, destination is foreign bank account.",
        "fraud_patterns",
    ),
    (
        "Velocity Fraud Patterns: Normal human behavior limits: a person cannot physically "
        "be in two cities more than 500km apart within 1 hour (impossible travel). "
        "Transaction velocity exceeding 10 transactions per hour from same account is anomalous. "
        "Round-trip transactions (send then receive same amount) within 24 hours suggest "
        "layering. Identical amounts sent to multiple recipients suggest automation.",
        "fraud_patterns",
    ),
    # ── Identity Verification ─────────────────────────────────────────────────
    (
        "KYC (Know Your Customer) Best Practices: Tier 1 - Basic identity (name, DOB, address). "
        "Tier 2 - Enhanced: government ID, proof of address, source of funds. "
        "Tier 3 - EDD (Enhanced Due Diligence): in-person verification, wealth source documentation. "
        "Risk-based approach: transaction limits increase with KYC tier. "
        "New accounts should have lower limits until activity history establishes trust baseline.",
        "identity_verification",
    ),
    (
        "Account Age Risk Scoring: Accounts less than 7 days old: high-risk for large transactions, "
        "limit to under $500. Accounts 7-30 days: medium risk, limit $2,000. "
        "Accounts 30-90 days with no flags: standard risk. "
        "Accounts 90+ days with positive history: reduced scrutiny. "
        "Trust score should incorporate: account age, transaction volume, complaint history, "
        "identity verification tier.",
        "identity_verification",
    ),
    (
        "Digital Identity Risk Signals: Device fingerprint changes indicate possible ATO. "
        "VPN/Tor/proxy usage elevates risk for financial transactions. "
        "Multiple accounts on same device suggest fraud ring. "
        "IP geolocation mismatch with stated residence is a moderate risk signal. "
        "Emulator detection (mobile): transactions from emulators are 3x more likely fraudulent. "
        "Biometric authentication failure after success suggests credential theft.",
        "identity_verification",
    ),
    # ── Risk Indicators ───────────────────────────────────────────────────────
    (
        "Transaction Risk Scoring Model - Key Features: "
        "(1) Amount: absolute value and deviation from user mean. "
        "(2) Velocity: transactions per hour/day vs baseline. "
        "(3) Counterparty: new vs known, risk country. "
        "(4) Time: hour of day, day of week anomaly. "
        "(5) Channel: API transactions higher risk than mobile app. "
        "(6) Geography: distance from home location. "
        "(7) Device: new device, emulator, VPN. "
        "Combined score using logistic regression or gradient boosting.",
        "risk_indicators",
    ),
    (
        "Layering in Money Laundering: The second stage of money laundering involves "
        "complex layers of financial transactions to disguise the audit trail. "
        "Common techniques: rapid movement through multiple accounts, "
        "currency conversion between multiple currencies, "
        "purchase of high-value assets (real estate, art, crypto), "
        "use of shell companies in multiple jurisdictions, "
        "loan-back schemes where criminal lends own laundered funds to self.",
        "risk_indicators",
    ),
    (
        "Structuring (Smurfing) Detection: Breaking large amounts into smaller transactions "
        "to avoid reporting thresholds. Red flags: multiple transactions just below $10,000 "
        "(e.g. $9,800, $9,500), same-day multiple deposits at different branches, "
        "transactions by multiple individuals to same account totaling above threshold, "
        "regular pattern of sub-threshold deposits. "
        "Structuring is illegal regardless of whether underlying funds are legitimate.",
        "risk_indicators",
    ),
    (
        "Crypto Asset Risk Indicators: Transactions to/from mixing services (tumblers). "
        "Interaction with addresses flagged by Chainalysis/Elliptic as darknet markets. "
        "Rapid conversion between multiple cryptocurrencies. "
        "Immediate withdrawal of crypto after fiat deposit (same-day). "
        "Peer-to-peer exchange usage to avoid regulated exchanges. "
        "Transaction amounts corresponding to known ransomware payment patterns.",
        "risk_indicators",
    ),
]


class RiskKnowledgeSeeder:
    """
    Seeds the GlobalMemorySystem knowledge base with public-domain
    financial risk knowledge to bootstrap RAG-powered risk validators.
    """

    def __init__(self, memory_system: GlobalMemorySystem):
        self.memory_system = memory_system

    async def seed_all(self, skip_if_exists: bool = True) -> int:
        """
        Seed all risk knowledge documents.

        Args:
            skip_if_exists: If True, skip seeding if KB already has documents.

        Returns:
            Number of documents seeded
        """
        if skip_if_exists:
            stats = self.memory_system.knowledge_base.get_stats()
            if stats.get("total_documents", 0) > 0:
                logger.info(
                    f"Knowledge base already has {stats['total_documents']} documents, "
                    f"skipping seed."
                )
                return 0

        count = 0
        for content, category in SEED_DATA:
            try:
                await self.memory_system.add_knowledge(
                    content=content,
                    category=category,
                    metadata={"source": "public_domain_seed", "seeded": True},
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to seed document [{category}]: {e}")

        logger.info(f"Seeded {count}/{len(SEED_DATA)} risk knowledge documents")
        return count

    async def seed_category(self, category: str) -> int:
        """
        Seed only documents belonging to a specific category.

        Args:
            category: One of 'aml_regulations', 'fraud_patterns',
                      'identity_verification', 'risk_indicators'

        Returns:
            Number of documents seeded
        """
        count = 0
        for content, cat in SEED_DATA:
            if cat == category:
                try:
                    await self.memory_system.add_knowledge(
                        content=content,
                        category=cat,
                        metadata={"source": "public_domain_seed", "seeded": True},
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to seed [{cat}]: {e}")
        logger.info(f"Seeded {count} documents for category '{category}'")
        return count

    @staticmethod
    def list_categories() -> list:
        """Return all available seed data categories."""
        return list({cat for _, cat in SEED_DATA})

