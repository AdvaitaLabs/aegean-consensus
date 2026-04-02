# Aegean Consensus

<div align="center">

**A Byzantine Fault-Tolerant Multi-Agent Consensus Platform with Financial Risk Assessment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/arXiv-2512.20184-b31b1b.svg)](https://arxiv.org/abs/2512.20184)

*Reaching Agreement Among Reasoning LLM Agents — with Formal Guarantees and Financial-Grade Risk Controls*

[Quick Start](#quick-start) • [Core Protocol](#core-consensus-protocol) • [Risk Assessment](#financial-risk-assessment) • [API Reference](#api-reference) • [Architecture](#system-architecture)

</div>

---

## Overview

Aegean is a production-ready implementation of the consensus protocol from *"Reaching Agreement Among Reasoning LLM Agents"* (arXiv:2512.20184), extended with:

- **Group Chat System** — multi-agent collaboration with weighted voting
- **Global Memory** — RAG-powered knowledge base + experience accumulation  
- **Financial Risk Assessment** — VAN (Verification Agent Network) for institution-grade risk evaluation

### Why Aegean?

| Problem | Traditional Approach | Aegean Solution | Gain |
|---------|---------------------|-----------------|------|
| Fixed rounds | Predetermined iteration limit | Adaptive termination via stability horizon | 1.2–20× faster |
| Barrier sync | Wait for slowest agent | Early termination after quorum | Latency decoupled |
| Homogeneous agents | All agents equal | Weighted voting by capability | More accurate |
| No domain memory | Stateless | RAG + ExperienceBase | Continuous learning |
| No risk controls | Ad-hoc | VAN multi-validator consensus | Institution-grade |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│              REST API  /  Python SDK  /  Web UI                 │
└────────────┬──────────────────────────────┬────────────────────┘
             │                              │
┌────────────▼──────────┐    ┌─────────────▼──────────────────────┐
│   Group Chat API       │    │     Risk Assessment API             │
│  /api/v1/groups/*      │    │     /api/v1/risk/*                  │
└────────────┬──────────┘    └─────────────┬──────────────────────┘
             │                              │
┌────────────▼──────────────────────────────▼────────────────────┐
│                   Aegean Consensus Engine                       │
│  ┌─────────────────────────┐  ┌────────────────────────────┐   │
│  │  ConsensusCoordinator   │  │  RiskConsensusCoordinator  │   │
│  │  • Leader Election      │  │  • Sequencer routing       │   │
│  │  • Quorum Detection     │  │  • Parallel validators     │   │
│  │  • Stability Horizon    │  │  • Weighted aggregation    │   │
│  │  • Early Termination    │  │  • Challenge-Response      │   │
│  └────────────┬────────────┘  └────────────┬───────────────┘   │
│               │                            │                    │
│  ┌────────────▼────────────────────────────▼───────────────┐   │
│  │            WeightedDecisionEngine                       │   │
│  │   weight = capability_weight × confidence × accuracy    │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│                    Global Memory System                         │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │    KnowledgeBase     │    │       ExperienceBase         │  │
│  │  • Vector embeddings │    │  • Consensus history         │  │
│  │  • RAG retrieval     │    │  • Agent performance         │  │
│  │  • Multi-backend     │    │  • Feedback learning         │  │
│  │  (mem/Milvus/Pine)   │    │  (mem/TimescaleDB/PG)        │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Consensus Protocol

Based on Algorithm 1 from arXiv:2512.20184.

### How It Works

```
Task Input
    │
    ▼
┌─────────────────┐
│ Leader Election  │  Select coordinator agent
└────────┬────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Collect Initial Solutions (parallel)    │
    │  asyncio.gather → cancel slow agents     │  Early Termination
    │  after ⌈N/2⌉ responses                  │  (Section 6)
    └────┬────────────────────────────────────┘
         │
    ┌────▼────────────────────────┐
    │  Quorum Check               │
    │  ≥ ⌈N/2⌉ agents agree?     │
    └────┬──────────┬─────────────┘
       YES          NO
         │          │
    ┌────▼──┐  ┌────▼────────────┐
    │Stable?│  │ Refinement Round │ ◄─ loop up to max_rounds
    │β rnds │  │ agents refine    │
    └────┬──┘  └────┬────────────┘
       YES          │
         │     back to Quorum Check
    ┌────▼────────────────┐
    │  Consensus Reached   │
    │  Return final answer │
    └─────────────────────┘
```

### Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Quorum size | α | ⌈N/2⌉ | Min agents to agree |
| Stability horizon | β | 2 | Consecutive stable rounds needed |
| Max rounds | — | 5 | Upper bound on refinement |
| Early termination | — | enabled | Cancel slow agents after quorum |

### Weighted Decision Engine

Solves the **capability heterogeneity** problem — a domain expert should outweigh a generalist:

```
vote_weight = capability_weight × confidence × historical_accuracy
```

Agents with higher domain proficiency (`specialization` map) and better historical accuracy accumulate more voting power over time.

---

## Group Chat System

Multi-agent collaboration with three modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `consensus` | All agents answer the same question, vote | Risk assessment, fact-checking |
| `collaboration` | Agents work on different subtasks | Complex pipelines |
| `hybrid` | Mix of both | General-purpose |

### Quick Example

```python
from aegean.services.group_chat_service import GroupChatService
from aegean.core.agent import AgentRegistry

service = GroupChatService(agent_registry=AgentRegistry())

# Create group
group = service.create_group("Risk Team", created_by="user_1", mode="consensus")

# Add agents with specializations
service.add_member(group.group_id, "agent_credit",
    capability_weight=0.9,
    specialization={"credit": 0.95, "fraud": 0.7})

# Execute consensus
result = service.execute_consensus(group.group_id, "Assess credit risk for customer X")
print(result.final_solution.answer)
```

---

## Global Memory System

RAG-powered memory combining static knowledge and dynamic experience:

```
                 Query / Task
                      │
          ┌───────────▼───────────┐
          │   GlobalMemorySystem  │
          └───────────┬───────────┘
               ┌──────┴──────┐
               │             │
    ┌──────────▼──┐   ┌──────▼──────────┐
    │KnowledgeBase│   │ ExperienceBase   │
    │             │   │                  │
    │ Regulations │   │ Past decisions   │
    │ Fraud rules │   │ Agent accuracy   │
    │ Best practs │   │ Feedback loop    │
    └──────────┬──┘   └──────┬──────────┘
               │             │
          ┌────▼─────────────▼────┐
          │   MemoryContext        │
          │ + PromptEnhancer       │
          └────────────┬──────────┘
                       │
              Enriched Prompt → LLM
```

### Backends

| Component | Development | Production |
|-----------|-------------|------------|
| KnowledgeBase | In-memory | Milvus / Pinecone |
| ExperienceBase | In-memory | TimescaleDB / PostgreSQL |

### Prompt Templates

Pre-built templates for domain-specific tasks:
- `reasoning` — general multi-agent reasoning
- `credit_assessment` — credit risk scoring (AAA–B)
- `fraud_detection` — transaction fraud analysis
- `consensus_refinement` — peer-solution refinement
- `risk_identity` / `risk_anomaly` / `risk_compliance` / `risk_amount` / `risk_context` — VAN validator prompts

---

## Financial Risk Assessment

Aegean implements a **multi-validator consensus-based risk evaluation pipeline** using a VAN (Verification Agent Network) architecture. Every agent-initiated financial action passes through a committee of specialist AI validators before execution.

### Architecture Overview

```
                        RiskRequest
                  (subject + context + trace)
                             │
              ┌──────────────▼──────────────┐
              │          Sequencer           │
              │   Score signals → Route to   │
              │   SIMPLE / MEDIUM / HARD tier │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │    SIMPLE         │  MEDIUM            │  HARD
         │  [Amount,Identity]│  [+Anomaly]        │  [All 5]
         │                   │                    │
┌────────▼───────────────────▼────────────────────▼────────┐
│              Validator Committee (parallel)               │
│                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐   │
│  │  Identity   │ │   Anomaly   │ │   Compliance     │   │
│  │  Validator  │ │  Validator  │ │   Validator      │   │
│  │  (KYA/KYC)  │ │  (Velocity) │ │  (AML/CTR/FATF)  │   │
│  │  w = 0.90   │ │  w = 0.85   │ │  w = 0.95        │   │
│  └──────┬──────┘ └──────┬──────┘ └────────┬─────────┘   │
│         │               │                 │              │
│  ┌──────▼──────┐ ┌──────▼──────┐          │              │
│  │   Amount    │ │   Context   │          │              │
│  │  Validator  │ │  Validator  │          │              │
│  │ (Velocity)  │ │ (Trace/RAG) │          │              │
│  │  w = 0.88   │ │  w = 0.80   │          │              │
│  └──────┬──────┘ └──────┬──────┘          │              │
└─────────┼───────────────┼─────────────────┼──────────────┘
          │               │                 │
          └───────────────┼─────────────────┘
                          │  ValidatorResult[]
              ┌───────────▼────────────┐
              │  RiskConsensusCoord.   │
              │  WeightedDecisionEngine│
              │  weight × confidence   │
              └───────────┬────────────┘
                          │
             ┌────────────┼─────────────┐
             │            │             │
        ┌────▼───┐  ┌─────▼────┐  ┌────▼──────┐
        │APPROVE │  │ REJECT   │  │ CHALLENGE │
        │low risk│  │high risk │  │uncertain  │
        └────────┘  └──────────┘  └─────┬─────┘
                                         │
                               ┌─────────▼──────────┐
                               │  ChallengeManager   │
                               │  Issue challenge    │
                               │  + required evidence│
                               └─────────┬──────────┘
                                         │ Caller submits evidence
                                         │
                               ┌─────────▼──────────┐
                               │   Re-evaluation     │
                               │  (inject evidence   │
                               │   into trace_ctx)   │
                               └─────────────────────┘
```

### Swim-Lane Flow

```
Caller          Sequencer        Validators(×N)    Coordinator     Session/Challenge
  │                │                   │                │                 │
  │──evaluate()───▶│                   │                │                 │
  │                │──classify()──────▶│                │                 │
  │                │◀──difficulty+cfg──│                │                 │
  │                │                   │                │──create_session─▶│
  │                │──dispatch─────────▶                │                 │
  │                │            parallel│asyncio.gather  │                 │
  │                │            pre_screen()             │                 │
  │                │            retrieve_context(RAG)    │                 │
  │                │            analyze_with_llm()       │                 │
  │                │◀───────────ValidatorResult[]────────│                 │
  │                │                   │──aggregate()──▶│                 │
  │                │                   │  weighted vote  │                 │
  │                │                   │◀──RiskDecision──│                 │
  │                │                   │                │──attach_decision▶│
  │                │                   │                │──persist(RAG)───▶│
  │◀──RiskDecision─│                   │                │                 │
  │                │                   │                │                 │
  │  [if CHALLENGE]│                   │                │                 │
  │──submit_evidence────────────────────────────────────────────────────▶│
  │                │                   │                │◀──challenge_ctx──│
  │──re_evaluate()─▶  [inject evidence into trace_context, repeat flow]   │
  │◀──new decision──│                   │                │                 │
```

### Validator Committees

Each validator runs a **three-stage pipeline**:

```
RiskRequest
     │
┌────▼──────────────────────────────────────────┐
│  Stage 1: Pre-screen (< 5ms, no LLM)          │
│  Deterministic rules → high-confidence signal? │
│  YES → return immediately (skip LLM)           │
│  NO  → continue                                │
└────┬──────────────────────────────────────────┘
     │
┌────▼──────────────────────────────────────────┐
│  Stage 2: RAG Context Retrieval               │
│  GlobalMemorySystem.retrieve_context()         │
│  → Knowledge docs + similar historical cases   │
└────┬──────────────────────────────────────────┘
     │
┌────▼──────────────────────────────────────────┐
│  Stage 3: LLM Deep Analysis                   │
│  Enriched prompt = RAG context + request data  │
│  Structured output: risk_level / confidence /  │
│  risk_indicators / reasoning                   │
└────┬──────────────────────────────────────────┘
     │
 ValidatorResult (risk_level, confidence, weight, reasoning)
```

| Validator | Domain | Base Weight | Pre-screen Rules |
|-----------|--------|-------------|------------------|
| `IdentityValidator` | KYA/KYC — trust score, account age, flag history | 0.90 | trust_score < 0.1 → CRITICAL instantly |
| `AnomalyValidator` | Velocity, geo, OFAC regions | 0.85 | Sanctioned region (KP/IR/SY) → CRITICAL instantly |
| `ComplianceValidator` | AML, CTR thresholds, structuring | 0.95 | Amount ≥ CTR threshold → HIGH + report required |
| `AmountValidator` | Single/hourly limits, round numbers | 0.88 | Exceeds hard limit → CRITICAL instantly |
| `ContextValidator` | Reasoning trace, prompt injection | 0.80 | Injection keywords detected → CRITICAL instantly |

### Sequencer Routing

```
Incoming RiskRequest
        │
    Score signals:
    • Amount ≥ $50k  → +3
    • Amount ≥ $10k  → +2
    • Trust score < 0.3 → +3
    • Flag count ≥ 3 → +2
    • Velocity ≥ 10/hr → +2
    • Missing trace (high-value) → +1
    • Cross-border → +1
    • priority=urgent → HARD override
        │
   Score 0–2       Score 3–5        Score 6+
      │                │                │
┌─────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
│   SIMPLE   │  │   MEDIUM    │  │    HARD    │
│ 2 validators│  │ 3 validators│  │ 5 validators│
│ 1 round    │  │ 2 rounds    │  │ 3 rounds   │
│ q=0.50     │  │ q=0.55      │  │ q=0.60     │
└────────────┘  └─────────────┘  └────────────┘
```

### Decision Framework

| Risk Level | Confidence | Decision | TTL |
|------------|------------|----------|-----|
| CRITICAL | any | REJECT | 5 min |
| HIGH | ≥ 0.70 | REJECT | 5 min |
| HIGH | < 0.70 | CHALLENGE | 5 min |
| MEDIUM (HARD tier) | < 0.55 | CHALLENGE | 1 hr |
| MEDIUM | any | REVIEW | 1 hr |
| LOW | any | APPROVE | 2 hr |

### Challenge-Response Flow

When a decision is `CHALLENGE`, the system pauses and requests additional evidence:

```python
# 1. Evaluate → get CHALLENGE decision
result = await coordinator.evaluate(request)
# result.decision = "challenge"
# result.challenge_id = "chal-abc123"
# result.required_evidence = ["purpose_proof", "identity_proof"]
# result.challenge_instructions = "Please provide..."

# 2. Submit evidence via API
POST /api/v1/risk/challenge/chal-abc123/respond
{
  "evidence_type": "purpose_proof",
  "evidence_content": "Invoice #INV-2024-001 for supplier payment",
  "submitted_by": "user_12345"
}

# 3. System injects evidence into trace_context and re-evaluates
# Returns new RiskDecision (approve/reject/challenge)
```

### Data Strategy

The risk system uses a three-tier data strategy:

```
Tier 1: Public Domain (day 1)              Tier 2: Accumulated (grows over time)
┌────────────────────────┐                 ┌─────────────────────────────┐
│ • FATF AML typologies  │                 │ • Every evaluation stored   │
│ • OFAC/FinCEN rules    │  ──seed──▶      │   in ExperienceBase         │
│ • CTR thresholds       │  KnowledgeBase  │ • User feedback updates     │
│ • Fraud pattern docs   │                 │   capability_weight         │
│ • 18 seed documents    │                 │ • RAG improves with volume  │
└────────────────────────┘                 └─────────────────────────────┘

Tier 3: External APIs (commercial stage)
┌──────────────────────────────────────────────┐
│ • OFAC SDN List (free API)                   │
│ • MaxMind GeoIP + fraud score                │
│ • 百行征信 / 芝麻信用 (commercial contract) │
│ • Chainalysis crypto risk scores             │
└──────────────────────────────────────────────┘
```

---

## API Reference

### Base URL

```
http://localhost:8000
```

Interactive docs available at `/docs` (Swagger UI) and `/redoc`.

---

### Group Chat API

#### Groups

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/groups` | Create a new agent group |
| `GET` | `/api/v1/groups/{group_id}` | Get group details |
| `GET` | `/api/v1/groups` | List all groups |
| `DELETE` | `/api/v1/groups/{group_id}` | Delete a group |

**Create Group**
```http
POST /api/v1/groups
Content-Type: application/json

{
  "group_name": "Financial Risk Team",
  "description": "Credit and fraud assessment group",
  "mode": "consensus",
  "created_by": "user_123"
}
```

```json
{
  "group_id": "group-a1b2c3d4",
  "group_name": "Financial Risk Team",
  "mode": "consensus",
  "created_by": "user_123",
  "created_at": "2026-03-18T10:00:00Z"
}
```

#### Members

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/groups/{id}/members` | Add agent to group |
| `DELETE` | `/api/v1/groups/{id}/members/{agent_id}` | Remove agent |
| `GET` | `/api/v1/groups/{id}/members` | List members |

**Add Member**
```http
POST /api/v1/groups/group-a1b2c3d4/members

{
  "agent_id": "agent_credit_analyst",
  "role": "credit_analyst",
  "capability_weight": 0.9,
  "specialization": {"credit": 0.95, "fraud": 0.75}
}
```

#### Messages

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/groups/{id}/messages` | Send message to group |
| `GET` | `/api/v1/groups/{id}/messages` | Get message history |

#### Consensus

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/groups/{id}/consensus` | Execute group consensus |
| `GET` | `/api/v1/groups/{id}/consensus/history` | Get consensus history |
| `GET` | `/api/v1/groups/consensus/{consensus_id}` | Get specific result |

**Execute Consensus**
```http
POST /api/v1/groups/group-a1b2c3d4/consensus

{
  "task": "Assess credit risk for customer with annual income $80k, credit score 720",
  "quorum_threshold": 0.6,
  "max_rounds": 3
}
```

```json
{
  "consensus_id": "consensus-x1y2z3",
  "group_id": "group-a1b2c3d4",
  "success": true,
  "final_solution": {
    "answer": "BBB - Medium credit risk, recommend approval with conditions",
    "confidence": 0.87
  },
  "weighted_votes": {"BBB": 1.75, "A": 0.62},
  "rounds_used": 2,
  "consensus_reached": true,
  "execution_time": 3.4
}
```

---

### Investment Analysis API (V1)

> V1 scope is intentionally limited to:
> - asset_type: `equity` / `etf` / `index`
> - market: `CN` / `HK` / `US`
>
> Requests using `fund` / `convertible_bond` (V2) or `futures` / `options` / `crypto` (V3) will return `400`.

#### `POST /api/v1/investment/analyze`

Run non-streaming investment analysis.

```http
POST /api/v1/investment/analyze
Content-Type: application/json

{
  "mode": "auto",
  "asset": {
    "symbol": "AAPL",
    "market": "US",
    "asset_type": "equity"
  },
  "timeframe": {
    "analysis_date": "2026-04-03T00:00:00Z",
    "lookback_window_days": 90,
    "horizon": "1m"
  },
  "risk_profile": "balanced",
  "objective": "balanced",
  "public_facts": [
    "Revenue growth remains stable",
    "Valuation above 5-year median"
  ],
  "custom_question": "Should I increase position this month?",
  "user_id": "user_123"
}
```

```json
{
  "request_id": "inv-abc123def456",
  "mode": "auto",
  "asset": {"symbol": "AAPL", "market": "US", "asset_type": "equity"},
  "recommendation": {
    "action": "hold",
    "confidence": 0.67,
    "position_suggestion": {
      "target_exposure_pct": 0.05,
      "max_drawdown_guard_pct": 0.08
    }
  },
  "risk_gate": {
    "status": "pass",
    "risk_level": "low",
    "risk_indicators": []
  },
  "consensus": {
    "enabled": false,
    "rounds_used": 0,
    "consensus_reached": false,
    "weighted_votes": {}
  }
}
```

Mode behavior:
- `fast`: single-agent fast response
- `auto`: multi-agent default; **does not auto-upgrade to roundtable**
- `collaborate`: more agents, still no consensus rounds
- `roundtable`: explicit consensus path enabled

Optional `constraints` contract (deterministic gate):
- `allowed_actions`: string array, e.g. `["hold", "buy"]`
- `no_short`: boolean, if true then `sell` is forced to `hold`
- `max_exposure_pct`: number, upper cap for `position_suggestion.target_exposure_pct`
- `max_drawdown_guard_pct`: number, overwrite `position_suggestion.max_drawdown_guard_pct`

Built-in deterministic exposure caps (always applied before `max_exposure_pct`):
- `risk_profile` cap:
  - `conservative` -> `0.05`
  - `balanced` -> `0.15`
  - `aggressive` -> `0.30`
- `objective` cap:
  - `defensive` -> `0.08`
  - `income` -> `0.10`
  - `balanced` -> `0.15`
  - `alpha` -> `0.30`

Final exposure is capped by the most conservative bound among model output, `risk_profile`, `objective`, and optional `constraints.max_exposure_pct`.

Example:

```json
{
  "constraints": {
    "allowed_actions": ["hold", "buy"],
    "no_short": true,
    "max_exposure_pct": 0.03,
    "max_drawdown_guard_pct": 0.02
  }
}
```

#### `POST /api/v1/investment/analyze/stream`

Run streaming analysis via SSE (`text/event-stream`).

Typical event sequence:
- `analysis_started`
- `request_validated`
- `agents_selected`
- `agent_completed` (repeated)
- `roundtable_started` / `roundtable_finished` (roundtable only)
- `constraints_applied`
- `recommendation_ready`
- `risk_gate_finished`
- `result`
- `end`

Example:

```bash
curl -N -X POST "http://localhost:8000/api/v1/investment/analyze/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "auto",
    "asset": {"symbol": "AAPL", "market": "US", "asset_type": "equity"},
    "timeframe": {"analysis_date": "2026-04-03T00:00:00Z", "lookback_window_days": 90, "horizon": "1m"},
    "risk_profile": "balanced",
    "objective": "balanced",
    "public_facts": ["Revenue growth remains stable"],
    "user_id": "user_123"
  }'
```

---

### Risk Assessment API

#### `POST /api/v1/risk/evaluate`

Run a full VAN pipeline risk evaluation.

**Request**
```http
POST /api/v1/risk/evaluate
Content-Type: application/json

{
  "subject_id": "user_12345",
  "subject_type": "user",
  "trust_score": 0.75,
  "total_transactions": 42,
  "flagged_count": 0,
  "jurisdiction": "US",

  "action_type": "payment",
  "description": "Transfer $3000 to supplier account",
  "amount": 3000.00,
  "currency": "USD",
  "counterparty_id": "vendor_xyz",
  "geo_location": "NY,US",
  "channel": "web",
  "trace_context": "Agent reasoning: user requested payment for invoice INV-001",
  "recent_transaction_count": 2,
  "recent_transaction_amount": 500.0,
  "priority": "normal",
  "debug_mode": false
}
```

**Response — Approved**
```json
{
  "decision_id": "dec-abc123def456",
  "request_id": "req-789xyz",
  "session_id": "sess-session123",
  "decision": "approve",
  "risk_level": "low",
  "confidence": 0.91,
  "ttl": 7200,
  "rationale": "[identity] Normal trust profile | [amount] Within limits | [anomaly] No anomalies",
  "risk_indicators": [],
  "challenge_eligible": false,
  "difficulty_level": "simple",
  "participating_validators": ["amount", "identity"],
  "execution_time": 0.34,
  "timestamp": "2026-03-18T10:05:00Z"
}
```

**Response — Challenge**
```json
{
  "decision": "challenge",
  "risk_level": "high",
  "confidence": 0.61,
  "challenge_eligible": true,
  "challenge_id": "chal-xyz789",
  "challenge_instructions": "Your request has been flagged...\nPlease provide: Purpose Proof, Business Justification",
  "required_evidence": ["purpose_proof", "business_justification"],
  "risk_indicators": ["cross_border_high_value", "potential_structuring_8500_vs_10000"]
}
```

#### `POST /api/v1/risk/challenge/{challenge_id}/respond`

Submit evidence to resolve a challenge and trigger re-evaluation.

```http
POST /api/v1/risk/challenge/chal-xyz789/respond

{
  "evidence_type": "purpose_proof",
  "evidence_content": "Payment for invoice #INV-2024-001, PO number PO-8823, verified by finance dept",
  "submitted_by": "user_12345"
}
```

Returns a new `RiskDecisionResponse` after re-evaluation.

Evidence types: `purpose_proof` · `identity_proof` · `authorization` · `transaction_log` · `business_justification` · `other`

#### `GET /api/v1/risk/sessions/{session_id}`

Get full session details including all decisions in the challenge lifecycle.

```json
{
  "session_id": "sess-session123",
  "subject_id": "user_12345",
  "status": "completed",
  "created_at": "2026-03-18T10:00:00Z",
  "expires_at": "2026-03-19T10:00:00Z",
  "challenge_count": 1,
  "decision_count": 2,
  "decisions": [
    {"decision_id": "dec-001", "decision": "challenge", "risk_level": "high", "confidence": 0.61},
    {"decision_id": "dec-002", "decision": "approve",   "risk_level": "low",  "confidence": 0.88}
  ]
}
```

#### `GET /api/v1/risk/sessions`

List sessions with optional filters: `?subject_id=user_12345&status=completed&limit=20`

#### `GET /api/v1/risk/stats`

Validator performance and session statistics.

```json
{
  "validators": [
    {"validator_id": "identity-v1", "capability_weight": 0.90, "total_evaluations": 142, "accuracy": 0.93},
    {"validator_id": "compliance-v1", "capability_weight": 0.95, "total_evaluations": 89, "accuracy": 0.97}
  ],
  "sessions": {
    "total_sessions": 234,
    "by_status": {"completed": 198, "challenged": 29, "active": 7},
    "challenged_sessions": 29
  }
}
```

#### `POST /api/v1/risk/seed`

Seed knowledge base with public-domain financial risk data (runs in background).

```http
POST /api/v1/risk/seed?force=false
```

```json
{"status": "seeding_started", "message": "Knowledge base seeding in background"}
```

Seeds 18 documents across: `aml_regulations` · `fraud_patterns` · `identity_verification` · `risk_indicators`

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/aegean-consensus.git
cd aegean-consensus

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

### Run the API Server

```bash
uvicorn aegean.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000/docs` for the interactive API explorer.

### Basic Consensus

```python
import asyncio
from aegean.core.coordinator import ConsensusCoordinator
from aegean.core.agent import AgentRegistry

async def main():
    registry = AgentRegistry()
    coordinator = ConsensusCoordinator(agent_registry=registry)
    result = await coordinator.run_consensus(task="What is 2+2?")
    print(result.final_solution.answer)

asyncio.run(main())
```

### Risk Evaluation (no LLM required for pre-screen)

```python
import asyncio
from aegean.risk import RiskConsensusCoordinator, RiskRequest, RiskSubject, RiskContext

async def main():
    coordinator = RiskConsensusCoordinator.create_default()

    request = RiskRequest(
        subject=RiskSubject(
            subject_id="user_001",
            subject_type="user",
            trust_score=0.85,
            total_transactions=120,
            flagged_count=0,
        ),
        context=RiskContext(
            action_type="payment",
            description="Pay supplier invoice",
            amount=2000.0,
            currency="USD",
            geo_location="NY,US",
            channel="web",
            recent_transaction_count=1,
            recent_transaction_amount=200.0,
        )
    )

    decision = await coordinator.evaluate(request)
    print(f"Decision : {decision.decision.value}")
    print(f"Risk     : {decision.risk_level.value}")
    print(f"Confidence: {decision.confidence:.0%}")
    print(f"Rationale: {decision.rationale[:120]}")

asyncio.run(main())
```

### Seed the Knowledge Base

```python
import asyncio
from aegean.memory.global_memory import GlobalMemorySystem
from aegean.risk.data_seed import RiskKnowledgeSeeder

async def main():
    memory = GlobalMemorySystem()
    seeder = RiskKnowledgeSeeder(memory)
    count = await seeder.seed_all()
    print(f"Seeded {count} documents")

asyncio.run(main())
```

---

## Project Structure

```
aegean-consensus/
├── src/aegean/
│   ├── core/                      # Core consensus protocol
│   │   ├── agent.py               # Agent base class + registry
│   │   ├── coordinator.py         # ConsensusCoordinator (Algorithm 1)
│   │   ├── decision_engine.py     # Default + WeightedDecisionEngine
│   │   └── models.py              # Solution, ConsensusResult, Group, ...
│   │
│   ├── memory/                    # Global Memory System
│   │   ├── knowledge_base.py      # Vector KB (mem/Milvus/Pinecone)
│   │   ├── experience_base.py     # Experience store (mem/TimescaleDB)
│   │   ├── global_memory.py       # Unified interface
│   │   ├── prompt_enhancer.py     # RAG prompt templates (9 templates)
│   │   └── knowledge_manager.py   # Bulk import (PDF/TXT/MD/DOCX)
│   │
│   ├── services/
│   │   └── group_chat_service.py  # Group management + consensus exec
│   │
│   ├── risk/                      # Financial Risk Assessment
│   │   ├── models.py              # RiskRequest/Decision/Session/...
│   │   ├── sequencer.py           # Complexity classifier + router
│   │   ├── session.py             # Session lifecycle (24h TTL)
│   │   ├── challenge.py           # Challenge-response gatekeeper
│   │   ├── risk_consensus.py      # RiskConsensusCoordinator
│   │   ├── data_seed.py           # 18-doc public knowledge seeder
│   │   └── validators/
│   │       ├── base_validator.py  # Abstract 3-stage pipeline
│   │       ├── identity_validator.py
│   │       ├── anomaly_validator.py
│   │       ├── compliance_validator.py
│   │       ├── amount_validator.py
│   │       └── context_validator.py
│   │
│   ├── api/
│   │   ├── app.py                 # FastAPI app factory
│   │   ├── group_chat_api.py      # 13 group endpoints
│   │   └── risk_api.py            # 6 risk endpoints
│   │
│   └── integrations/
│       └── autogen_adapter.py     # AutoGen agent adapter
│
├── tests/
├── docs/
├── config/
├── docker/
├── requirements.txt
└── pyproject.toml
```

---

## Performance (from paper)

| Dataset | Latency | Tokens | Accuracy |
|---------|---------|--------|----------|
| GSM8K | 4.1× faster | 1.1× fewer | +2.3% |
| MMLU | 8.8× faster | 2.7× fewer | +1.8% |
| AIME | 20.2× faster | 4.4× fewer | +5.1% |

*N=5 agents, α=3, β=2. Results from original paper.*

---

## Configuration

```yaml
# config/production.yaml
consensus:
  quorum_size: 2
  stability_horizon: 2
  max_rounds: 5
  timeout: 300
  enable_early_termination: true

memory:
  knowledge_backend: milvus       # memory | milvus | pinecone
  experience_backend: timescaledb # memory | timescaledb | postgresql
  knowledge_top_k: 5
  cases_top_k: 3

risk:
  validator_config:
    min_trust_score: 0.3
    new_account_days: 7
    single_limit: 50000
    hourly_limit: 20000
    require_trace_above_amount: 5000
  challenge_ttl_minutes: 30
  session_ttl_hours: 24
```

---

## Deployment

```bash
# Docker
docker-compose up -d

# Kubernetes
kubectl apply -f k8s/aegean-deployment.yaml
```

---

## Contributing

```bash
pip install -r requirements-dev.txt
pytest tests/
flake8 src/
black src/
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

MIT License. See [LICENSE](./LICENSE).

---

## Citation

```bibtex
@article{aegean2024,
  title={Reaching Agreement Among Reasoning LLM Agents},
  journal={arXiv preprint arXiv:2512.20184},
  year={2024}
}
```

---

<div align="center">

Built for the Multi-Agent AI Community

</div>
