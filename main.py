#!/usr/bin/env python3
"""
Aegean Consensus - Application Entry Point.

Starts the FastAPI server with configured agents and services.
Reads configuration from environment variables / .env file.

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 8000
    uvicorn main:app --reload
"""

import os
import asyncio
import argparse
import logging
from typing import Optional

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment directly

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

log_level = os.getenv("AEGEAN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("aegean.main")


def build_llm_client():
    """
    Build LLM client from environment variables.
    Returns None if no API key configured (validators fall back to pre-screen only).
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if openai_key and not openai_key.startswith("sk-your"):
        try:
            from aegean.llm.openai_client import OpenAIClient
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            logger.info(f"LLM client: OpenAI ({model})")
            return OpenAIClient(api_key=openai_key, model=model)
        except ImportError:
            logger.warning("OpenAI client not available, trying built-in")
            return _build_simple_openai_client(openai_key)

    if anthropic_key and not anthropic_key.startswith("sk-ant-your"):
        logger.info("LLM client: Anthropic Claude")
        return _build_simple_anthropic_client(anthropic_key)

    logger.warning(
        "No LLM API key configured. Risk validators will use pre-screen rules only "
        "(no LLM deep analysis). Set OPENAI_API_KEY in .env to enable full analysis."
    )
    return None


def _build_simple_openai_client(api_key: str):
    """Minimal OpenAI client wrapper if no custom client module exists."""
    try:
        import openai

        class SimpleOpenAIClient:
            def __init__(self, api_key: str, model: str):
                self.client = openai.AsyncOpenAI(api_key=api_key)
                self.model = model

            async def complete(self, prompt: str) -> str:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                return resp.choices[0].message.content or ""

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"LLM client: OpenAI built-in ({model})")
        return SimpleOpenAIClient(api_key=api_key, model=model)
    except ImportError:
        logger.warning("openai package not installed. Run: pip install openai")
        return None


def _build_simple_anthropic_client(api_key: str):
    """Minimal Anthropic client wrapper."""
    try:
        import anthropic

        class SimpleAnthropicClient:
            def __init__(self, api_key: str, model: str):
                self.client = anthropic.AsyncAnthropic(api_key=api_key)
                self.model = model

            async def complete(self, prompt: str) -> str:
                msg = await self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text if msg.content else ""

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        logger.info(f"LLM client: Anthropic ({model})")
        return SimpleAnthropicClient(api_key=api_key, model=model)
    except ImportError:
        logger.warning("anthropic package not installed. Run: pip install anthropic")
        return None


def build_agent_registry():
    """
    Build AgentRegistry with configured agents.

    Agent configuration:
    - Each agent needs an LLM client to generate solutions
    - capability_weight differentiates domain expertise
    - specialization maps domain names to proficiency scores
    """
    from aegean.core.agent import AgentRegistry

    registry = AgentRegistry()
    agent_count = int(os.getenv("AEGEAN_AGENT_COUNT", "3"))
    llm_client = build_llm_client()

    # Try to register AutoGen-based agents if available
    try:
        from aegean.integrations.autogen_adapter import AutoGenAgentAdapter
        for i in range(agent_count):
            agent = AutoGenAgentAdapter(
                agent_id=f"agent_{i}",
                llm_client=llm_client,
                capability_weight=1.0,
            )
            registry.register_agent(agent)
        logger.info(f"Registered {agent_count} AutoGen agents")
    except Exception as e:
        logger.warning(f"Could not create AutoGen agents ({e}), using minimal agents")
        _register_minimal_agents(registry, agent_count, llm_client)

    logger.info(f"AgentRegistry ready: {registry}")
    return registry


def _register_minimal_agents(registry, count: int, llm_client):
    """
    Register minimal agents that work without AutoGen.
    These agents use the LLM client directly.
    """
    from aegean.core.agent import Agent
    from aegean.core.models import Solution

    class MinimalAgent(Agent):
        def __init__(self, agent_id: str, llm_client=None, **kwargs):
            super().__init__(agent_id=agent_id, **kwargs)
            self._llm = llm_client

        async def generate_solution(self, task: str) -> Solution:
            if self._llm:
                try:
                    answer = await self._llm.complete(
                        f"Task: {task}\n\nProvide a concise answer:"
                    )
                    return Solution(agent_id=self.agent_id, answer=answer, confidence=0.8)
                except Exception as e:
                    logger.warning(f"{self.agent_id} LLM failed: {e}")
            return Solution(
                agent_id=self.agent_id,
                answer=f"[{self.agent_id}] Unable to analyze without LLM",
                confidence=0.3,
            )

        async def refine_solution(self, refinement_set) -> Solution:
            if not refinement_set:
                return await self.generate_solution("refinement")
            # Default: agree with majority answer
            from collections import Counter
            answers = [s.answer for s in refinement_set]
            majority = Counter(answers).most_common(1)[0][0]
            return Solution(
                agent_id=self.agent_id,
                answer=majority,
                confidence=0.75,
                reasoning="Refined based on peer solutions",
            )

    for i in range(count):
        agent = MinimalAgent(agent_id=f"agent_{i}", llm_client=llm_client)
        registry.register_agent(agent)
    logger.info(f"Registered {count} minimal agents")


async def seed_on_startup(app_state: dict):
    """Seed knowledge base on first startup."""
    try:
        from aegean.risk.data_seed import RiskKnowledgeSeeder
        memory = app_state.get("memory_system")
        if memory:
            seeder = RiskKnowledgeSeeder(memory)
            count = await seeder.seed_all(skip_if_exists=True)
            if count > 0:
                logger.info(f"Knowledge base seeded with {count} documents")
    except Exception as e:
        logger.warning(f"Knowledge base seed failed (non-fatal): {e}")


def create_application():
    """
    Create FastAPI application with all services configured.
    This is the factory used by uvicorn.
    """
    from aegean.api.app import create_app
    from aegean.memory.global_memory import GlobalMemorySystem

    # Build memory system
    memory = GlobalMemorySystem(
        config={
            "knowledge_backend": os.getenv("KNOWLEDGE_BACKEND", "memory"),
            "experience_backend": os.getenv("EXPERIENCE_BACKEND", "memory"),
        }
    )

    # Build LLM client
    llm_client = build_llm_client()

    # Build agent registry
    registry = build_agent_registry()

    # Risk validator config from env
    risk_config = {
        "validator_config": {
            "single_limit": float(os.getenv("RISK_SINGLE_LIMIT", "50000")),
            "hourly_limit": float(os.getenv("RISK_HOURLY_LIMIT", "20000")),
            "require_trace_above_amount": float(os.getenv("RISK_TRACE_REQUIRED_ABOVE", "5000")),
        },
        "challenge_ttl_minutes": int(os.getenv("RISK_CHALLENGE_TTL_MINUTES", "30")),
        "session_ttl_hours": int(os.getenv("RISK_SESSION_TTL_HOURS", "24")),
    }

    # Create FastAPI app
    fastapi_app = create_app(
        agent_registry=registry,
        memory_system=memory,
        llm_client=llm_client,
        enable_cors=True,
    )

    # Register startup: seed knowledge base
    @fastapi_app.on_event("startup")
    async def on_startup():
        await seed_on_startup({"memory_system": memory})
        logger.info("Aegean Consensus API ready")
        logger.info(f"  Docs: http://{os.getenv('AEGEAN_HOST','0.0.0.0')}:{os.getenv('AEGEAN_PORT','8000')}/docs")

    return fastapi_app


# ASGI app instance for uvicorn
app = create_application()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Aegean Consensus API Server")
    parser.add_argument("--host", default=os.getenv("AEGEAN_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AEGEAN_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", default=os.getenv("AEGEAN_RELOAD", "false").lower() == "true")
    parser.add_argument("--workers", type=int, default=int(os.getenv("AEGEAN_WORKERS", "1")))
    args = parser.parse_args()

    logger.info(f"Starting Aegean Consensus on {args.host}:{args.port}")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=log_level.lower(),
    )

