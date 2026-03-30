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
    """Minimal OpenAI client wrapper supporting custom base_url (OpenAI-compatible endpoints)."""
    try:
        import openai

        class SimpleOpenAIClient:
            def __init__(self, api_key: str, model: str, base_url: str = None):
                client_kwargs = {"api_key": api_key}
                if base_url:
                    # Strip trailing /chat/completions if user mistakenly included it
                    base_url = base_url.rstrip("/")
                    if base_url.endswith("/chat/completions"):
                        base_url = base_url[: -len("/chat/completions")]
                    client_kwargs["base_url"] = base_url
                self.client = openai.AsyncOpenAI(**client_kwargs)
                self.model = model
                self.last_usage = None
                self.last_provider = "openai"

            async def complete(self, prompt: str) -> str:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                usage = getattr(resp, "usage", None)
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                }
                return resp.choices[0].message.content or ""

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        if base_url:
            logger.info(f"LLM client: OpenAI-compatible ({base_url}) model={model}")
        else:
            logger.info(f"LLM client: OpenAI ({model})")
        return SimpleOpenAIClient(api_key=api_key, model=model, base_url=base_url)
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
                self.last_usage = None
                self.last_provider = "anthropic"

            async def complete(self, prompt: str) -> str:
                msg = await self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = getattr(msg, "usage", None)
                self.last_usage = {
                    "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                    "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                    "total_tokens": (
                        (getattr(usage, "input_tokens", 0) if usage else 0)
                        + (getattr(usage, "output_tokens", 0) if usage else 0)
                    ),
                }
                return msg.content[0].text if msg.content else ""

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        logger.info(f"LLM client: Anthropic ({model})")
        return SimpleAnthropicClient(api_key=api_key, model=model)
    except ImportError:
        logger.warning("anthropic package not installed. Run: pip install anthropic")
        return None


def build_per_agent_llm_clients() -> list:
    """
    Build per-agent LLM clients from AGENT_{i}_MODEL env vars.

    Supports two configuration styles:

    Style 1 - All agents share one model (simple):
        OPENAI_API_KEY=sk-xxx
        OPENAI_BASE_URL=https://praka.ai/v1
        OPENAI_MODEL=gpt-4o
        AEGEAN_AGENT_COUNT=3

    Style 2 - Each agent uses a different model (multi-model consensus):
        OPENAI_API_KEY=sk-xxx
        OPENAI_BASE_URL=https://praka.ai/v1
        AEGEAN_AGENT_COUNT=4
        AGENT_0_MODEL=gpt-4o
        AGENT_1_MODEL=deepseek-v3.2
        AGENT_2_MODEL=gpt-5.1
        AGENT_3_MODEL=kimi-k2.5-thinking

    Returns list of (model_name, llm_client) tuples.
    """
    agent_count = int(os.getenv("AEGEAN_AGENT_COUNT", "3"))
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Fix common mistake: base_url should not include /chat/completions
    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
            logger.warning(
                f"OPENAI_BASE_URL should not include /chat/completions. "
                f"Auto-corrected to: {base_url}"
            )

    clients = []
    for i in range(agent_count):
        # Per-agent model override: AGENT_0_MODEL, AGENT_1_MODEL, ...
        model = os.getenv(f"AGENT_{i}_MODEL", default_model)
        client = None
        if api_key and not api_key.startswith("sk-your"):
            client = _build_simple_openai_client_with(api_key, model, base_url)
        clients.append((model, client))
        logger.info(f"  agent_{i} -> model={model}")

    return clients


def _build_simple_openai_client_with(api_key: str, model: str, base_url: str = None):
    """Build a single OpenAI-compatible client for a specific model."""
    try:
        import openai

        class PerModelClient:
            def __init__(self, api_key, model, base_url):
                kwargs = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                self.client = openai.AsyncOpenAI(**kwargs)
                self.model = model
                self.last_usage = None
                self.last_provider = "openai"

            async def complete(self, prompt: str) -> str:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                usage = getattr(resp, "usage", None)
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                }
                return resp.choices[0].message.content or ""

        return PerModelClient(api_key, model, base_url)
    except ImportError:
        return None


def build_agent_registry():
    """
    Build AgentRegistry with configured agents.

    Supports per-agent model configuration via AGENT_{i}_MODEL env vars.
    See build_per_agent_llm_clients() for details.
    """
    from aegean.core.agent import AgentRegistry

    registry = AgentRegistry()
    agent_count = int(os.getenv("AEGEAN_AGENT_COUNT", "3"))

    logger.info(f"Building AgentRegistry with {agent_count} agents:")
    per_agent_clients = build_per_agent_llm_clients()

    # Try AutoGen adapter first, fall back to minimal agents
    try:
        from aegean.integrations.autogen_adapter import AutoGenAgentAdapter
        for i, (model, llm_client) in enumerate(per_agent_clients):
            agent = AutoGenAgentAdapter(
                agent_id=f"agent_{i}",
                llm_client=llm_client,
                capability_weight=1.0,
            )
            registry.register_agent(agent)
        logger.info(f"Registered {agent_count} AutoGen agents")
    except Exception as e:
        logger.warning(f"AutoGen not available ({e}), using minimal agents")
        _register_minimal_agents_multi(registry, per_agent_clients)

    logger.info(f"AgentRegistry ready: {registry}")
    return registry


def _register_minimal_agents(registry, count: int, llm_client):
    """Register minimal agents all sharing one LLM client (legacy fallback)."""
    clients = [(os.getenv("OPENAI_MODEL", "gpt-4o"), llm_client)] * count
    _register_minimal_agents_multi(registry, clients)


def _register_minimal_agents_multi(registry, per_agent_clients: list):
    """
    Register minimal agents with per-agent LLM clients.
    Used when AutoGen is not available.
    """
    from aegean.core.agent import Agent
    from aegean.core.models import Solution

    class MinimalAgent(Agent):
        def __init__(self, agent_id: str, llm_client=None, model_name: str = "", **kwargs):
            super().__init__(agent_id=agent_id, **kwargs)
            self._llm = llm_client
            self._model_name = model_name

        async def generate_solution(self, task: str) -> Solution:
            if self._llm:
                try:
                    answer = await self._llm.complete(
                        f"Task: {task}\n\nProvide a concise answer:"
                    )
                    return Solution(agent_id=self.agent_id, answer=answer, confidence=0.8)
                except Exception as e:
                    logger.warning(f"{self.agent_id} ({self._model_name}) LLM failed: {e}")
            return Solution(
                agent_id=self.agent_id,
                answer=f"[{self.agent_id}] Unable to analyze without LLM",
                confidence=0.3,
            )

        async def refine_solution(self, refinement_set) -> Solution:
            if not refinement_set:
                return await self.generate_solution("refinement")
            from collections import Counter
            answers = [s.answer for s in refinement_set]
            majority = Counter(answers).most_common(1)[0][0]
            return Solution(
                agent_id=self.agent_id,
                answer=majority,
                confidence=0.75,
                reasoning="Refined based on peer solutions",
            )

    for i, (model_name, llm_client) in enumerate(per_agent_clients):
        agent = MinimalAgent(
            agent_id=f"agent_{i}",
            llm_client=llm_client,
            model_name=model_name,
        )
        registry.register_agent(agent)
    logger.info(f"Registered {len(per_agent_clients)} minimal agents")


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
    experience_backend = os.getenv("EXPERIENCE_BACKEND", "memory")
    experience_config = {}
    if experience_backend in ("postgresql", "timescaledb"):
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            logger.warning(
                "EXPERIENCE_BACKEND=postgresql but DATABASE_URL not set. "
                "Falling back to memory backend."
            )
            experience_backend = "memory"
        else:
            experience_config["database_url"] = database_url

    memory = GlobalMemorySystem(
        config={
            "knowledge_backend": os.getenv("KNOWLEDGE_BACKEND", "memory"),
            "experience_backend": experience_backend,
            "experience_config": experience_config,
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

