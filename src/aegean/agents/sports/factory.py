"""
Factory for building the full World Cup agent panel.

build_worldcup_agents() returns a list of 7 agent instances (6 prediction
agents + 1 crowd-sentiment ChatAgent), each wired to its assigned LLM
client. The factory does NOT register them in any registry; callers
typically do that explicitly so they retain control over registration
order, weights, and per-agent overrides.

LLM-client construction is delegated to a caller-provided factory:
    llm_factory(model_name: str) -> llm_client | None
This avoids hard-coding OpenAI vs Anthropic vs DeepSeek logic here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from aegean.agents.sports.base_sports_agent import BaseSportsAgent
from aegean.agents.sports.chat_agent import ChatAgent, ChatFetcher
from aegean.agents.sports.market_agent import MarketAgent
from aegean.agents.sports.news_agent import NewsAgent
from aegean.agents.sports.occult_agent import OccultAgent
from aegean.agents.sports.player_agent import PlayerAgent
from aegean.agents.sports.stats_agent import StatsAgent
from aegean.agents.sports.strategy_agent import StrategyAgent

logger = logging.getLogger(__name__)


# Default LLM model assignment per agent (sprint configuration).
# Operators can override by passing a custom model_map to the factory.
DEFAULT_AGENT_MODEL_MAP: Dict[str, str] = {
    "stats_specialist": "claude-opus-4-7",
    "player_specialist": "gpt-5",
    "strategy_specialist": "claude-opus-4-7",
    "market_specialist": "deepseek-v3",
    "news_specialist": "claude-haiku-4-5",
    "occult_specialist": "claude-haiku-4-5",
    "chat_specialist": "claude-haiku-4-5",
}


def build_worldcup_agents(
    llm_factory: Optional[Callable[[str], Any]] = None,
    model_map: Optional[Dict[str, str]] = None,
    chat_fetcher: Optional[ChatFetcher] = None,
    include_chat: bool = True,
    capability_weight_overrides: Optional[Dict[str, float]] = None,
) -> List[BaseSportsAgent]:
    """
    Instantiate the World Cup agent panel.

    Args:
        llm_factory: callable that returns an LLM client given a model name.
            Signature: f(model_name) -> client or None. If None, agents are
            constructed without LLM clients (they will return uniform
            fallbacks). This is fine for dry-runs and tests.
        model_map: agent_role -> model_name. Falls back to DEFAULT_AGENT_MODEL_MAP.
        chat_fetcher: optional ChatFetcher injected into ChatAgent. If None,
            ChatAgent uses its default HTTP fetcher at request time.
        include_chat: when False, return only the six prediction agents.
            Useful in benchmark modes where the chat service is unavailable.
        capability_weight_overrides: agent_role -> weight, overrides defaults.

    Returns:
        Ordered list of BaseSportsAgent instances. Suggested registration
        order matches paper convention (highest-weight first):
            stats, player, strategy, market, news, occult, [chat].
    """
    mm = dict(DEFAULT_AGENT_MODEL_MAP)
    if model_map:
        mm.update(model_map)
    weight_overrides = capability_weight_overrides or {}

    def _make_client(role: str) -> Any:
        if llm_factory is None:
            return None
        model = mm.get(role)
        if not model:
            return None
        try:
            return llm_factory(model)
        except Exception as e:
            logger.warning("LLM factory failed for %s (%s): %s", role, model, e)
            return None

    def _make(cls):
        role = cls.ROLE
        return cls(
            llm_client=_make_client(role),
            model_name=mm.get(role, ""),
            capability_weight=weight_overrides.get(role),
        )

    agents: List[BaseSportsAgent] = [
        _make(StatsAgent),
        _make(PlayerAgent),
        _make(StrategyAgent),
        _make(MarketAgent),
        _make(NewsAgent),
        _make(OccultAgent),
    ]

    if include_chat:
        agents.append(
            ChatAgent(
                llm_client=_make_client("chat_specialist"),
                model_name=mm.get("chat_specialist", ""),
                capability_weight=weight_overrides.get("chat_specialist"),
                chat_fetcher=chat_fetcher,
            )
        )

    return agents
