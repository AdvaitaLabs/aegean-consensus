"""
Sports prediction agents for World Cup 2026.

Public entry points:
    - 6 prediction agents that vote in consensus:
        StatsAgent, PlayerAgent, StrategyAgent,
        MarketAgent, NewsAgent, OccultAgent
    - 1 crowd sentiment agent:
        ChatAgent
    - Helper factory:
        build_worldcup_agents(...)
    - Q&A handler for @-mentions (V2 feature):
        ChatQAHandler, ChatQAResponse
"""

from aegean.agents.sports.base_sports_agent import BaseSportsAgent
from aegean.agents.sports.iching_agent import IChingAgent
from aegean.agents.sports.chat_agent import ChatAgent, ChatFetcher
from aegean.agents.sports.chat_qa import ChatQAHandler, ChatQAResponse
from aegean.agents.sports.factory import (
    DEFAULT_AGENT_MODEL_MAP,
    build_worldcup_agents,
)
from aegean.agents.sports.market_agent import MarketAgent
from aegean.agents.sports.news_agent import NewsAgent
from aegean.agents.sports.occult_agent import OccultAgent
from aegean.agents.sports.player_agent import PlayerAgent
from aegean.agents.sports.stats_agent import StatsAgent
from aegean.agents.sports.strategy_agent import StrategyAgent

__all__ = [
    "BaseSportsAgent",
    "StatsAgent",
    "PlayerAgent",
    "StrategyAgent",
    "MarketAgent",
    "NewsAgent",
    "OccultAgent",
    "IChingAgent",
    "ChatAgent",
    "ChatFetcher",
    "ChatQAHandler",
    "ChatQAResponse",
    "DEFAULT_AGENT_MODEL_MAP",
    "build_worldcup_agents",
]
