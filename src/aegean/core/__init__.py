"""
Core consensus protocol implementation.
"""

from aegean.core.agent import Agent, AgentRegistry
from aegean.core.models import (
    Solution,
    ConsensusState,
    ConsensusResult,
    ConsensusConfig,
    CollaborationMode,
    Group,
    GroupMember,
    Message,
    GroupConsensusResult,
)
from aegean.core.decision_engine import DecisionEngine, DefaultDecisionEngine, WeightedDecisionEngine
from aegean.core.coordinator import ConsensusCoordinator

__all__ = [
    "Agent",
    "AgentRegistry",
    "Solution",
    "ConsensusState",
    "ConsensusResult",
    "ConsensusConfig",
    "CollaborationMode",
    "Group",
    "GroupMember",
    "Message",
    "GroupConsensusResult",
    "DecisionEngine",
    "DefaultDecisionEngine",
    "WeightedDecisionEngine",
    "ConsensusCoordinator",
]

