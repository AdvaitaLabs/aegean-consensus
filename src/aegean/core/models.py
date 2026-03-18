"""
Data models for the Aegean consensus protocol.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ConsensusStatus(str, Enum):
    """Consensus execution status."""
    INITIALIZING = "initializing"
    ELECTING_LEADER = "electing_leader"
    COLLECTING_INITIAL = "collecting_initial"
    REFINING = "refining"
    CONSENSUS_REACHED = "consensus_reached"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Solution(BaseModel):
    """Agent's solution to a task."""
    agent_id: str = Field(..., description="ID of the agent that generated this solution")
    answer: str = Field(..., description="The actual answer/solution")
    reasoning: str = Field(default="", description="Reasoning trace for this solution")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "agent_0",
                "answer": "42",
                "reasoning": "The answer to life, universe, and everything",
                "confidence": 0.95,
            }
        }


class ConsensusConfig(BaseModel):
    """Configuration for consensus execution."""
    quorum_size: int = Field(2, ge=1, description="Minimum agents needed for quorum (α)")
    stability_horizon: int = Field(2, ge=1, description="Rounds to maintain stability (β)")
    max_rounds: int = Field(5, ge=1, description="Maximum refinement rounds")
    timeout: int = Field(300, ge=1, description="Timeout in seconds")
    enable_early_termination: bool = Field(True, description="Cancel slow agents after quorum")
    enable_openclaw: bool = Field(False, description="Enable OpenClaw integration")


class ConsensusState(BaseModel):
    """Current state of consensus execution."""
    consensus_id: str = Field(..., description="Unique consensus execution ID")
    status: ConsensusStatus = Field(ConsensusStatus.INITIALIZING)
    current_round: int = Field(0, ge=0)
    leader_id: Optional[str] = Field(None, description="Current leader agent ID")
    participating_agents: List[str] = Field(default_factory=list)
    candidate_solution: Optional[Solution] = Field(None)
    stability_counter: int = Field(0, ge=0, description="Consecutive rounds with same candidate")
    solutions_history: List[List[Solution]] = Field(
        default_factory=list,
        description="Solutions from each round"
    )
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class ConsensusResult(BaseModel):
    """Final result of consensus execution."""
    consensus_id: str
    success: bool = Field(..., description="Whether consensus was reached")
    final_solution: Optional[Solution] = Field(None)
    rounds_used: int = Field(0, ge=0)
    participating_agents: List[str] = Field(default_factory=list)
    execution_time: float = Field(0.0, ge=0.0, description="Total execution time in seconds")
    tokens_used: int = Field(0, ge=0, description="Total tokens consumed")
    consensus_reached: bool = Field(False)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "consensus_id": "task-001",
                "success": True,
                "final_solution": {
                    "agent_id": "agent_0",
                    "answer": "42",
                    "reasoning": "Consensus reached",
                },
                "rounds_used": 2,
                "participating_agents": ["agent_0", "agent_1", "agent_2"],
                "execution_time": 5.2,
                "consensus_reached": True,
            }
        }


class OpenClawNodeInfo(BaseModel):
    """Information about an OpenClaw node."""
    node_id: str = Field(..., description="Unique node identifier")
    endpoint: str = Field(..., description="Node HTTP/gRPC endpoint")
    status: str = Field("idle", description="Node status: idle, busy, error")
    capabilities: List[str] = Field(default_factory=list)
    activated_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None


class CollaborationMode(str, Enum):
    """Agent collaboration mode."""
    COLLABORATION = "collaboration"  # Different agents do different tasks
    CONSENSUS = "consensus"          # Multiple agents answer same question
    HYBRID = "hybrid"                # Mix of both


class Group(BaseModel):
    """
    Group of agents working together.
    
    A group can operate in different modes:
    - Collaboration: Agents work on different subtasks
    - Consensus: Agents vote on the same question
    - Hybrid: Mix of both approaches
    """
    group_id: str = Field(..., description="Unique group identifier")
    group_name: str = Field(..., description="Human-readable group name")
    description: Optional[str] = Field(None, description="Group description")
    mode: CollaborationMode = Field(
        CollaborationMode.CONSENSUS,
        description="Collaboration mode"
    )
    created_by: str = Field(..., description="User ID who created this group")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "group_id": "group-001",
                "group_name": "Financial Risk Team",
                "description": "Team for credit assessment and fraud detection",
                "mode": "hybrid",
                "created_by": "user_123",
            }
        }


class GroupMember(BaseModel):
    """
    Member of a group (agent).
    
    Tracks agent's role, mode, and participation in the group.
    """
    group_id: str = Field(..., description="Group this member belongs to")
    agent_id: str = Field(..., description="Agent identifier")
    role: Optional[str] = Field(None, description="Agent's role (e.g., 'analyst', 'reviewer')")
    mode: CollaborationMode = Field(
        CollaborationMode.CONSENSUS,
        description="How this agent participates"
    )
    capability_weight: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Agent's capability weight"
    )
    specialization: Dict[str, float] = Field(
        default_factory=dict,
        description="Domain -> proficiency mapping"
    )
    added_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(True, description="Whether agent is active in group")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "group_id": "group-001",
                "agent_id": "agent_0",
                "role": "credit_analyst",
                "mode": "consensus",
                "capability_weight": 1.0,
                "specialization": {"credit": 0.95, "fraud": 0.85},
            }
        }


class Message(BaseModel):
    """
    Message in a group chat.
    
    Can be from user or agent.
    """
    message_id: str = Field(..., description="Unique message identifier")
    group_id: str = Field(..., description="Group this message belongs to")
    sender_id: str = Field(..., description="Sender ID (user or agent)")
    sender_type: str = Field(..., description="Sender type: 'user' or 'agent'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg-001",
                "group_id": "group-001",
                "sender_id": "user_123",
                "sender_type": "user",
                "content": "Evaluate customer credit rating",
            }
        }


class GroupConsensusResult(BaseModel):
    """
    Result of group consensus execution.
    
    Extends ConsensusResult with group-specific information.
    """
    consensus_id: str
    group_id: str = Field(..., description="Group that executed this consensus")
    message_id: Optional[str] = Field(None, description="Message that triggered this")
    mode: CollaborationMode = Field(..., description="Collaboration mode used")
    
    # Consensus results
    success: bool = Field(..., description="Whether consensus was reached")
    final_solution: Optional[Solution] = Field(None)
    
    # Agent responses (for display)
    agent_responses: List[Solution] = Field(
        default_factory=list,
        description="Individual agent responses"
    )
    
    # Weighted voting details
    weighted_votes: Optional[Dict[str, float]] = Field(
        None,
        description="Weighted votes for each answer"
    )
    total_weight: Optional[float] = Field(None, description="Total voting weight")
    
    # Execution details
    rounds_used: int = Field(0, ge=0)
    participating_agents: List[str] = Field(default_factory=list)
    execution_time: float = Field(0.0, ge=0.0)
    consensus_reached: bool = Field(False)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "consensus_id": "consensus-001",
                "group_id": "group-001",
                "message_id": "msg-001",
                "mode": "consensus",
                "success": True,
                "final_solution": {
                    "agent_id": "consensus",
                    "answer": "B",
                    "reasoning": "2 out of 3 agents agreed",
                },
                "agent_responses": [
                    {"agent_id": "agent_0", "answer": "B", "confidence": 0.9},
                    {"agent_id": "agent_1", "answer": "B", "confidence": 0.85},
                    {"agent_id": "agent_2", "answer": "C", "confidence": 0.7},
                ],
                "weighted_votes": {"B": 1.75, "C": 0.7},
                "total_weight": 2.45,
                "rounds_used": 1,
                "consensus_reached": True,
            }
        }

