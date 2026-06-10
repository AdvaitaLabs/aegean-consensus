"""
Data models for the Aegean consensus protocol.
"""

from typing import List, Optional, Dict, Any, Tuple
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


class TokenUsage(BaseModel):
    """Unified token usage statistics."""
    tokens_prompt: int = Field(0, ge=0)
    tokens_completion: int = Field(0, ge=0)
    tokens_total: int = Field(0, ge=0)

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> "TokenUsage":
        if not raw:
            return cls()

        prompt = int(
            raw.get("tokens_prompt")
            or raw.get("prompt_tokens")
            or raw.get("input_tokens")
            or 0
        )
        completion = int(
            raw.get("tokens_completion")
            or raw.get("completion_tokens")
            or raw.get("output_tokens")
            or 0
        )
        total = int(
            raw.get("tokens_total")
            or raw.get("total_tokens")
            or (prompt + completion)
        )
        return cls(
            tokens_prompt=max(prompt, 0),
            tokens_completion=max(completion, 0),
            tokens_total=max(total, 0),
        )


class ActionProposal(BaseModel):
    """Optional structured action proposal for sequential decision tasks like ARC."""
    primary_action: str = Field(..., description="Preferred action token")
    backup_action: Optional[str] = Field(None, description="Fallback action token")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Compact reason for the choice")

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["ActionProposal"]:
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            return None

        primary = raw.get("primary_action")
        if not primary:
            return None

        try:
            confidence = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        return cls(
            primary_action=str(primary),
            backup_action=(str(raw["backup_action"]) if raw.get("backup_action") else None),
            confidence=max(0.0, min(confidence, 1.0)),
            reason=str(raw.get("reason", "")),
        )


class Solution(BaseModel):
    """Agent's solution to a task."""
    agent_id: str = Field(..., description="ID of the agent that generated this solution")
    answer: str = Field(..., description="The actual answer/solution")
    reasoning: str = Field(default="", description="Reasoning trace for this solution")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_prompt: int = Field(0, ge=0, description="Prompt tokens consumed by this response")
    tokens_completion: int = Field(0, ge=0, description="Completion tokens consumed by this response")
    usage: Optional[TokenUsage] = Field(
        None,
        description="Optional unified token usage details"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def proposal(self) -> Optional[ActionProposal]:
        return ActionProposal.from_raw(self.metadata.get("proposal"))

    def set_proposal(self, proposal: Optional[ActionProposal]) -> None:
        if proposal is None:
            self.metadata.pop("proposal", None)
        else:
            self.metadata["proposal"] = proposal.model_dump()

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
    # When True, the initial proposal round waits for ALL agents to
    # produce a solution (barrier sync) instead of cutting off at
    # quorum_size. Refinement rounds still honor early termination
    # if it's enabled. Set True for sports/UX use cases where the
    # front-end wants to render every specialist's first take.
    wait_all_in_initial_round: bool = Field(False, description="Disable early termination for round 0")
    # When True, refinement rounds also wait for all agents (barrier
    # sync). Slower (every round runs N LLM calls) but lets the chat
    # UI render every specialist's reaction to peers, not just the
    # first 2 to finish. Set True for the sports chat UX.
    wait_all_in_refinement: bool = Field(False, description="Disable early termination for refinement rounds")
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


class RoundSnapshot(BaseModel):
    """
    One round of the refinement loop, exposed to API consumers.

    Lets downstream services (AegeanBench, dashboards) render the
    multi-agent discussion trace: who said what at each step, how the
    weighted vote shifted, whether quorum was reached.
    """
    round_number: int = Field(..., ge=0)
    quorum_reached: bool = False
    candidate_answer: Optional[str] = None
    candidate_confidence: float = Field(0.0, ge=0.0, le=1.0)
    agent_solutions: List[Solution] = Field(default_factory=list)
    weighted_votes: Dict[str, float] = Field(default_factory=dict)
    stability_counter: int = Field(0, ge=0)


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
    # NEW: per-round discussion trace. Empty for stateless callers that
    # don't want the verbose payload; populated when callers ask for it
    # via include_rounds_history in the API request body.
    rounds_history: List[RoundSnapshot] = Field(
        default_factory=list,
        description="Per-round agent solutions + weighted votes for UI rendering",
    )

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
    COLLABORATION = "collaboration"
    CONSENSUS = "consensus"
    HYBRID = "hybrid"


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
    shared_knowledge: "GroupSharedKnowledge" = Field(default_factory=lambda: GroupSharedKnowledge())

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


class RoundDiscussion(BaseModel):
    """
    Discussion record for a single consensus round.
    """
    round_number: int = Field(..., ge=1)
    agent_responses: Dict[str, Solution] = Field(default_factory=dict)
    candidate_answer: Optional[str] = None
    candidate_confidence: Optional[float] = None
    stability_counter: int = Field(0, ge=0)
    consensus_status: str = Field("ongoing")
    timestamp: datetime = Field(default_factory=datetime.now)
    # Weighted vote distribution for this round, keyed by normalized
    # answer label (e.g. "home_win", "draw", "away_win" for sports).
    # Empty until the orchestrator populates it via record_round().
    weighted_votes: Dict[str, float] = Field(default_factory=dict)


class KnowledgeGraphEntity(BaseModel):
    entity_id: str
    entity_type: str
    name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphRelation(BaseModel):
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    graph_id: str
    consensus_id: str
    group_id: str
    entities: List[KnowledgeGraphEntity] = Field(default_factory=list)
    relations: List[KnowledgeGraphRelation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class GroupSkill(BaseModel):
    skill_id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(default="", description="Skill description")
    applicable_task_types: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    required_data_sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupKnowledgeDocument(BaseModel):
    doc_id: str = Field(..., description="Document identifier")
    category: str = Field(..., description="Knowledge category")
    title: Optional[str] = Field(None, description="Optional document title")
    summary: Optional[str] = Field(None, description="Optional short summary")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupSharedKnowledge(BaseModel):
    static_documents: List[GroupKnowledgeDocument] = Field(default_factory=list)
    historical_case_ids: List[str] = Field(default_factory=list)
    skills: List[GroupSkill] = Field(default_factory=list)
    knowledge_graph_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupKnowledgeInjection(BaseModel):
    group_id: str
    memory_context: str = ""
    skill_descriptions: List[str] = Field(default_factory=list)
    document_summaries: List[str] = Field(default_factory=list)
    historical_case_ids: List[str] = Field(default_factory=list)
    graph_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupConsensusResult(BaseModel):
    """Extended result for group consensus/collaboration execution."""
    consensus_id: str
    group_id: str
    message_id: Optional[str] = None
    mode: CollaborationMode = CollaborationMode.CONSENSUS
    success: bool = False
    final_solution: Optional[Solution] = None
    agent_responses: List[Solution] = Field(default_factory=list)
    discussion_rounds: List[RoundDiscussion] = Field(default_factory=list)
    consensus_path: List[str] = Field(default_factory=list)
    weighted_votes: Optional[Dict[str, float]] = None
    total_weight: Optional[float] = None
    rounds_used: int = Field(0, ge=0)
    participating_agents: List[str] = Field(default_factory=list)
    execution_time: float = Field(0.0, ge=0.0)
    consensus_reached: bool = False
    tokens_prompt: int = Field(0, ge=0)
    tokens_completion: int = Field(0, ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    knowledge_graph: Optional[KnowledgeGraph] = None
    agent_graph: Optional[Dict[str, Any]] = None
