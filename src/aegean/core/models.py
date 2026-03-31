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


class RoundDiscussion(BaseModel):
    """
    Discussion record for a single consensus round.
    
    Tracks what each agent said and how the consensus evolved.
    """
    round_number: int = Field(..., description="Round number (1-indexed)")
    agent_responses: Dict[str, Solution] = Field(
        ...,
        description="agent_id -> Solution for this round"
    )
    consensus_status: str = Field(
        ...,
        description="Status: 'forming', 'reached', 'diverging'"
    )
    candidate_answer: Optional[str] = Field(None, description="Current candidate answer")
    candidate_confidence: Optional[float] = Field(None, description="Candidate confidence")
    stability_counter: int = Field(0, description="Consecutive rounds with same candidate")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentRelationship(BaseModel):
    """
    Relationship between two agents in a group.
    
    Tracks influence, trust, and interaction patterns.
    """
    source_agent_id: str = Field(..., description="Source agent")
    target_agent_id: str = Field(..., description="Target agent")
    influence_weight: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="How much source influences target (0-1)"
    )
    trust_score: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="How much target trusts source (0-1)"
    )
    disagreement_count: int = Field(0, ge=0, description="Times they disagreed")
    agreement_count: int = Field(0, ge=0, description="Times they agreed")
    last_interaction: Optional[datetime] = None


class GroupGraph(BaseModel):
    """
    Agent relationship graph for a group.
    
    Visualizes how agents influence each other.
    """
    group_id: str = Field(..., description="Group ID")
    nodes: List[str] = Field(..., description="Agent IDs in the graph")
    edges: List[AgentRelationship] = Field(
        default_factory=list,
        description="Relationships between agents"
    )
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def get_key_agents(self) -> List[tuple[str, float]]:
        """Return agents sorted by total influence (outgoing edges)."""
        influence_map: Dict[str, float] = {node: 0.0 for node in self.nodes}
        for edge in self.edges:
            influence_map[edge.source_agent_id] += edge.influence_weight
        return sorted(influence_map.items(), key=lambda x: x[1], reverse=True)
    
    def get_influence_path(self, from_agent: str, to_agent: str) -> Optional[List[str]]:
        """Find influence path from one agent to another (BFS)."""
        if from_agent == to_agent:
            return [from_agent]
        
        visited = set()
        queue = [(from_agent, [from_agent])]
        
        while queue:
            current, path = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self.edges:
                if edge.source_agent_id == current and edge.influence_weight > 0:
                    next_agent = edge.target_agent_id
                    if next_agent == to_agent:
                        return path + [next_agent]
                    if next_agent not in visited:
                        queue.append((next_agent, path + [next_agent]))
        
        return None


class KnowledgeGraphEntity(BaseModel):
    """
    Entity in a knowledge graph (person, company, event, etc).
    """
    entity_id: str = Field(..., description="Unique entity ID")
    entity_type: str = Field(..., description="Type: user, company, event, location, etc")
    name: str = Field(..., description="Entity name")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entity attributes (trust_score, amount, etc)"
    )


class KnowledgeGraphRelation(BaseModel):
    """
    Relationship between entities in a knowledge graph.
    """
    relation_id: str = Field(..., description="Unique relation ID")
    source_entity_id: str = Field(..., description="Source entity")
    target_entity_id: str = Field(..., description="Target entity")
    relation_type: str = Field(..., description="Type: initiates, transfers, influences, etc")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relation properties (amount, confidence, etc)"
    )


class KnowledgeGraph(BaseModel):
    """
    Knowledge graph extracted from seed data or risk context.
    
    Represents entities and their relationships.
    """
    graph_id: str = Field(..., description="Unique graph ID")
    source_type: str = Field(..., description="Source: risk_request, seed_data, etc")
    source_id: Optional[str] = Field(None, description="ID of source (request_id, etc)")
    
    entities: List[KnowledgeGraphEntity] = Field(
        default_factory=list,
        description="Nodes in the graph"
    )
    relations: List[KnowledgeGraphRelation] = Field(
        default_factory=list,
        description="Edges in the graph"
    )
    
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeGraphEntity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None
    
    def get_related_entities(self, entity_id: str, relation_type: Optional[str] = None) -> List[str]:
        """Get entities related to given entity."""
        related = []
        for rel in self.relations:
            if rel.source_entity_id == entity_id:
                if relation_type is None or rel.relation_type == relation_type:
                    related.append(rel.target_entity_id)
        return related


class GroupConsensusResult(BaseModel):
    """
    Result of group consensus execution.
    
    Extends ConsensusResult with group-specific information and discussion history.
    """
    consensus_id: str
    group_id: str = Field(..., description="Group that executed this consensus")
    message_id: Optional[str] = Field(None, description="Message that triggered this")
    mode: CollaborationMode = Field(..., description="Collaboration mode used")

    # Unified token usage
    tokens_prompt: int = Field(0, ge=0, description="Aggregated prompt tokens")
    tokens_completion: int = Field(0, ge=0, description="Aggregated completion tokens")
    usage: Optional[TokenUsage] = Field(
        None,
        description="Optional aggregate token usage details"
    )
    
    # Consensus results
    success: bool = Field(..., description="Whether consensus was reached")
    final_solution: Optional[Solution] = Field(None)
    
    # Agent responses (for display)
    agent_responses: List[Solution] = Field(
        default_factory=list,
        description="Individual agent responses"
    )
    
    # Discussion history (NEW)
    discussion_rounds: List[RoundDiscussion] = Field(
        default_factory=list,
        description="Discussion process across rounds"
    )
    consensus_path: List[str] = Field(
        default_factory=list,
        description="Order of agents that influenced consensus formation"
    )
    
    # Agent relationship graph (NEW)
    agent_graph: Optional[GroupGraph] = Field(
        None,
        description="Agent relationship graph after consensus"
    )
    
    # Knowledge graph (NEW)
    knowledge_graph: Optional[KnowledgeGraph] = Field(
        None,
        description="Knowledge graph extracted from context"
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
                "discussion_rounds": [
                    {
                        "round_number": 1,
                        "agent_responses": {
                            "agent_0": {"agent_id": "agent_0", "answer": "B"},
                            "agent_1": {"agent_id": "agent_1", "answer": "B"},
                            "agent_2": {"agent_id": "agent_2", "answer": "C"},
                        },
                        "consensus_status": "forming",
                        "candidate_answer": "B",
                        "stability_counter": 1,
                    }
                ],
                "consensus_path": ["agent_0", "agent_1", "agent_2"],
                "weighted_votes": {"B": 1.75, "C": 0.7},
                "total_weight": 2.45,
                "rounds_used": 1,
                "consensus_reached": True,
            }
        }

