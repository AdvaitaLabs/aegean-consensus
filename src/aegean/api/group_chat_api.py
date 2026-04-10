"""
FastAPI endpoints for GroupChat functionality.

Provides REST API for:
- Group management
- Member management
- Message sending
- Consensus execution
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from aegean.core import AgentRegistry
from aegean.core.models import (
    Group,
    GroupMember,
    Message,
    GroupConsensusResult,
    CollaborationMode,
)
from aegean.services.group_chat_service import GroupChatService


_setu_service = None


# ==================== Request/Response Models ====================

class InitialMemberSpec(BaseModel):
    """Specification for initial group member."""
    agent_id: str = Field(..., description="Agent to add")
    role: Optional[str] = Field(None, description="Agent's role")
    capability_weight: float = Field(1.0, ge=0.0, le=1.0)
    specialization: Optional[Dict[str, float]] = Field(None)


class CreateGroupRequest(BaseModel):
    """Request to create a new group."""
    group_name: str = Field(..., description="Human-readable group name")
    description: Optional[str] = Field(None, description="Group description")
    mode: CollaborationMode = Field(
        CollaborationMode.CONSENSUS,
        description="Collaboration mode"
    )
    created_by: str = Field(..., description="User ID creating this group")
    initial_members: Optional[List[InitialMemberSpec]] = Field(
        None,
        description="Optional list of agents to add on creation"
    )
    metadata: Optional[Dict[str, Any]] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "group_name": "Financial Risk Team",
                "description": "Team for credit assessment",
                "mode": "consensus",
                "created_by": "user_123",
                "initial_members": [
                    {"agent_id": "agent_0", "role": "credit_analyst", "capability_weight": 1.0},
                    {"agent_id": "agent_1", "role": "fraud_analyst", "capability_weight": 0.95},
                    {"agent_id": "agent_2", "role": "compliance_officer", "capability_weight": 0.9},
                ]
            }
        }


class AddMemberRequest(BaseModel):
    """Request to add a member to a group."""
    agent_id: str = Field(..., description="Agent to add")
    role: Optional[str] = Field(None, description="Agent's role")
    mode: Optional[CollaborationMode] = Field(None, description="Collaboration mode")
    capability_weight: float = Field(1.0, ge=0.0, le=1.0)
    specialization: Optional[Dict[str, float]] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "agent_0",
                "role": "credit_analyst",
                "capability_weight": 1.0,
                "specialization": {"credit": 0.95, "fraud": 0.85},
            }
        }


class SendMessageRequest(BaseModel):
    """Request to send a message to a group."""
    sender_id: str = Field(..., description="Sender ID")
    sender_type: str = Field(..., description="'user' or 'agent'")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "sender_id": "user_123",
                "sender_type": "user",
                "content": "Evaluate customer credit rating",
            }
        }


class RiskContextRequest(BaseModel):
    """Optional risk context for knowledge graph extraction."""
    subject_id: str = Field(..., description="Subject ID (user, company, etc)")
    subject_type: str = Field("user", description="Subject type")
    action_type: str = Field(..., description="Type of action (payment, transfer, etc)")
    description: str = Field(..., description="Action description")
    amount: Optional[float] = Field(None, description="Transaction amount")
    currency: str = Field("USD", description="Currency code")
    geo_location: Optional[str] = Field(None, description="Geographic location")
    counterparty_id: Optional[str] = Field(None, description="Counterparty ID")
    trace_context: Optional[str] = Field(None, description="Additional context")


class ExecuteConsensusRequest(BaseModel):
    """Request to execute consensus."""
    task: str = Field(..., description="Task/question for consensus")
    message_id: Optional[str] = Field(None, description="Message that triggered this")
    quorum_threshold: float = Field(0.5, ge=0.0, le=1.0)
    stability_horizon: int = Field(2, ge=1)
    max_rounds: int = Field(3, ge=1)
    risk_context: Optional[RiskContextRequest] = Field(
        None,
        description="Optional risk context for knowledge graph extraction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task": "What is the customer's credit rating?",
                "quorum_threshold": 0.5,
                "stability_horizon": 2,
                "max_rounds": 3,
            }
        }


# ==================== API Router ====================

router = APIRouter(prefix="/api/v1/groups", tags=["GroupChat"])

# Global service instance (will be initialized by app)
_service: Optional[GroupChatService] = None


def get_service() -> GroupChatService:
    """Dependency to get GroupChatService instance."""
    if _service is None:
        raise HTTPException(
            status_code=500,
            detail="GroupChatService not initialized"
        )
    return _service


def init_service(agent_registry: AgentRegistry, storage_backend=None):
    """Initialize the global service instance."""
    global _service
    _service = GroupChatService(agent_registry, storage_backend)


def bind_setu_service(setu_service) -> None:
    """Bind Setu adapter for exclusive-group protection in generic group APIs."""
    global _setu_service
    _setu_service = setu_service


def _is_hidden_setu_group(group: Group) -> bool:
    return bool(_setu_service and _setu_service.is_setu_bound_group(group.group_id))


def _assert_group_not_reserved(group_id: str) -> None:
    if _setu_service:
        _setu_service.assert_not_setu_bound_group(group_id)


# ==================== Group Management Endpoints ====================

@router.get("/agents", response_model=List[Dict[str, Any]])
async def list_available_agents(
    service: GroupChatService = Depends(get_service)
):
    """
    List all available agents that can be added to groups.

    Returns agent base profile + global historical stats.
    """
    agents = service.agent_registry.get_all_agents()
    result = []

    for agent in agents:
        global_stats = service.get_agent_global_stats(agent.agent_id)
        result.append(
            {
                "agent_id": agent.agent_id,
                "capability_weight": agent.capability_weight,
                "specialization": agent.specialization,
                "role": agent.role,
                "historical_accuracy": global_stats.get("global_accuracy", 1.0),
                "total_evaluations": global_stats.get("total_evaluations", 0),
                "correct_count": global_stats.get("correct_count", 0),
                "group_breakdown": global_stats.get("group_breakdown", []),
                "last_updated": global_stats.get("last_updated"),
            }
        )

    return result


@router.post("/", response_model=Group, status_code=201)
async def create_group(
    request: CreateGroupRequest,
    allow_reserved_setu_group: bool = Query(False, description="Internal escape hatch for Setu bootstrap only"),
    service: GroupChatService = Depends(get_service)
):
    """
    Create a new agent group with optional initial members.
    
    Returns the created Group object.
    """
    try:
        if not allow_reserved_setu_group and request.metadata:
            if (
                request.metadata.get("integration") == "setu"
                and request.metadata.get("binding_type") == "exclusive"
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Reserved Setu-exclusive groups cannot be created through general group APIs"
                )

        # Convert initial_members to dict format for service
        initial_members = None
        if request.initial_members:
            initial_members = [m.dict() for m in request.initial_members]
        
        group = service.create_group(
            group_name=request.group_name,
            created_by=request.created_by,
            description=request.description,
            mode=request.mode,
            metadata=request.metadata,
            initial_members=initial_members
        )
        return group
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{group_id}", response_model=Group)
async def get_group(
    group_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get a group by ID.
    
    Returns the Group object or 404 if not found.
    """
    group = service.get_group(group_id)
    if not group or _is_hidden_setu_group(group):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    return group


@router.get("/", response_model=List[Group])
async def list_groups(
    created_by: Optional[str] = None,
    service: GroupChatService = Depends(get_service)
):
    """
    List all groups, optionally filtered by creator.
    
    Returns list of Group objects.
    """
    groups = service.list_groups(created_by=created_by)
    groups = [g for g in groups if not _is_hidden_setu_group(g)]
    return groups


@router.delete("/{group_id}", status_code=204)
async def delete_group(
    group_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Delete a group and all its data.
    
    Returns 204 on success, 404 if not found.
    """
    _assert_group_not_reserved(group_id)
    success = service.delete_group(group_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    return None


# ==================== Member Management Endpoints ====================

@router.post("/{group_id}/members", response_model=GroupMember, status_code=201)
async def add_member(
    group_id: str,
    request: AddMemberRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    Add an agent to a group.
    
    Returns the created GroupMember object.
    """
    try:
        _assert_group_not_reserved(group_id)
        member = service.add_member(
            group_id=group_id,
            agent_id=request.agent_id,
            role=request.role,
            mode=request.mode,
            capability_weight=request.capability_weight,
            specialization=request.specialization
        )
        return member
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{group_id}/members/{agent_id}", status_code=204)
async def remove_member(
    group_id: str,
    agent_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Remove an agent from a group.
    
    Returns 204 on success, 404 if not found.
    """
    _assert_group_not_reserved(group_id)
    success = service.remove_member(group_id, agent_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Member {agent_id} not found in group {group_id}"
        )
    return None


@router.get("/{group_id}/members", response_model=List[GroupMember])
async def get_members(
    group_id: str,
    active_only: bool = False,
    service: GroupChatService = Depends(get_service)
):
    """
    Get all members of a group.
    
    Args:
        group_id: Group ID
        active_only: If True, only return active members
        
    Returns list of GroupMember objects with is_active status.
    """
    # Verify group exists
    group = service.get_group(group_id)
    if not group or _is_hidden_setu_group(group):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    
    if active_only:
        members = service.get_active_members(group_id)
    else:
        members = service.get_members(group_id)
    return members


# ==================== Message Management Endpoints ====================

@router.post("/{group_id}/messages", response_model=Message, status_code=201)
async def send_message(
    group_id: str,
    request: SendMessageRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    Send a message to a group.
    
    Returns the created Message object.
    """
    try:
        _assert_group_not_reserved(group_id)
        message = service.send_message(
            group_id=group_id,
            sender_id=request.sender_id,
            sender_type=request.sender_type,
            content=request.content,
            metadata=request.metadata
        )
        return message
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{group_id}/messages", response_model=List[Message])
async def get_messages(
    group_id: str,
    limit: Optional[int] = None,
    service: GroupChatService = Depends(get_service)
):
    """
    Get messages from a group.
    
    Args:
        group_id: Group ID
        limit: Optional limit on number of messages
        
    Returns list of Message objects (newest first).
    """
    group = service.get_group(group_id)
    if not group or _is_hidden_setu_group(group):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    messages = service.get_messages(group_id, limit=limit)
    return messages


# ==================== Consensus Execution Endpoints ====================

@router.post(
    "/{group_id}/consensus",
    response_model=GroupConsensusResult,
    status_code=201
)
async def execute_consensus(
    group_id: str,
    request: ExecuteConsensusRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    Execute consensus with group members.
    
    Returns GroupConsensusResult with consensus outcome.
    """
    try:
        _assert_group_not_reserved(group_id)
        risk_context_dict = None
        if request.risk_context:
            risk_context_dict = request.risk_context.dict(exclude_none=True)

        result = await service.execute_consensus(
            group_id=group_id,
            task=request.task,
            message_id=request.message_id,
            quorum_threshold=request.quorum_threshold,
            stability_horizon=request.stability_horizon,
            max_rounds=request.max_rounds,
            risk_context=risk_context_dict,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{group_id}/consensus/history",
    response_model=List[GroupConsensusResult]
)
async def get_consensus_history(
    group_id: str,
    limit: Optional[int] = None,
    service: GroupChatService = Depends(get_service)
):
    """
    Get consensus execution history for a group.
    
    Args:
        group_id: Group ID
        limit: Optional limit on number of results
        
    Returns list of GroupConsensusResult objects (newest first).
    """
    results = service.get_group_consensus_history(group_id, limit=limit)
    return results


@router.get("/consensus/{consensus_id}", response_model=GroupConsensusResult)
async def get_consensus_result(
    consensus_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get a specific consensus result by ID.
    
    Returns GroupConsensusResult or 404 if not found.
    """
    result = service.get_consensus_result(consensus_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Consensus result {consensus_id} not found"
        )
    return result


# ==================== Auto Verification & Weight Management ====================

class AutoVerifyRequest(BaseModel):
    """Request to auto-verify consensus result."""
    correct_answer: str = Field(..., description="The correct answer")
    verification_source: str = Field(
        "user_input",
        description="Source of verification: auto, user_input, or external_system"
    )


class UpdateMemberRequest(BaseModel):
    """Request to update member weight."""
    capability_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="New capability weight (0.0-1.0)"
    )


@router.post("/{group_id}/consensus/{consensus_id}/auto-verify", status_code=200)
async def auto_verify_consensus(
    group_id: str,
    consensus_id: str,
    request: AutoVerifyRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    自动验证共识结果并更新历史准确率
    
    根据正确答案自动比较每个 Agent 的答案，更新其准确率。
    
    Returns:
        Dict with verification results and accuracy updates
    """
    try:
        result = await service.auto_verify_consensus(
            group_id=group_id,
            consensus_id=consensus_id,
            correct_answer=request.correct_answer,
            verification_source=request.verification_source
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/global-stats", response_model=Dict[str, Any])
async def get_agent_global_stats(
    agent_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    获取 Agent 的全局统计信息
    
    包括全局准确率、总评估次数、各群组的分布情况。
    
    Returns:
        Dict with global statistics
    """
    try:
        stats = service.get_agent_global_stats(agent_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{group_id}/members/{agent_id}", response_model=GroupMember)
async def update_member_weight(
    group_id: str,
    agent_id: str,
    request: UpdateMemberRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    更新 Agent 的能力权重
    
    根据历史准确率或其他因素调整 Agent 的能力权重。
    
    Returns:
        Updated GroupMember object
    """
    try:
        member = service.update_member_weight(
            group_id=group_id,
            agent_id=agent_id,
            capability_weight=request.capability_weight
        )
        return member
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Agent Weight & Stats Endpoints ====================

@router.get("/{group_id}/agent-stats/{agent_id}", response_model=Dict[str, Any])
async def get_agent_stats(
    group_id: str,
    agent_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get agent's weight statistics in a group.
    
    Returns:
    - capability_weight: Agent's base capability (0.0-1.0)
    - historical_accuracy: Agent's accuracy from past evaluations
    - total_evaluations: Number of times this agent was evaluated
    - correct_count: Number of correct evaluations
    - last_updated: When the stats were last updated
    """
    # Verify group exists
    group = service.get_group(group_id)
    if not group or _is_hidden_setu_group(group):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    
    # Get member info
    members = service.get_members(group_id)
    member = next((m for m in members if m.agent_id == agent_id), None)
    
    if not member:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found in group {group_id}"
        )
    
    # Get historical accuracy from decision engine
    stats = service.get_agent_accuracy_stats(group_id, agent_id)
    
    return {
        "agent_id": agent_id,
        "group_id": group_id,
        "capability_weight": member.capability_weight,
        "total_evaluations": stats.get("total", 0),
        "correct_count": stats.get("correct", 0),
        "historical_accuracy": stats.get("accuracy", 1.0),
        "last_updated": stats.get("last_updated", None)
    }


class AgentFeedbackRequest(BaseModel):
    """Request to provide feedback on agent's answer."""
    agent_id: str = Field(..., description="Agent ID")
    consensus_id: str = Field(..., description="Consensus ID that produced the answer")
    was_correct: bool = Field(..., description="Whether the agent's answer was correct")
    feedback_notes: Optional[str] = Field(None, description="Optional feedback notes")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "agent_0",
                "consensus_id": "consensus-xxx",
                "was_correct": True,
                "feedback_notes": "Correctly identified the fraud pattern"
            }
        }


@router.post("/{group_id}/agent-feedback", status_code=200)
async def submit_agent_feedback(
    group_id: str,
    request: AgentFeedbackRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    Submit feedback on an agent's answer to update historical accuracy.
    
    This updates the agent's historical accuracy score which affects
    their voting weight in future consensus rounds.
    
    Returns updated agent stats.
    """
    # Verify group exists
    group = service.get_group(group_id)
    if not group or _is_hidden_setu_group(group):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    
    # Verify agent is in group
    members = service.get_members(group_id)
    member = next((m for m in members if m.agent_id == request.agent_id), None)
    
    if not member:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {request.agent_id} not found in group {group_id}"
        )
    
    # Update agent accuracy
    try:
        service.update_agent_accuracy(
            group_id=group_id,
            agent_id=request.agent_id,
            was_correct=request.was_correct
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Return updated stats
    stats = service.get_agent_accuracy_stats(group_id, request.agent_id)
    
    return {
        "agent_id": request.agent_id,
        "group_id": group_id,
        "capability_weight": member.capability_weight,
        "total_evaluations": stats.get("total", 0),
        "correct_count": stats.get("correct", 0),
        "historical_accuracy": stats.get("accuracy", 1.0),
        "feedback_submitted": True,
        "feedback_notes": request.feedback_notes
    }


# ==================== Graph Visualization Endpoints ====================

@router.get("/consensus/{consensus_id}/agent-graph", response_model=Dict[str, Any])
async def get_agent_graph(
    consensus_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get the agent relationship graph for a consensus execution.

    Returns nodes (agents) and edges (influence relationships) in
    D3.js / Cytoscape-compatible format.

    Response shape:
    {
      "nodes": [{"id": "agent_0", "influence_score": 0.8, ...}],
      "edges": [{"source": "agent_0", "target": "agent_1", "influence_weight": 0.6, ...}],
      "key_agents": [["agent_0", 0.8], ...]
    }
    """
    result = service.get_consensus_result(consensus_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Consensus {consensus_id} not found")

    if not result.agent_graph:
        return {"nodes": [], "edges": [], "key_agents": []}

    graph = result.agent_graph
    nodes = [
        {
            "id": node_id,
            "label": node_id,
        }
        for node_id in graph.nodes
    ]
    edges = [
        {
            "source": edge.source_agent_id,
            "target": edge.target_agent_id,
            "influence_weight": edge.influence_weight,
            "trust_score": edge.trust_score,
            "agreement_count": edge.agreement_count,
            "disagreement_count": edge.disagreement_count,
        }
        for edge in graph.edges
        # exclude self-loops from the edge list (they are shown via key_agents)
        if edge.source_agent_id != edge.target_agent_id
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "key_agents": graph.get_key_agents(),
    }


@router.get("/consensus/{consensus_id}/knowledge-graph", response_model=Dict[str, Any])
async def get_knowledge_graph(
    consensus_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get the knowledge graph extracted from the risk context of a consensus.

    Only populated when the consensus was executed with a risk_context.

    Response shape:
    {
      "graph_id": "...",
      "source_type": "risk_request",
      "nodes": [{"id": "...", "label": "...", "type": "...", "attributes": {...}}],
      "links": [{"source": "...", "target": "...", "type": "...", "properties": {...}}]
    }
    """
    from aegean.core.graph_extractor import RiskGraphBuilder

    result = service.get_consensus_result(consensus_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Consensus {consensus_id} not found")

    if not result.knowledge_graph:
        return {"graph_id": None, "source_type": None, "nodes": [], "links": []}

    builder = RiskGraphBuilder()
    return builder.visualize_for_ui(result.knowledge_graph)


@router.get("/consensus/{consensus_id}/discussion", response_model=Dict[str, Any])
async def get_discussion_rounds(
    consensus_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get the full discussion process for a consensus execution.

    Returns all rounds with each agent's answer, confidence, and reasoning,
    plus the consensus path showing how agreement formed.

    Response shape:
    {
      "consensus_id": "...",
      "total_rounds": 2,
      "consensus_path": ["agent_0", "agent_1", "agent_2"],
      "rounds": [
        {
          "round_number": 1,
          "consensus_status": "forming",
          "candidate_answer": "BBB",
          "stability_counter": 1,
          "agent_responses": {
            "agent_0": {"answer": "BBB", "confidence": 0.85, "reasoning": "..."},
            ...
          }
        },
        ...
      ]
    }
    """
    result = service.get_consensus_result(consensus_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Consensus {consensus_id} not found")

    rounds = []
    for rd in result.discussion_rounds:
        rounds.append({
            "round_number": rd.round_number,
            "consensus_status": rd.consensus_status,
            "candidate_answer": rd.candidate_answer,
            "candidate_confidence": rd.candidate_confidence,
            "stability_counter": rd.stability_counter,
            "agent_responses": {
                agent_id: {
                    "answer": sol.answer,
                    "confidence": sol.confidence,
                    "reasoning": sol.reasoning,
                }
                for agent_id, sol in rd.agent_responses.items()
            },
        })

    return {
        "consensus_id": consensus_id,
        "total_rounds": len(rounds),
        "consensus_reached": result.consensus_reached,
        "final_answer": result.final_solution.answer if result.final_solution else None,
        "consensus_path": result.consensus_path,
        "rounds": rounds,
    }


@router.get("/{group_id}/weights-summary", response_model=Dict[str, Any])
async def get_group_weights_summary(
    group_id: str,
    service: GroupChatService = Depends(get_service)
):
    """
    Get a summary of all agents' weights in a group.
    
    Useful for understanding the voting power distribution.
    """
    # Verify group exists
    group = service.get_group(group_id)
    if not group or _is_hidden_setu_group(group):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    
    members = service.get_members(group_id)
    
    agent_weights = []
    for member in members:
        stats = service.get_agent_accuracy_stats(group_id, member.agent_id)
        
        # Calculate current voting weight
        capability_weight = member.capability_weight
        historical_accuracy = stats.get("accuracy", 1.0)
        # Note: confidence varies per answer, so we show the components
        
        agent_weights.append({
            "agent_id": member.agent_id,
            "role": member.role,
            "capability_weight": capability_weight,
            "historical_accuracy": historical_accuracy,
            "total_evaluations": stats.get("total", 0),
            "correct_count": stats.get("correct", 0),
            "is_active": member.is_active,
            "note": f"Voting weight = capability_weight ({capability_weight}) × confidence (varies) × historical_accuracy ({historical_accuracy})"
        })
    
    return {
        "group_id": group_id,
        "group_name": group.group_name,
        "mode": group.mode.value,
        "agent_count": len(members),
        "agents": agent_weights
    }

