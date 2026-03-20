"""
FastAPI endpoints for GroupChat functionality.

Provides REST API for:
- Group management
- Member management
- Message sending
- Consensus execution
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from aegean.core.models import (
    Group,
    GroupMember,
    Message,
    GroupConsensusResult,
    CollaborationMode,
)
from aegean.services.group_chat_service import GroupChatService
from aegean.core.agent import AgentRegistry


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


class ExecuteConsensusRequest(BaseModel):
    """Request to execute consensus."""
    task: str = Field(..., description="Task/question for consensus")
    message_id: Optional[str] = Field(None, description="Message that triggered this")
    quorum_threshold: float = Field(0.5, ge=0.0, le=1.0)
    stability_horizon: int = Field(2, ge=1)
    max_rounds: int = Field(3, ge=1)

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


# ==================== Group Management Endpoints ====================

@router.get("/agents", response_model=List[Dict[str, Any]])
async def list_available_agents(
    service: GroupChatService = Depends(get_service)
):
    """
    List all available agents that can be added to groups.
    
    Returns list of agent info with agent_id, role, and capabilities.
    """
    agents = service.agent_registry.get_all_agents()
    return [
        {
            "agent_id": agent.agent_id,
            "capability_weight": agent.capability_weight,
            "specialization": agent.specialization,
            "role": agent.role,
        }
        for agent in agents
    ]


@router.post("/", response_model=Group, status_code=201)
async def create_group(
    request: CreateGroupRequest,
    service: GroupChatService = Depends(get_service)
):
    """
    Create a new agent group with optional initial members.
    
    Returns the created Group object.
    """
    try:
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
    if not group:
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
    if not group:
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
        result = service.execute_consensus(
            group_id=group_id,
            task=request.task,
            message_id=request.message_id,
            quorum_threshold=request.quorum_threshold,
            stability_horizon=request.stability_horizon,
            max_rounds=request.max_rounds
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

