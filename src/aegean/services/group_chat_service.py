"""
GroupChatService for managing multi-agent group conversations.

Handles group creation, member management, message routing, and consensus execution.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from aegean.core.models import (
    Group,
    GroupMember,
    Message,
    GroupConsensusResult,
    CollaborationMode,
    Solution,
)
from aegean.core.agent import Agent, AgentRegistry
from aegean.core.coordinator import ConsensusCoordinator
from aegean.core.decision_engine import WeightedDecisionEngine


class GroupChatService:
    """
    Service for managing group chat and consensus execution.
    
    Features:
    - Create and manage agent groups
    - Add/remove group members
    - Send messages to groups
    - Execute consensus with weighted voting
    - Track conversation history
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        storage_backend: Optional[Any] = None
    ):
        """
        Initialize GroupChatService.
        
        Args:
            agent_registry: Registry of available agents
            storage_backend: Optional storage backend for persistence
        """
        self.agent_registry = agent_registry
        self.storage = storage_backend
        
        # In-memory storage (if no backend provided)
        self.groups: Dict[str, Group] = {}
        self.members: Dict[str, List[GroupMember]] = {}  # group_id -> members
        self.messages: Dict[str, List[Message]] = {}  # group_id -> messages
        self.consensus_results: Dict[str, GroupConsensusResult] = {}

    # ==================== Group Management ====================

    def create_group(
        self,
        group_name: str,
        created_by: str,
        description: Optional[str] = None,
        mode: CollaborationMode = CollaborationMode.CONSENSUS,
        metadata: Optional[Dict[str, Any]] = None,
        initial_members: Optional[List[Dict[str, Any]]] = None
    ) -> Group:
        """
        Create a new agent group with optional initial members.
        
        Args:
            group_name: Human-readable group name
            created_by: User ID who creates this group
            description: Optional group description
            mode: Collaboration mode (consensus/collaboration/hybrid)
            metadata: Optional metadata
            initial_members: Optional list of members to add on creation
                Each member dict should have: agent_id, and optionally: role, capability_weight, specialization
            
        Returns:
            Created Group object
        """
        group_id = f"group-{uuid.uuid4().hex[:8]}"
        
        group = Group(
            group_id=group_id,
            group_name=group_name,
            description=description,
            mode=mode,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        self.groups[group_id] = group
        self.members[group_id] = []
        self.messages[group_id] = []
        
        # Add initial members if provided
        if initial_members:
            for member_spec in initial_members:
                agent_id = member_spec.get("agent_id")
                if not agent_id:
                    continue
                try:
                    self.add_member(
                        group_id=group_id,
                        agent_id=agent_id,
                        role=member_spec.get("role"),
                        mode=member_spec.get("mode"),
                        capability_weight=member_spec.get("capability_weight", 1.0),
                        specialization=member_spec.get("specialization")
                    )
                except ValueError:
                    # Skip if agent already exists or other error
                    pass
        
        return group

    def get_group(self, group_id: str) -> Optional[Group]:
        """Get group by ID."""
        return self.groups.get(group_id)

    def list_groups(self, created_by: Optional[str] = None) -> List[Group]:
        """
        List all groups, optionally filtered by creator.
        
        Args:
            created_by: Optional user ID to filter by
            
        Returns:
            List of Group objects
        """
        groups = list(self.groups.values())
        
        if created_by:
            groups = [g for g in groups if g.created_by == created_by]
        
        return groups

    def delete_group(self, group_id: str) -> bool:
        """
        Delete a group and all its data.
        
        Args:
            group_id: Group to delete
            
        Returns:
            True if deleted, False if not found
        """
        if group_id not in self.groups:
            return False
        
        del self.groups[group_id]
        del self.members[group_id]
        del self.messages[group_id]
        
        return True

    # ==================== Member Management ====================

    def add_member(
        self,
        group_id: str,
        agent_id: str,
        role: Optional[str] = None,
        mode: Optional[CollaborationMode] = None,
        capability_weight: float = 1.0,
        specialization: Optional[Dict[str, float]] = None
    ) -> GroupMember:
        """
        Add an agent to a group.
        
        Args:
            group_id: Group to add to
            agent_id: Agent to add
            role: Optional role (e.g., 'analyst', 'reviewer')
            mode: Optional collaboration mode (defaults to group mode)
            capability_weight: Agent's capability weight (0.0-1.0)
            specialization: Domain -> proficiency mapping
            
        Returns:
            Created GroupMember object
            
        Raises:
            ValueError: If group not found or agent already in group
        """
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} not found")
        
        # Check if agent already in group
        existing = [m for m in self.members[group_id] if m.agent_id == agent_id]
        if existing:
            raise ValueError(f"Agent {agent_id} already in group {group_id}")
        
        # Get group mode if not specified
        if mode is None:
            mode = self.groups[group_id].mode
        
        member = GroupMember(
            group_id=group_id,
            agent_id=agent_id,
            role=role,
            mode=mode,
            capability_weight=capability_weight,
            specialization=specialization or {},
            is_active=True  # Explicitly set to True
        )
        
        self.members[group_id].append(member)
        
        return member

    def remove_member(self, group_id: str, agent_id: str) -> bool:
        """
        Remove an agent from a group.
        
        Args:
            group_id: Group to remove from
            agent_id: Agent to remove
            
        Returns:
            True if removed, False if not found
        """
        if group_id not in self.members:
            return False
        
        original_count = len(self.members[group_id])
        self.members[group_id] = [
            m for m in self.members[group_id] if m.agent_id != agent_id
        ]
        
        return len(self.members[group_id]) < original_count

    def get_members(self, group_id: str) -> List[GroupMember]:
        """Get all members of a group."""
        return self.members.get(group_id, [])

    def get_active_members(self, group_id: str) -> List[GroupMember]:
        """Get active members of a group."""
        members = self.members.get(group_id, [])
        return [m for m in members if m.is_active]

    # ==================== Message Management ====================

    def send_message(
        self,
        group_id: str,
        sender_id: str,
        sender_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Send a message to a group.
        
        Args:
            group_id: Group to send to
            sender_id: Sender ID (user or agent)
            sender_type: 'user' or 'agent'
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message object
            
        Raises:
            ValueError: If group not found
        """
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} not found")
        
        message_id = f"msg-{uuid.uuid4().hex[:8]}"
        
        message = Message(
            message_id=message_id,
            group_id=group_id,
            sender_id=sender_id,
            sender_type=sender_type,
            content=content,
            metadata=metadata or {}
        )
        
        self.messages[group_id].append(message)
        
        return message

    def get_messages(
        self,
        group_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages from a group.
        
        Args:
            group_id: Group to get messages from
            limit: Optional limit on number of messages
            
        Returns:
            List of Message objects (newest first)
        """
        messages = self.messages.get(group_id, [])
        messages = sorted(messages, key=lambda m: m.timestamp, reverse=True)
        
        if limit:
            messages = messages[:limit]
        
        return messages

    # ==================== Consensus Execution ====================

    def execute_consensus(
        self,
        group_id: str,
        task: str,
        message_id: Optional[str] = None,
        quorum_threshold: float = 0.5,
        stability_horizon: int = 2,
        max_rounds: int = 3
    ) -> GroupConsensusResult:
        """
        Execute consensus/collaboration with group members.
        
        Behavior depends on group mode:
        - CONSENSUS: All agents answer same question, weighted voting
        - COLLABORATION: Each agent works independently, no voting
        - HYBRID: Mix of both (agents first work independently, then consensus)
        
        Args:
            group_id: Group to execute with
            task: Task/question for consensus
            message_id: Optional message that triggered this
            quorum_threshold: Weighted quorum threshold (0.0-1.0)
            stability_horizon: Rounds to maintain stability
            max_rounds: Maximum refinement rounds
            
        Returns:
            GroupConsensusResult with outcome
            
        Raises:
            ValueError: If group not found or has no active members
        """
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} not found")
        
        group = self.groups[group_id]
        active_members = self.get_active_members(group_id)
        
        if not active_members:
            raise ValueError(f"Group {group_id} has no active members")
        
        # Get agents from registry
        agents = []
        for member in active_members:
            agent = self.agent_registry.get_agent(member.agent_id)
            if agent:
                agents.append(agent)
        
        if not agents:
            raise ValueError(f"No agents found for group {group_id}")
        
        start_time = datetime.now()
        consensus_id = f"consensus-{uuid.uuid4().hex[:8]}"
        
        # Handle different collaboration modes
        if group.mode == CollaborationMode.COLLABORATION:
            # Collaboration mode: each agent works independently, no voting
            result = await self._execute_collaboration(agents, task)
        elif group.mode == CollaborationMode.HYBRID:
            # Hybrid mode: agents work independently first, then consensus
            result = await self._execute_hybrid(
                agents, task, quorum_threshold, stability_horizon, max_rounds
            )
        else:
            # Consensus mode (default): all agents answer same question, weighted voting
            result = await self._execute_consensus_voting(
                agents, task, quorum_threshold, stability_horizon, max_rounds
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        group_result = GroupConsensusResult(
            consensus_id=consensus_id,
            group_id=group_id,
            message_id=message_id,
            mode=group.mode,
            success=result.get("success", False),
            final_solution=result.get("final_solution"),
            agent_responses=result.get("agent_responses", []),
            weighted_votes=result.get("weighted_votes"),
            total_weight=result.get("total_weight"),
            rounds_used=result.get("rounds_used", 1),
            participating_agents=[a.agent_id for a in agents],
            execution_time=execution_time,
            consensus_reached=result.get("consensus_reached", False),
            metadata=result.get("metadata", {})
        )
        
        # Store result
        self.consensus_results[consensus_id] = group_result
        
        return group_result

    async def _execute_collaboration(self, agents: List[Agent], task: str) -> Dict[str, Any]:
        """
        Collaboration mode: each agent works independently.
        No voting, just collect all responses.
        """
        agent_responses = []
        for agent in agents:
            try:
                solution = await agent.generate_solution(task)
                agent_responses.append(solution)
            except Exception as e:
                # Agent failed, skip
                pass
        
        return {
            "success": len(agent_responses) > 0,
            "agent_responses": agent_responses,
            "final_solution": agent_responses[0] if agent_responses else None,
            "rounds_used": 1,
            "consensus_reached": False,
            "metadata": {"mode": "collaboration", "agent_count": len(agents)}
        }

    async def _execute_hybrid(
        self,
        agents: List[Agent],
        task: str,
        quorum_threshold: float,
        stability_horizon: int,
        max_rounds: int
    ) -> Dict[str, Any]:
        """
        Hybrid mode: agents work independently first, then consensus on results.
        """
        # Phase 1: Independent work
        initial_responses = []
        for agent in agents:
            try:
                solution = await agent.generate_solution(task)
                initial_responses.append(solution)
            except Exception:
                pass
        
        if not initial_responses:
            return {
                "success": False,
                "agent_responses": [],
                "final_solution": None,
                "rounds_used": 0,
                "consensus_reached": False,
                "metadata": {"mode": "hybrid", "phase": "initial_work_failed"}
            }
        
        # Phase 2: Consensus on results
        decision_engine = WeightedDecisionEngine(
            quorum_threshold=quorum_threshold,
            stability_horizon=stability_horizon,
            agent_registry=self.agent_registry
        )
        
        coordinator = ConsensusCoordinator(
            agents=agents,
            decision_engine=decision_engine,
            max_rounds=max_rounds
        )
        
        result = coordinator.run_consensus(task)
        
        weighted_votes, total_weight = None, None
        if coordinator.state.solutions_history:
            weighted_votes, total_weight = decision_engine._calculate_weighted_votes(
                coordinator.state.solutions_history[0]
            )
        
        return {
            "success": result.success,
            "agent_responses": initial_responses,
            "final_solution": result.final_solution,
            "weighted_votes": weighted_votes,
            "total_weight": total_weight,
            "rounds_used": result.rounds_used,
            "consensus_reached": result.consensus_reached,
            "metadata": {"mode": "hybrid", "phase": "consensus"}
        }

    async def _execute_consensus_voting(
        self,
        agents: List[Agent],
        task: str,
        quorum_threshold: float,
        stability_horizon: int,
        max_rounds: int
    ) -> Dict[str, Any]:
        """
        Consensus mode: all agents answer same question, weighted voting.
        """
        decision_engine = WeightedDecisionEngine(
            quorum_threshold=quorum_threshold,
            stability_horizon=stability_horizon,
            agent_registry=self.agent_registry
        )
        
        coordinator = ConsensusCoordinator(
            agents=agents,
            decision_engine=decision_engine,
            max_rounds=max_rounds
        )
        
        result = coordinator.run_consensus(task)
        
        agent_responses = []
        if result.success and coordinator.state.solutions_history:
            agent_responses = coordinator.state.solutions_history[0]
        
        weighted_votes, total_weight = None, None
        if agent_responses:
            weighted_votes, total_weight = decision_engine._calculate_weighted_votes(
                agent_responses
            )
        
        return {
            "success": result.success,
            "agent_responses": agent_responses,
            "final_solution": result.final_solution,
            "weighted_votes": weighted_votes,
            "total_weight": total_weight,
            "rounds_used": result.rounds_used,
            "consensus_reached": result.consensus_reached,
            "metadata": {"mode": "consensus"}
        }

    def get_consensus_result(
        self,
        consensus_id: str
    ) -> Optional[GroupConsensusResult]:
        """Get consensus result by ID."""
        return self.consensus_results.get(consensus_id)

    def get_group_consensus_history(
        self,
        group_id: str,
        limit: Optional[int] = None
    ) -> List[GroupConsensusResult]:
        """
        Get consensus execution history for a group.
        
        Args:
            group_id: Group to get history for
            limit: Optional limit on number of results
            
        Returns:
            List of GroupConsensusResult objects (newest first)
        """
        results = [
            r for r in self.consensus_results.values()
            if r.group_id == group_id
        ]
        results = sorted(results, key=lambda r: r.timestamp, reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results

