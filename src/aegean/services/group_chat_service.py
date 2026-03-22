"""
GroupChatService for managing multi-agent group conversations.

Handles group creation, member management, message routing, and consensus execution.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import re

from aegean.core.models import (
    Group,
    GroupMember,
    Message,
    GroupConsensusResult,
    CollaborationMode,
    Solution,
    RoundDiscussion,
    ConsensusConfig,
    KnowledgeGraph,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
)
from aegean.core.agent import Agent, AgentRegistry
from aegean.core.coordinator import ConsensusCoordinator
from aegean.core.decision_engine import WeightedDecisionEngine
from aegean.core.discussion_tracker import DiscussionTracker
from aegean.core.graph_extractor import RiskGraphBuilder


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
        
        # Track agent accuracy per group: group_id -> agent_id -> {total, correct}
        self.agent_accuracy: Dict[str, Dict[str, Dict[str, int]]] = {}

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

    async def execute_consensus(
        self,
        group_id: str,
        task: str,
        message_id: Optional[str] = None,
        quorum_threshold: float = 0.5,
        stability_horizon: int = 2,
        max_rounds: int = 3,
        risk_context: Optional[Dict[str, Any]] = None,
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
            risk_context: Optional risk context for knowledge graph extraction
            
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
        agent_ids = []
        for member in active_members:
            agent = self.agent_registry.get_agent(member.agent_id)
            if agent:
                agents.append(agent)
                agent_ids.append(member.agent_id)
        
        if not agents:
            raise ValueError(f"No agents found for group {group_id}")
        
        start_time = datetime.now()
        consensus_id = f"consensus-{uuid.uuid4().hex[:8]}"
        
        # Initialize discussion tracker
        discussion_tracker = DiscussionTracker(group_id, agent_ids)
        
        # Handle different collaboration modes
        if group.mode == CollaborationMode.COLLABORATION:
            # Collaboration mode: each agent works independently, no voting
            result = await self._execute_collaboration(agents, task)
        elif group.mode == CollaborationMode.HYBRID:
            # Hybrid mode: agents work independently first, then consensus
            result = await self._execute_hybrid(
                agents, task, quorum_threshold, stability_horizon, max_rounds,
                discussion_tracker
            )
        else:
            # Consensus mode (default): all agents answer same question, weighted voting
            result = await self._execute_consensus_voting(
                agents, task, quorum_threshold, stability_horizon, max_rounds,
                discussion_tracker
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Build agent graph from discussion
        agent_graph = discussion_tracker.build_agent_graph()
        
        # Build knowledge graph
        # Priority:
        # 1) risk_context provided by caller
        # 2) fallback: derive from multi-round discussion signals
        knowledge_graph = None
        if risk_context:
            graph_builder = RiskGraphBuilder()
            knowledge_graph = graph_builder.build_from_risk_context(**risk_context)
        else:
            knowledge_graph = self._build_knowledge_graph_from_discussion(
                consensus_id=consensus_id,
                group_id=group_id,
                discussion_rounds=result.get("discussion_rounds", []),
                agent_ids=agent_ids,
            )
        
        # Build response
        group_result = GroupConsensusResult(
            consensus_id=consensus_id,
            group_id=group_id,
            message_id=message_id,
            mode=group.mode,
            success=result.get("success", False),
            final_solution=result.get("final_solution"),
            agent_responses=result.get("agent_responses", []),
            discussion_rounds=result.get("discussion_rounds", []),
            consensus_path=discussion_tracker.get_consensus_path(),
            agent_graph=agent_graph,
            knowledge_graph=knowledge_graph,
            weighted_votes=result.get("weighted_votes"),
            total_weight=result.get("total_weight"),
            rounds_used=result.get("rounds_used", 1),
            participating_agents=agent_ids,
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
        max_rounds: int,
        discussion_tracker: DiscussionTracker,
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
                "discussion_rounds": [],
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

        registry = AgentRegistry()
        for agent in agents:
            registry.register_agent(agent)

        coordinator = ConsensusCoordinator(
            agent_registry=registry,
            config=ConsensusConfig(max_rounds=max_rounds, stability_horizon=stability_horizon),
            decision_engine=decision_engine,
        )

        result = await coordinator.run_consensus(task)
        
        discussion_rounds = []
        if result.success and coordinator.state.solutions_history:
            for round_num, solutions in enumerate(coordinator.state.solutions_history, 1):
                round_disc = discussion_tracker.record_round(
                    round_number=round_num,
                    agent_responses={s.agent_id: s for s in solutions},
                    candidate_answer=coordinator.state.candidate_solution.answer if coordinator.state.candidate_solution else None,
                    candidate_confidence=coordinator.state.candidate_solution.confidence if coordinator.state.candidate_solution else None,
                    stability_counter=coordinator.state.stability_counter,
                )
                discussion_rounds.append(round_disc)
        
        weighted_votes, total_weight = None, None
        if initial_responses:
            weighted_votes, total_weight = decision_engine._calculate_weighted_votes(
                initial_responses
            )
        
        return {
            "success": result.success,
            "agent_responses": initial_responses,
            "discussion_rounds": discussion_rounds,
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
        max_rounds: int,
        discussion_tracker: DiscussionTracker,
    ) -> Dict[str, Any]:
        """
        Consensus mode: all agents answer same question, weighted voting.
        """
        decision_engine = WeightedDecisionEngine(
            quorum_threshold=quorum_threshold,
            stability_horizon=stability_horizon,
            agent_registry=self.agent_registry
        )
        
        registry = AgentRegistry()
        for agent in agents:
            registry.register_agent(agent)

        coordinator = ConsensusCoordinator(
            agent_registry=registry,
            config=ConsensusConfig(max_rounds=max_rounds, stability_horizon=stability_horizon),
            decision_engine=decision_engine,
        )

        result = await coordinator.run_consensus(task)
        
        agent_responses = []
        discussion_rounds = []
        
        # Record discussion rounds
        if result.success and coordinator.state.solutions_history:
            for round_num, solutions in enumerate(coordinator.state.solutions_history, 1):
                agent_responses = solutions
                
                # Create round discussion record
                round_disc = discussion_tracker.record_round(
                    round_number=round_num,
                    agent_responses={s.agent_id: s for s in solutions},
                    candidate_answer=coordinator.state.candidate_solution.answer if coordinator.state.candidate_solution else None,
                    candidate_confidence=coordinator.state.candidate_solution.confidence if coordinator.state.candidate_solution else None,
                    stability_counter=coordinator.state.stability_counter,
                )
                discussion_rounds.append(round_disc)
        
        weighted_votes, total_weight = None, None
        if agent_responses:
            weighted_votes, total_weight = decision_engine._calculate_weighted_votes(
                agent_responses
            )
        
        return {
            "success": result.success,
            "agent_responses": agent_responses,
            "discussion_rounds": discussion_rounds,
            "final_solution": result.final_solution,
            "weighted_votes": weighted_votes,
            "total_weight": total_weight,
            "rounds_used": result.rounds_used,
            "consensus_reached": result.consensus_reached,
            "metadata": {"mode": "consensus"}
        }

    def _extract_stance_label(self, answer: str) -> str:
        """Extract a compact stance label from answer text."""
        if not answer:
            return "unknown"

        patterns = [
            r"\b([ABC])\b",
            r"\b(APPROVE|REVIEW|REJECT)\b",
            r"\b(同意|反对|弃权)\b",
        ]

        upper_answer = answer.upper()
        for pattern in patterns:
            match = re.search(pattern, upper_answer)
            if match:
                return match.group(1)

        compact = re.sub(r"\s+", " ", answer).strip()
        return compact[:24] if compact else "unknown"

    def _build_knowledge_graph_from_discussion(
        self,
        consensus_id: str,
        group_id: str,
        discussion_rounds: List[RoundDiscussion],
        agent_ids: List[str],
    ) -> Optional[KnowledgeGraph]:
        """Build knowledge graph from discussion rounds when no risk_context exists."""
        if not discussion_rounds:
            return None

        entities: List[KnowledgeGraphEntity] = []
        relations: List[KnowledgeGraphRelation] = []
        entity_ids = set()

        for agent_id in agent_ids:
            eid = f"agent_{agent_id}"
            entity_ids.add(eid)
            entities.append(
                KnowledgeGraphEntity(
                    entity_id=eid,
                    entity_type="agent",
                    name=agent_id,
                    attributes={"group_id": group_id},
                )
            )

        last_stance_by_agent: Dict[str, str] = {}

        for round_disc in discussion_rounds:
            round_entity_id = f"round_{round_disc.round_number}"
            if round_entity_id not in entity_ids:
                entity_ids.add(round_entity_id)
                entities.append(
                    KnowledgeGraphEntity(
                        entity_id=round_entity_id,
                        entity_type="round",
                        name=f"Round {round_disc.round_number}",
                        attributes={"status": round_disc.consensus_status},
                    )
                )

            for agent_id, solution in round_disc.agent_responses.items():
                stance = self._extract_stance_label(solution.answer)
                stance_entity_id = f"stance_{stance}"

                if stance_entity_id not in entity_ids:
                    entity_ids.add(stance_entity_id)
                    entities.append(
                        KnowledgeGraphEntity(
                            entity_id=stance_entity_id,
                            entity_type="stance",
                            name=stance,
                            attributes={},
                        )
                    )

                relations.append(
                    KnowledgeGraphRelation(
                        relation_id=f"rel_{len(relations)}",
                        source_entity_id=f"agent_{agent_id}",
                        target_entity_id=stance_entity_id,
                        relation_type="supports",
                        properties={"round": round_disc.round_number},
                    )
                )

                relations.append(
                    KnowledgeGraphRelation(
                        relation_id=f"rel_{len(relations)}",
                        source_entity_id=f"agent_{agent_id}",
                        target_entity_id=round_entity_id,
                        relation_type="participates_in",
                        properties={"round": round_disc.round_number},
                    )
                )

                if agent_id in last_stance_by_agent and last_stance_by_agent[agent_id] != stance:
                    relations.append(
                        KnowledgeGraphRelation(
                            relation_id=f"rel_{len(relations)}",
                            source_entity_id=f"agent_{agent_id}",
                            target_entity_id=stance_entity_id,
                            relation_type="changes_to",
                            properties={"round": round_disc.round_number},
                        )
                    )

                last_stance_by_agent[agent_id] = stance

        return KnowledgeGraph(
            graph_id=f"discussion_graph_{uuid.uuid4().hex[:8]}",
            source_type="discussion_rounds",
            source_id=consensus_id,
            entities=entities,
            relations=relations,
            metadata={
                "round_count": len(discussion_rounds),
                "agent_count": len(agent_ids),
            },
        )

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

    # ==================== Agent Accuracy Tracking ====================

    def get_agent_accuracy_stats(
        self,
        group_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get agent's accuracy statistics in a group.
        
        Args:
            group_id: Group ID
            agent_id: Agent ID
            
        Returns:
            Dict with total, correct, and accuracy
        """
        if group_id not in self.agent_accuracy:
            return {"total": 0, "correct": 0, "accuracy": 1.0}
        
        if agent_id not in self.agent_accuracy[group_id]:
            return {"total": 0, "correct": 0, "accuracy": 1.0}
        
        stats = self.agent_accuracy[group_id][agent_id]
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        
        accuracy = correct / total if total > 0 else 1.0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "last_updated": datetime.now().isoformat()
        }

    def update_agent_accuracy(
        self,
        group_id: str,
        agent_id: str,
        was_correct: bool
    ) -> None:
        """
        Update agent's accuracy based on feedback.
        
        Args:
            group_id: Group ID
            agent_id: Agent ID
            was_correct: Whether the agent's answer was correct
        """
        # Initialize group accuracy tracking if needed
        if group_id not in self.agent_accuracy:
            self.agent_accuracy[group_id] = {}
        
        # Initialize agent accuracy if needed
        if agent_id not in self.agent_accuracy[group_id]:
            self.agent_accuracy[group_id][agent_id] = {"total": 0, "correct": 0}
        
        # Update stats
        self.agent_accuracy[group_id][agent_id]["total"] += 1
        if was_correct:
            self.agent_accuracy[group_id][agent_id]["correct"] += 1

    # ==================== Auto Verification ====================

    async def auto_verify_consensus(
        self,
        group_id: str,
        consensus_id: str,
        correct_answer: str,
        verification_source: str = "user_input"
    ) -> Dict[str, Any]:
        """
        自动验证共识结果并更新历史准确率
        
        Args:
            group_id: Group ID
            consensus_id: Consensus ID
            correct_answer: The correct answer
            verification_source: Source of verification (auto/user_input/external_system)
            
        Returns:
            Dict with verification results and accuracy updates
        """
        # Get consensus result
        consensus_result = self.get_consensus_result(consensus_id)
        if not consensus_result:
            raise ValueError(f"Consensus {consensus_id} not found")
        
        # Compare each agent's answer with correct answer
        accuracy_updates = []
        for agent_response in consensus_result.agent_responses:
            was_correct = agent_response.answer == correct_answer
            
            # Update accuracy
            self.update_agent_accuracy(
                group_id=group_id,
                agent_id=agent_response.agent_id,
                was_correct=was_correct
            )
            
            # Get updated stats
            stats = self.get_agent_accuracy_stats(group_id, agent_response.agent_id)
            
            accuracy_updates.append({
                "agent_id": agent_response.agent_id,
                "was_correct": was_correct,
                "updated_accuracy": stats["accuracy"],
                "total_evaluations": stats["total"],
                "correct_count": stats["correct"]
            })
        
        return {
            "consensus_id": consensus_id,
            "correct_answer": correct_answer,
            "verification_source": verification_source,
            "agent_accuracy_updates": accuracy_updates,
            "timestamp": datetime.now().isoformat()
        }

    def get_agent_global_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        获取 Agent 的全局统计信息
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dict with global statistics
        """
        total_evaluations = 0
        total_correct = 0
        group_breakdown = []
        
        # Aggregate stats across all groups
        for group_id in self.agent_accuracy:
            if agent_id in self.agent_accuracy[group_id]:
                stats = self.agent_accuracy[group_id][agent_id]
                group_total = stats.get("total", 0)
                group_correct = stats.get("correct", 0)
                
                total_evaluations += group_total
                total_correct += group_correct
                
                if group_total > 0:
                    group_breakdown.append({
                        "group_id": group_id,
                        "accuracy": group_correct / group_total,
                        "evaluations": group_total,
                        "correct_count": group_correct
                    })
        
        global_accuracy = total_correct / total_evaluations if total_evaluations > 0 else 1.0
        
        return {
            "agent_id": agent_id,
            "global_accuracy": global_accuracy,
            "total_evaluations": total_evaluations,
            "correct_count": total_correct,
            "group_breakdown": group_breakdown,
            "last_updated": datetime.now().isoformat()
        }

    def update_member_weight(
        self,
        group_id: str,
        agent_id: str,
        capability_weight: float
    ) -> GroupMember:
        """
        更新 Agent 的能力权重
        
        Args:
            group_id: Group ID
            agent_id: Agent ID
            capability_weight: New capability weight (0.0-1.0)
            
        Returns:
            Updated GroupMember object
            
        Raises:
            ValueError: If agent not found in group
        """
        if not 0.0 <= capability_weight <= 1.0:
            raise ValueError("capability_weight must be between 0.0 and 1.0")
        
        members = self.get_members(group_id)
        member = next((m for m in members if m.agent_id == agent_id), None)
        
        if not member:
            raise ValueError(f"Agent {agent_id} not found in group {group_id}")
        
        # Update weight
        member.capability_weight = capability_weight
        
        return member

