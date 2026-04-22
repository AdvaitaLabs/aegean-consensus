"""
Global Memory System - Unified interface for knowledge and experience.

Combines:
- Knowledge Base (static knowledge via RAG)
- Experience Base (historical decisions and feedback)

Provides:
- Unified context retrieval
- Prompt enhancement
- Learning from feedback
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from aegean.memory.knowledge_base import KnowledgeBase, Document, RetrievalResult, KnowledgeQuery
from aegean.memory.experience_base import ExperienceBase, ConsensusRecord, AgentPerformance


@dataclass
class MemoryContext:
    """
    Unified memory context for prompt enhancement.
    
    Combines knowledge base documents and historical cases.
    """
    # Knowledge base results
    knowledge_documents: List[Document]
    knowledge_scores: List[float]
    
    # Historical cases
    similar_cases: List[ConsensusRecord]
    
    # Agent performance
    agent_performance: Dict[str, AgentPerformance]

    # Group shared context
    group_id: Optional[str] = None
    group_skills: List[str] = None
    group_graph_ids: List[str] = None
    
    # Statistics
    total_knowledge_docs: int = 0
    total_similar_cases: int = 0
    retrieval_time: float = 0.0
    
    def format_for_prompt(self, max_docs: int = 3, max_cases: int = 2) -> str:
        """
        Format memory context for prompt enhancement.
        
        Args:
            max_docs: Maximum knowledge documents to include
            max_cases: Maximum historical cases to include
            
        Returns:
            Formatted string for prompt
        """
        sections = []
        
        # Knowledge base section
        if self.knowledge_documents:
            sections.append("【组织知识库】")
            for i, doc in enumerate(self.knowledge_documents[:max_docs]):
                sections.append(f"\n{i+1}. [{doc.category}] {doc.content[:200]}...")
        
        # Historical cases section
        if self.similar_cases:
            sections.append("\n\n【历史案例】")
            for i, case in enumerate(self.similar_cases[:max_cases]):
                sections.append(
                    f"\n{i+1}. 任务: {case.task[:100]}...\n"
                    f"   结果: {case.final_answer}\n"
                    f"   轮数: {case.rounds_used}, "
                    f"   成功: {'是' if case.consensus_reached else '否'}"
                )
        
        # Agent performance section (optional)
        if self.agent_performance:
            sections.append("\n\n【团队表现】")
            for agent_id, perf in list(self.agent_performance.items())[:3]:
                sections.append(
                    f"\n- {agent_id}: 准确率 {perf.accuracy:.1%}, "
                    f"参与 {perf.total_participations} 次"
                )

        if self.group_id:
            sections.append(f"\n\n【Group上下文】\n- Group ID: {self.group_id}")
            if self.group_skills:
                sections.append(f"- Group Skills: {', '.join(self.group_skills[:8])}")
            if self.group_graph_ids:
                sections.append(f"- Knowledge Graph IDs: {', '.join(self.group_graph_ids[:5])}")
        
        return "\n".join(sections)


class GlobalMemorySystem:
    """
    Global memory system combining knowledge and experience.
    
    Features:
    - Unified context retrieval
    - Prompt enhancement with RAG
    - Learning from feedback
    - Agent performance tracking
    - Historical case-based reasoning
    
    Architecture:
    - Knowledge Base: Static knowledge (regulations, best practices)
    - Experience Base: Dynamic learning (historical decisions, feedback)
    """
    
    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        experience_base: Optional[ExperienceBase] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize global memory system.
        
        Args:
            knowledge_base: KnowledgeBase instance (creates default if None)
            experience_base: ExperienceBase instance (creates default if None)
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize components
        self.knowledge_base = knowledge_base or KnowledgeBase(
            backend=self.config.get("knowledge_backend", "memory"),
            embedding_model=self.config.get("embedding_model"),
            config=self.config.get("knowledge_config", {})
        )
        
        self.experience_base = experience_base or ExperienceBase(
            backend=self.config.get("experience_backend", "memory"),
            config=self.config.get("experience_config", {})
        )
    
    async def retrieve_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        include_knowledge: bool = True,
        include_cases: bool = True,
        include_performance: bool = True,
        group_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        group_context: Optional[Dict[str, Any]] = None,
    ) -> MemoryContext:
        """
        Retrieve unified memory context for a query.
        
        Args:
            query: Query/task description
            user_id: Optional user ID for personalization
            category: Optional category filter for knowledge
            include_knowledge: Whether to retrieve knowledge documents
            include_cases: Whether to retrieve similar cases
            include_performance: Whether to include agent performance
            
        Returns:
            MemoryContext with all relevant information
        """
        import time
        start_time = time.time()
        
        # Retrieve knowledge documents
        knowledge_docs = []
        knowledge_scores = []
        if include_knowledge:
            kb_result = await self.knowledge_base.retrieve(
                query=query,
                top_k=self.config.get("knowledge_top_k", 5),
                category=category,
                min_score=self.config.get("knowledge_min_score", 0.5),
                group_id=group_id,
                categories=categories,
                metadata_filters=metadata_filters,
            )
            knowledge_docs = kb_result.documents
            knowledge_scores = kb_result.scores
        
        # Retrieve similar historical cases
        similar_cases = []
        if include_cases:
            similar_cases = await self.experience_base.get_similar_cases(
                task=query,
                top_k=self.config.get("cases_top_k", 3),
                min_similarity=self.config.get("cases_min_similarity", 0.3)
            )
        
        # Get agent performance metrics
        agent_performance = {}
        if include_performance:
            # Get all agents from recent cases
            recent_records = await self.experience_base.query_consensus_history(limit=100)
            agent_ids = set()
            for record in recent_records:
                agent_ids.update(record.participating_agents)
            
            # Fetch performance for each agent
            for agent_id in agent_ids:
                perf = await self.experience_base.get_agent_performance(agent_id)
                if perf:
                    agent_performance[agent_id] = perf
        
        retrieval_time = time.time() - start_time
        
        group_skills: List[str] = []
        group_graph_ids: List[str] = []
        if group_context:
            group_skills = list(group_context.get("skills", []))
            group_graph_ids = list(group_context.get("knowledge_graph_ids", []))
        
        return MemoryContext(
            knowledge_documents=knowledge_docs,
            knowledge_scores=knowledge_scores,
            similar_cases=similar_cases,
            agent_performance=agent_performance,
            group_id=group_id,
            group_skills=group_skills,
            group_graph_ids=group_graph_ids,
            total_knowledge_docs=len(knowledge_docs),
            total_similar_cases=len(similar_cases),
            retrieval_time=retrieval_time
        )
    
    async def enhance_prompt(
        self,
        base_prompt: str,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Enhance a prompt with memory context.
        
        Args:
            base_prompt: Base prompt template
            query: Current query/task
            user_id: Optional user ID
            category: Optional category filter
            
        Returns:
            Enhanced prompt with memory context
        """
        # Retrieve context
        context = await self.retrieve_context(
            query=query,
            user_id=user_id,
            category=category
        )
        
        # Format context
        context_str = context.format_for_prompt(
            max_docs=self.config.get("prompt_max_docs", 3),
            max_cases=self.config.get("prompt_max_cases", 2)
        )
        
        # Combine with base prompt
        if context_str:
            enhanced_prompt = f"{context_str}\n\n{base_prompt}"
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt
    
    async def store_consensus_result(
        self,
        consensus_id: str,
        task: str,
        final_answer: Optional[str],
        rounds_used: int,
        consensus_reached: bool,
        participating_agents: List[str],
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a consensus execution result.
        
        Args:
            consensus_id: Unique consensus ID
            task: Task description
            final_answer: Final consensus answer
            rounds_used: Number of rounds used
            consensus_reached: Whether consensus was reached
            participating_agents: List of agent IDs
            execution_time: Total execution time
            metadata: Additional metadata
            
        Returns:
            Consensus ID
        """
        record = ConsensusRecord(
            consensus_id=consensus_id,
            task=task,
            final_answer=final_answer,
            rounds_used=rounds_used,
            consensus_reached=consensus_reached,
            participating_agents=participating_agents,
            execution_time=execution_time,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        return await self.experience_base.store_consensus(record)
    
    async def learn_from_feedback(
        self,
        consensus_id: str,
        was_correct: bool,
        feedback_source: str = "user",
        comments: Optional[str] = None
    ) -> str:
        """
        Learn from feedback on a consensus result.
        
        Updates agent performance metrics based on feedback.
        
        Args:
            consensus_id: Consensus ID to provide feedback on
            was_correct: Whether the consensus result was correct
            feedback_source: Source of feedback ('user', 'system', 'ground_truth')
            comments: Optional comments
            
        Returns:
            Feedback ID
        """
        from aegean.memory.experience_base import FeedbackRecord
        import uuid
        
        feedback = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            consensus_id=consensus_id,
            was_correct=was_correct,
            feedback_source=feedback_source,
            comments=comments,
            timestamp=datetime.utcnow()
        )
        
        return await self.experience_base.store_feedback(feedback)
    
    async def add_knowledge(
        self,
        content: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to knowledge base.
        
        Args:
            content: Document content
            category: Document category
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        return await self.knowledge_base.add_document(
            content=content,
            category=category,
            metadata=metadata
        )
    
    async def get_agent_insights(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get insights about an agent's performance.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary with agent insights
        """
        perf = await self.experience_base.get_agent_performance(agent_id)
        
        if not perf:
            return {
                "agent_id": agent_id,
                "status": "no_data",
                "message": "No performance data available"
            }
        
        # Calculate insights
        insights = {
            "agent_id": agent_id,
            "status": "active",
            "metrics": {
                "accuracy": perf.accuracy,
                "total_participations": perf.total_participations,
                "correct_answers": perf.correct_answers,
                "avg_confidence": perf.avg_confidence,
                "avg_response_time": perf.avg_response_time
            },
            "rating": self._calculate_rating(perf),
            "recommendations": self._generate_recommendations(perf),
            "last_updated": perf.last_updated.isoformat()
        }
        
        return insights
    
    async def get_system_statistics(
        self,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get overall system statistics.
        
        Args:
            time_range_days: Number of days to look back
            
        Returns:
            Dictionary with system statistics
        """
        from datetime import timedelta
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=time_range_days)
        
        # Get consensus statistics
        consensus_stats = await self.experience_base.get_statistics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get knowledge base statistics
        kb_stats = self.knowledge_base.get_stats()
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": time_range_days
            },
            "consensus": consensus_stats,
            "knowledge_base": kb_stats,
            "memory_system": {
                "knowledge_backend": self.knowledge_base.backend,
                "experience_backend": self.experience_base.backend
            }
        }
    
    # ==================== Helper Methods ====================
    
    def _calculate_rating(self, perf: AgentPerformance) -> str:
        """
        Calculate agent rating based on performance.
        
        Returns:
            Rating string ('excellent', 'good', 'average', 'poor')
        """
        if perf.total_participations < 5:
            return "insufficient_data"
        
        if perf.accuracy >= 0.9:
            return "excellent"
        elif perf.accuracy >= 0.75:
            return "good"
        elif perf.accuracy >= 0.6:
            return "average"
        else:
            return "poor"
    
    def _generate_recommendations(self, perf: AgentPerformance) -> List[str]:
        """Generate recommendations based on agent performance."""
        recommendations = []
        
        if perf.total_participations < 10:
            recommendations.append("需要更多数据以提供准确评估")
        
        if perf.accuracy < 0.6:
            recommendations.append("准确率较低，建议降低capability_weight")
        elif perf.accuracy > 0.9:
            recommendations.append("表现优秀，可以提高capability_weight")
        
        if perf.avg_confidence < 0.5:
            recommendations.append("置信度较低，可能需要更好的prompt")
        
        if perf.avg_response_time > 10.0:
            recommendations.append("响应时间较长，考虑优化或使用更快的模型")
        
        if not recommendations:
            recommendations.append("表现稳定，继续保持")
        
        return recommendations
    
    async def export_knowledge_base(self, output_path: str) -> int:
        """Export in-memory knowledge base docs to a JSON file."""
        import json
        from pathlib import Path

        docs = getattr(self.knowledge_base, "_documents", {}) or {}
        payload = []
        for doc_id, doc in docs.items():
            payload.append(
                {
                    "doc_id": doc_id,
                    "content": doc.content,
                    "category": doc.category,
                    "metadata": doc.metadata,
                }
            )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(payload)

    async def import_knowledge_base(self, input_path: str) -> int:
        """Import documents from a JSON file produced by :meth:`export_knowledge_base`."""
        import json
        from pathlib import Path

        path = Path(input_path)
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0
        if not isinstance(payload, list):
            return 0

        count = 0
        for entry in payload:
            if not isinstance(entry, dict) or "content" not in entry:
                continue
            await self.knowledge_base.add_document(
                content=entry["content"],
                category=entry.get("category", "imported"),
                metadata=entry.get("metadata") or {},
                doc_id=entry.get("doc_id"),
            )
            count += 1
        return count

    async def clear_old_data(self, days: int = 90) -> Dict[str, int]:
        """Drop in-memory consensus/feedback records older than ``days``."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)
        consensus_deleted = 0
        feedback_deleted = 0

        consensus_store = getattr(self.experience_base, "_consensus_records", None)
        if isinstance(consensus_store, dict):
            stale = [
                cid for cid, rec in consensus_store.items()
                if getattr(rec, "timestamp", datetime.utcnow()) < cutoff
            ]
            for cid in stale:
                consensus_store.pop(cid, None)
            consensus_deleted = len(stale)

        feedback_store = getattr(self.experience_base, "_feedback_records", None)
        if isinstance(feedback_store, dict):
            stale = [
                fid for fid, rec in feedback_store.items()
                if getattr(rec, "timestamp", datetime.utcnow()) < cutoff
            ]
            for fid in stale:
                feedback_store.pop(fid, None)
            feedback_deleted = len(stale)

        return {
            "consensus_records_deleted": consensus_deleted,
            "feedback_records_deleted": feedback_deleted,
        }

