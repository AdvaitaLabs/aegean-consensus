"""
Experience Base for storing and learning from historical decisions.

Uses time-series database (TimescaleDB) for efficient historical queries.
Tracks consensus history, agent performance, and feedback.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class ConsensusRecord:
    """Record of a consensus execution."""
    consensus_id: str
    task: str
    final_answer: Optional[str]
    rounds_used: int
    consensus_reached: bool
    participating_agents: List[str]
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AgentPerformance:
    """Agent performance metrics."""
    agent_id: str
    total_participations: int
    correct_answers: int
    accuracy: float
    avg_confidence: float
    avg_response_time: float
    last_updated: datetime


@dataclass
class FeedbackRecord:
    """Feedback on a consensus result."""
    feedback_id: str
    consensus_id: str
    was_correct: bool
    feedback_source: str  # 'user', 'system', 'ground_truth'
    comments: Optional[str]
    timestamp: datetime


class ExperienceBase:
    """
    Experience base for storing and learning from historical decisions.
    
    Features:
    - Consensus history tracking
    - Agent performance metrics
    - Feedback collection and learning
    - Historical case retrieval
    - Time-series analysis
    
    Storage backends:
    - TimescaleDB (recommended for production)
    - PostgreSQL (alternative)
    - In-memory (for development/testing)
    """
    
    def __init__(
        self,
        backend: str = "memory",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experience base.
        
        Args:
            backend: Storage backend ('memory', 'timescaledb', 'postgresql')
            config: Backend-specific configuration
        """
        self.backend = backend
        self.config = config or {}
        
        # In-memory storage (for development)
        self._consensus_records: Dict[str, ConsensusRecord] = {}
        self._agent_performance: Dict[str, AgentPerformance] = {}
        self._feedback_records: Dict[str, FeedbackRecord] = {}
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize storage backend."""
        if self.backend == "memory":
            # Already initialized above
            pass
        elif self.backend == "timescaledb":
            self._init_timescaledb()
        elif self.backend == "postgresql":
            self._init_postgresql()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _init_timescaledb(self):
        """Initialize TimescaleDB/PostgreSQL connection."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            # Support both DATABASE_URL and individual params
            database_url = self.config.get("database_url") or self.config.get("DATABASE_URL")
            if database_url:
                self.conn = psycopg2.connect(database_url)
            else:
                self.conn = psycopg2.connect(
                    host=self.config.get("host", "localhost"),
                    port=self.config.get("port", 5432),
                    database=self.config.get("database", "aegean"),
                    user=self.config.get("user", "aegean"),
                    password=self.config.get("password", "")
                )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            self._create_tables()

        except ImportError:
            raise ImportError("psycopg2-binary not installed. Run: pip install psycopg2-binary")
        except Exception as e:
            raise RuntimeError(f"PostgreSQL connection failed: {e}. Check DATABASE_URL in .env")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL connection."""
        # Same as TimescaleDB but without hypertable
        self._init_timescaledb()
    
    def _create_tables(self):
        """Create database tables."""
        # Consensus records table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS consensus_records (
                consensus_id VARCHAR(255) PRIMARY KEY,
                task TEXT NOT NULL,
                final_answer TEXT,
                rounds_used INTEGER,
                consensus_reached BOOLEAN,
                participating_agents JSONB,
                execution_time FLOAT,
                timestamp TIMESTAMPTZ NOT NULL,
                metadata JSONB
            )
        """)
        
        # Convert to hypertable if TimescaleDB
        if self.backend == "timescaledb":
            try:
                self.cursor.execute("""
                    SELECT create_hypertable('consensus_records', 'timestamp', 
                                            if_not_exists => TRUE)
                """)
            except Exception:
                pass  # Already a hypertable
        
        # Agent performance table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                agent_id VARCHAR(255) PRIMARY KEY,
                total_participations INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                accuracy FLOAT DEFAULT 0.0,
                avg_confidence FLOAT DEFAULT 0.0,
                avg_response_time FLOAT DEFAULT 0.0,
                last_updated TIMESTAMPTZ
            )
        """)
        
        # Feedback records table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_records (
                feedback_id VARCHAR(255) PRIMARY KEY,
                consensus_id VARCHAR(255) REFERENCES consensus_records(consensus_id),
                was_correct BOOLEAN,
                feedback_source VARCHAR(50),
                comments TEXT,
                timestamp TIMESTAMPTZ NOT NULL
            )
        """)
        
        self.conn.commit()
    
    async def store_consensus(self, record: ConsensusRecord) -> str:
        """
        Store a consensus execution record.
        
        Args:
            record: ConsensusRecord to store
            
        Returns:
            Consensus ID
        """
        if self.backend == "memory":
            self._consensus_records[record.consensus_id] = record
        elif self.backend in ["timescaledb", "postgresql"]:
            self._store_consensus_db(record)
        
        # Update agent participation counts
        for agent_id in record.participating_agents:
            await self._update_agent_participation(agent_id)
        
        return record.consensus_id
    
    async def get_consensus(self, consensus_id: str) -> Optional[ConsensusRecord]:
        """Get a consensus record by ID."""
        if self.backend == "memory":
            return self._consensus_records.get(consensus_id)
        elif self.backend in ["timescaledb", "postgresql"]:
            return self._get_consensus_db(consensus_id)
        
        return None
    
    async def query_consensus_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ConsensusRecord]:
        """
        Query consensus history with filters.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            agent_id: Filter by agent participation
            limit: Maximum number of records
            
        Returns:
            List of ConsensusRecord objects
        """
        if self.backend == "memory":
            return self._query_consensus_memory(start_time, end_time, agent_id, limit)
        elif self.backend in ["timescaledb", "postgresql"]:
            return self._query_consensus_db(start_time, end_time, agent_id, limit)
        
        return []
    
    async def store_feedback(self, feedback: FeedbackRecord) -> str:
        """
        Store feedback on a consensus result.
        
        Args:
            feedback: FeedbackRecord to store
            
        Returns:
            Feedback ID
        """
        if self.backend == "memory":
            self._feedback_records[feedback.feedback_id] = feedback
        elif self.backend in ["timescaledb", "postgresql"]:
            self._store_feedback_db(feedback)
        
        # Update agent accuracy based on feedback
        consensus = await self.get_consensus(feedback.consensus_id)
        if consensus:
            for agent_id in consensus.participating_agents:
                await self._update_agent_accuracy(agent_id, feedback.was_correct)
        
        return feedback.feedback_id
    
    async def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get performance metrics for an agent."""
        if self.backend == "memory":
            return self._agent_performance.get(agent_id)
        elif self.backend in ["timescaledb", "postgresql"]:
            return self._get_agent_performance_db(agent_id)
        
        return None
    
    async def get_similar_cases(
        self,
        task: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[ConsensusRecord]:
        """
        Retrieve similar historical cases.
        
        Args:
            task: Current task description
            top_k: Number of similar cases to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar ConsensusRecord objects
        """
        # Simple implementation: keyword matching
        # TODO: Use embeddings for semantic similarity
        
        if self.backend == "memory":
            all_records = list(self._consensus_records.values())
        else:
            all_records = await self.query_consensus_history(limit=1000)
        
        # Calculate similarity (simple keyword overlap)
        task_words = set(task.lower().split())
        scored_records = []
        
        for record in all_records:
            record_words = set(record.task.lower().split())
            similarity = len(task_words & record_words) / len(task_words | record_words)
            
            if similarity >= min_similarity:
                scored_records.append((record, similarity))
        
        # Sort by similarity
        scored_records.sort(key=lambda x: x[1], reverse=True)
        
        return [record for record, _ in scored_records[:top_k]]
    
    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about consensus executions.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dictionary with statistics
        """
        records = await self.query_consensus_history(start_time, end_time, limit=10000)
        
        if not records:
            return {
                "total_consensus": 0,
                "success_rate": 0.0,
                "avg_rounds": 0.0,
                "avg_execution_time": 0.0
            }
        
        total = len(records)
        successful = sum(1 for r in records if r.consensus_reached)
        total_rounds = sum(r.rounds_used for r in records)
        total_time = sum(r.execution_time for r in records)
        
        return {
            "total_consensus": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_rounds": total_rounds / total if total > 0 else 0.0,
            "avg_execution_time": total_time / total if total > 0 else 0.0,
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        }
    
    # ==================== Helper Methods ====================
    
    async def _update_agent_participation(self, agent_id: str):
        """Update agent participation count."""
        if self.backend == "memory":
            if agent_id not in self._agent_performance:
                self._agent_performance[agent_id] = AgentPerformance(
                    agent_id=agent_id,
                    total_participations=0,
                    correct_answers=0,
                    accuracy=0.0,
                    avg_confidence=0.0,
                    avg_response_time=0.0,
                    last_updated=datetime.utcnow()
                )
            self._agent_performance[agent_id].total_participations += 1
            self._agent_performance[agent_id].last_updated = datetime.utcnow()
        elif self.backend in ["timescaledb", "postgresql"]:
            self.cursor.execute("""
                INSERT INTO agent_performance (agent_id, total_participations, last_updated)
                VALUES (%s, 1, %s)
                ON CONFLICT (agent_id) DO UPDATE
                SET total_participations = agent_performance.total_participations + 1,
                    last_updated = EXCLUDED.last_updated
            """, (agent_id, datetime.utcnow()))
            self.conn.commit()
    
    async def _update_agent_accuracy(self, agent_id: str, was_correct: bool):
        """Update agent accuracy based on feedback."""
        if self.backend == "memory":
            if agent_id in self._agent_performance:
                perf = self._agent_performance[agent_id]
                if was_correct:
                    perf.correct_answers += 1
                perf.accuracy = perf.correct_answers / perf.total_participations
                perf.last_updated = datetime.utcnow()
        elif self.backend in ["timescaledb", "postgresql"]:
            self.cursor.execute("""
                UPDATE agent_performance
                SET correct_answers = correct_answers + %s,
                    accuracy = CAST(correct_answers + %s AS FLOAT) / total_participations,
                    last_updated = %s
                WHERE agent_id = %s
            """, (1 if was_correct else 0, 1 if was_correct else 0, datetime.utcnow(), agent_id))
            self.conn.commit()
    
    def _store_consensus_db(self, record: ConsensusRecord):
        """Store consensus record in database."""
        import json
        
        self.cursor.execute("""
            INSERT INTO consensus_records 
            (consensus_id, task, final_answer, rounds_used, consensus_reached,
             participating_agents, execution_time, timestamp, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (consensus_id) DO UPDATE
            SET final_answer = EXCLUDED.final_answer,
                rounds_used = EXCLUDED.rounds_used,
                consensus_reached = EXCLUDED.consensus_reached
        """, (
            record.consensus_id,
            record.task,
            record.final_answer,
            record.rounds_used,
            record.consensus_reached,
            json.dumps(record.participating_agents),
            record.execution_time,
            record.timestamp,
            json.dumps(record.metadata)
        ))
        self.conn.commit()
    
    def _get_consensus_db(self, consensus_id: str) -> Optional[ConsensusRecord]:
        """Get consensus record from database."""
        import json
        
        self.cursor.execute("""
            SELECT * FROM consensus_records WHERE consensus_id = %s
        """, (consensus_id,))
        
        row = self.cursor.fetchone()
        if not row:
            return None
        
        return ConsensusRecord(
            consensus_id=row["consensus_id"],
            task=row["task"],
            final_answer=row["final_answer"],
            rounds_used=row["rounds_used"],
            consensus_reached=row["consensus_reached"],
            participating_agents=json.loads(row["participating_agents"]),
            execution_time=row["execution_time"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )
    
    def _query_consensus_memory(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        agent_id: Optional[str],
        limit: int
    ) -> List[ConsensusRecord]:
        """Query consensus records from memory."""
        records = list(self._consensus_records.values())
        
        # Filter by time range
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]
        
        # Filter by agent
        if agent_id:
            records = [r for r in records if agent_id in r.participating_agents]
        
        # Sort by timestamp (newest first)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        
        return records[:limit]
    
    def _query_consensus_db(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        agent_id: Optional[str],
        limit: int
    ) -> List[ConsensusRecord]:
        """Query consensus records from database."""
        import json
        
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        if agent_id:
            conditions.append("participating_agents @> %s")
            params.append(json.dumps([agent_id]))
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)
        
        self.cursor.execute(f"""
            SELECT * FROM consensus_records
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """, params)
        
        records = []
        for row in self.cursor.fetchall():
            records.append(ConsensusRecord(
                consensus_id=row["consensus_id"],
                task=row["task"],
                final_answer=row["final_answer"],
                rounds_used=row["rounds_used"],
                consensus_reached=row["consensus_reached"],
                participating_agents=json.loads(row["participating_agents"]),
                execution_time=row["execution_time"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            ))
        
        return records
    
    def _store_feedback_db(self, feedback: FeedbackRecord):
        """Store feedback record in database."""
        self.cursor.execute("""
            INSERT INTO feedback_records
            (feedback_id, consensus_id, was_correct, feedback_source, comments, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            feedback.feedback_id,
            feedback.consensus_id,
            feedback.was_correct,
            feedback.feedback_source,
            feedback.comments,
            feedback.timestamp
        ))
        self.conn.commit()
    
    def _get_agent_performance_db(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get agent performance from database."""
        self.cursor.execute("""
            SELECT * FROM agent_performance WHERE agent_id = %s
        """, (agent_id,))
        
        row = self.cursor.fetchone()
        if not row:
            return None
        
        return AgentPerformance(
            agent_id=row["agent_id"],
            total_participations=row["total_participations"],
            correct_answers=row["correct_answers"],
            accuracy=row["accuracy"],
            avg_confidence=row["avg_confidence"],
            avg_response_time=row["avg_response_time"],
            last_updated=row["last_updated"]
        )

