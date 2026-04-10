"""
Discussion tracker for recording consensus rounds and agent interactions.

Tracks how agents discuss, change opinions, and form consensus.
"""

from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from pydantic import BaseModel, Field

from aegean.core.models import Solution, RoundDiscussion


class AgentRelationship(BaseModel):
    """Relationship edge between two agents in the discussion graph."""

    source_agent_id: str
    target_agent_id: str
    influence_weight: float = Field(0.0, ge=0.0, le=1.0)
    trust_score: float = Field(0.0, ge=0.0, le=1.0)
    disagreement_count: int = Field(0, ge=0)
    agreement_count: int = Field(0, ge=0)


class GroupGraph(BaseModel):
    """Simple discussion graph for a group of agents."""

    group_id: str
    nodes: List[str] = Field(default_factory=list)
    edges: List[AgentRelationship] = Field(default_factory=list)


class DiscussionTracker:
    """
    Tracks discussion process across consensus rounds.
    
    Records:
    - What each agent said in each round
    - How consensus evolved
    - Which agents influenced which
    """

    def __init__(self, group_id: str, agent_ids: List[str]):
        """
        Initialize discussion tracker.
        
        Args:
            group_id: Group ID
            agent_ids: List of participating agent IDs
        """
        self.group_id = group_id
        self.agent_ids = agent_ids
        self.rounds: List[RoundDiscussion] = []
        
        # Track agent opinions over time: agent_id -> [answer1, answer2, ...]
        self.opinion_history: Dict[str, List[str]] = {aid: [] for aid in agent_ids}
        
        # Track who influenced whom: (source, target) -> count
        self.influence_pairs: Dict[tuple, int] = defaultdict(int)
        
        # Track consensus path (order agents agreed)
        self.consensus_path: List[str] = []

    def record_round(
        self,
        round_number: int,
        agent_responses: Dict[str, Solution],
        candidate_answer: Optional[str] = None,
        candidate_confidence: Optional[float] = None,
        stability_counter: int = 0,
    ) -> RoundDiscussion:
        """
        Record a consensus round.
        
        Args:
            round_number: Round number (1-indexed)
            agent_responses: agent_id -> Solution mapping
            candidate_answer: Current candidate answer
            candidate_confidence: Candidate confidence
            stability_counter: Stability counter value
            
        Returns:
            RoundDiscussion object
        """
        # Determine consensus status
        if candidate_answer is None:
            consensus_status = "forming"
        elif stability_counter >= 2:
            consensus_status = "reached"
        else:
            consensus_status = "forming"
        
        # Record opinions
        for agent_id, solution in agent_responses.items():
            self.opinion_history[agent_id].append(solution.answer)
        
        # Create round record
        round_discussion = RoundDiscussion(
            round_number=round_number,
            agent_responses=agent_responses,
            consensus_status=consensus_status,
            candidate_answer=candidate_answer,
            candidate_confidence=candidate_confidence,
            stability_counter=stability_counter,
        )
        
        self.rounds.append(round_discussion)
        
        return round_discussion

    def analyze_influence(self) -> Dict[str, float]:
        """
        Analyze which agents influenced consensus formation.
        
        Returns:
            Dict mapping agent_id -> influence_score (0-1)
        """
        if len(self.rounds) < 2:
            # Not enough rounds to analyze influence
            return {aid: 0.5 for aid in self.agent_ids}
        
        influence_scores = {aid: 0.0 for aid in self.agent_ids}
        
        # Check who changed opinion and who they might have been influenced by
        for agent_id in self.agent_ids:
            opinions = self.opinion_history[agent_id]
            
            if len(opinions) < 2:
                continue
            
            # Count opinion changes
            changes = sum(1 for i in range(1, len(opinions)) if opinions[i] != opinions[i-1])
            
            # If agent changed opinion, they were influenced
            if changes > 0:
                # Find who had the final opinion before they changed
                final_opinion = opinions[-1]
                
                # Check other agents who had this opinion earlier
                for other_id in self.agent_ids:
                    if other_id == agent_id:
                        continue
                    
                    other_opinions = self.opinion_history[other_id]
                    if other_opinions and other_opinions[0] == final_opinion:
                        # Other agent had this opinion from the start
                        self.influence_pairs[(other_id, agent_id)] += 1
                        influence_scores[other_id] += 0.3
            else:
                # Agent never changed, they might have influenced others
                first_opinion = opinions[0]
                
                for other_id in self.agent_ids:
                    if other_id == agent_id:
                        continue
                    
                    other_opinions = self.opinion_history[other_id]
                    if len(other_opinions) > 1 and other_opinions[-1] == first_opinion:
                        # Other agent ended up with this agent's opinion
                        self.influence_pairs[(agent_id, other_id)] += 1
                        influence_scores[agent_id] += 0.3
        
        # Normalize scores
        max_score = max(influence_scores.values()) if influence_scores.values() else 1.0
        if max_score > 0:
            influence_scores = {k: v / max_score for k, v in influence_scores.items()}
        
        return influence_scores

    def build_agent_graph(self) -> GroupGraph:
        """
        Build agent relationship graph based on discussion.
        
        Returns:
            GroupGraph with agent relationships
        """
        influence_scores = self.analyze_influence()
        
        edges: List[AgentRelationship] = []
        
        # Create edges based on influence pairs
        for (source, target), count in self.influence_pairs.items():
            influence_weight = min(count * 0.3, 1.0)  # Cap at 1.0
            
            # Calculate trust based on agreement
            agreement_count = 0
            for round_disc in self.rounds:
                if source in round_disc.agent_responses and target in round_disc.agent_responses:
                    source_ans = round_disc.agent_responses[source].answer
                    target_ans = round_disc.agent_responses[target].answer
                    if source_ans == target_ans:
                        agreement_count += 1
            
            trust_score = agreement_count / len(self.rounds) if self.rounds else 0.5
            
            edge = AgentRelationship(
                source_agent_id=source,
                target_agent_id=target,
                influence_weight=influence_weight,
                trust_score=trust_score,
                disagreement_count=len(self.rounds) - agreement_count,
                agreement_count=agreement_count,
            )
            edges.append(edge)
        
        # Add self-loops for agents with high influence
        for agent_id, score in influence_scores.items():
            if score > 0.5:
                edge = AgentRelationship(
                    source_agent_id=agent_id,
                    target_agent_id=agent_id,
                    influence_weight=score,
                    trust_score=1.0,
                    disagreement_count=0,
                    agreement_count=len(self.rounds),
                )
                edges.append(edge)
        
        graph = GroupGraph(
            group_id=self.group_id,
            nodes=self.agent_ids,
            edges=edges,
        )
        
        return graph

    def get_consensus_path(self) -> List[str]:
        """
        Get the order in which agents agreed to consensus.
        
        Returns:
            List of agent IDs in order of consensus agreement
        """
        if not self.rounds:
            return []
        
        # Find the final consensus answer
        final_round = self.rounds[-1]
        if not final_round.candidate_answer:
            return []
        
        final_answer = final_round.candidate_answer
        path = []
        
        # Trace back which agents agreed to this answer
        for round_disc in self.rounds:
            for agent_id, solution in round_disc.agent_responses.items():
                if solution.answer == final_answer and agent_id not in path:
                    path.append(agent_id)
        
        return path

    def get_discussion_summary(self) -> Dict:
        """
        Get summary of discussion process.
        
        Returns:
            Dict with discussion statistics
        """
        if not self.rounds:
            return {
                "total_rounds": 0,
                "agents": self.agent_ids,
                "consensus_reached": False,
            }
        
        final_round = self.rounds[-1]
        
        # Count opinion changes per agent
        opinion_changes = {}
        for agent_id, opinions in self.opinion_history.items():
            changes = sum(1 for i in range(1, len(opinions)) if opinions[i] != opinions[i-1])
            opinion_changes[agent_id] = changes
        
        return {
            "total_rounds": len(self.rounds),
            "agents": self.agent_ids,
            "consensus_reached": final_round.consensus_status == "reached",
            "final_answer": final_round.candidate_answer,
            "final_confidence": final_round.candidate_confidence,
            "opinion_changes": opinion_changes,
            "influence_pairs": dict(self.influence_pairs),
            "consensus_path": self.get_consensus_path(),
        }

