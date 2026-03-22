"""
Knowledge graph extraction from text and risk context.

Extracts entities and relationships to build knowledge graphs.
"""

from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import uuid
import re

from aegean.core.models import (
    KnowledgeGraph,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
)


class GraphExtractor:
    """
    Extracts knowledge graphs from text and structured data.
    
    Performs:
    - Entity recognition (NER)
    - Relationship extraction (RE)
    - Graph construction
    """

    def __init__(self):
        """Initialize graph extractor."""
        # Simple entity patterns (can be extended with NLP)
        self.entity_patterns = {
            "user": r"user[_\s]?(\w+|\d+)",
            "company": r"(company|organization|corp|inc|ltd)\s+(\w+)",
            "location": r"(US|CN|UK|EU|JP|IN|SG|HK|AU|CA|DE|FR|IT|ES|NL|SE|CH|NO|DK|BE|AT|PL|RU|BR|MX|ZA|AE|SG)",
            "amount": r"\$[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*(USD|CNY|EUR|GBP|JPY)",
            "transaction": r"(transaction|transfer|payment|trade|deal|order)",
            "event": r"(event|incident|alert|flag|report|case)",
        }
        
        # Simple relationship patterns
        self.relation_patterns = {
            "initiates": r"(\w+)\s+(initiates|starts|creates|sends|makes)\s+(\w+)",
            "transfers": r"(\w+)\s+(transfers|sends|pays|moves)\s+(\w+)",
            "involves": r"(\w+)\s+(involves|includes|contains)\s+(\w+)",
            "located_in": r"(\w+)\s+(located|based|in|from)\s+(\w+)",
            "influences": r"(\w+)\s+(influences|affects|impacts)\s+(\w+)",
            "depends_on": r"(\w+)\s+(depends|relies|needs)\s+(\w+)",
        }

    def extract_entities(self, text: str, source_type: str = "text") -> List[KnowledgeGraphEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            source_type: Type of source (text, risk_request, etc)
            
        Returns:
            List of extracted entities
        """
        entities = []
        seen_ids: Set[str] = set()
        
        # Extract by pattern
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity_name = match.group(0)
                
                # Create unique ID
                entity_id = f"{entity_type}_{len(seen_ids)}"
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                
                # Extract attributes based on type
                attributes = self._extract_attributes(entity_name, entity_type)
                
                entity = KnowledgeGraphEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_name,
                    attributes=attributes,
                )
                entities.append(entity)
        
        return entities

    def extract_relations(
        self,
        text: str,
        entities: List[KnowledgeGraphEntity]
    ) -> List[KnowledgeGraphRelation]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of extracted relationships
        """
        relations = []
        entity_names = {e.name: e.entity_id for e in entities}
        
        # Extract by pattern
        for rel_type, pattern in self.relation_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                source_name = match.group(1)
                target_name = match.group(3)
                
                # Find matching entities
                source_id = entity_names.get(source_name)
                target_id = entity_names.get(target_name)
                
                if source_id and target_id:
                    relation = KnowledgeGraphRelation(
                        relation_id=f"rel_{len(relations)}",
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_type=rel_type,
                        properties={"confidence": 0.7},
                    )
                    relations.append(relation)
        
        return relations

    def build_graph(
        self,
        text: str,
        source_type: str = "text",
        source_id: Optional[str] = None,
    ) -> KnowledgeGraph:
        """
        Build complete knowledge graph from text.
        
        Args:
            text: Input text
            source_type: Type of source
            source_id: Optional source ID
            
        Returns:
            KnowledgeGraph object
        """
        # Extract entities
        entities = self.extract_entities(text, source_type)
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        
        # Create graph
        graph = KnowledgeGraph(
            graph_id=f"graph_{uuid.uuid4().hex[:8]}",
            source_type=source_type,
            source_id=source_id,
            entities=entities,
            relations=relations,
            metadata={
                "extraction_method": "pattern_based",
                "entity_count": len(entities),
                "relation_count": len(relations),
            }
        )
        
        return graph

    def _extract_attributes(self, entity_name: str, entity_type: str) -> Dict:
        """
        Extract attributes from entity name.
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
            
        Returns:
            Dict of attributes
        """
        attributes = {}
        
        if entity_type == "amount":
            # Extract numeric value
            match = re.search(r"[\d,]+(?:\.\d{2})?", entity_name)
            if match:
                amount_str = match.group(0).replace(",", "")
                try:
                    attributes["value"] = float(amount_str)
                except ValueError:
                    pass
            
            # Extract currency
            match = re.search(r"(USD|CNY|EUR|GBP|JPY)", entity_name, re.IGNORECASE)
            if match:
                attributes["currency"] = match.group(1).upper()
        
        elif entity_type == "location":
            attributes["region"] = entity_name.upper()
        
        elif entity_type == "user":
            attributes["type"] = "user"
        
        elif entity_type == "company":
            attributes["type"] = "organization"
        
        return attributes


class RiskGraphBuilder:
    """
    Builds knowledge graphs specifically for risk assessment context.
    
    Extracts entities and relationships from risk requests.
    """

    def __init__(self):
        """Initialize risk graph builder."""
        self.extractor = GraphExtractor()

    def build_from_risk_context(
        self,
        subject_id: str,
        subject_type: str,
        action_type: str,
        description: str,
        amount: Optional[float] = None,
        currency: str = "USD",
        geo_location: Optional[str] = None,
        counterparty_id: Optional[str] = None,
        trace_context: Optional[str] = None,
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from risk assessment context.
        
        Args:
            subject_id: Subject ID (user, company, etc)
            subject_type: Subject type
            action_type: Type of action (payment, transfer, etc)
            description: Action description
            amount: Transaction amount
            currency: Currency code
            geo_location: Geographic location
            counterparty_id: Counterparty ID
            trace_context: Additional context/reasoning
            
        Returns:
            KnowledgeGraph for risk context
        """
        entities: List[KnowledgeGraphEntity] = []
        relations: List[KnowledgeGraphRelation] = []
        
        # Create subject entity
        subject_entity = KnowledgeGraphEntity(
            entity_id=f"subject_{subject_id}",
            entity_type=subject_type,
            name=subject_id,
            attributes={"id": subject_id},
        )
        entities.append(subject_entity)
        
        # Create action entity
        action_entity = KnowledgeGraphEntity(
            entity_id=f"action_{uuid.uuid4().hex[:8]}",
            entity_type="action",
            name=action_type,
            attributes={"type": action_type, "description": description},
        )
        entities.append(action_entity)
        
        # Create initiates relation
        initiates_rel = KnowledgeGraphRelation(
            relation_id=f"rel_{len(relations)}",
            source_entity_id=subject_entity.entity_id,
            target_entity_id=action_entity.entity_id,
            relation_type="initiates",
            properties={"confidence": 1.0},
        )
        relations.append(initiates_rel)
        
        # Add amount if present
        if amount is not None:
            amount_entity = KnowledgeGraphEntity(
                entity_id=f"amount_{uuid.uuid4().hex[:8]}",
                entity_type="amount",
                name=f"{amount} {currency}",
                attributes={"value": amount, "currency": currency},
            )
            entities.append(amount_entity)
            
            amount_rel = KnowledgeGraphRelation(
                relation_id=f"rel_{len(relations)}",
                source_entity_id=action_entity.entity_id,
                target_entity_id=amount_entity.entity_id,
                relation_type="has_amount",
                properties={"confidence": 1.0},
            )
            relations.append(amount_rel)
        
        # Add location if present
        if geo_location:
            location_entity = KnowledgeGraphEntity(
                entity_id=f"location_{uuid.uuid4().hex[:8]}",
                entity_type="location",
                name=geo_location,
                attributes={"region": geo_location},
            )
            entities.append(location_entity)
            
            location_rel = KnowledgeGraphRelation(
                relation_id=f"rel_{len(relations)}",
                source_entity_id=action_entity.entity_id,
                target_entity_id=location_entity.entity_id,
                relation_type="located_in",
                properties={"confidence": 1.0},
            )
            relations.append(location_rel)
        
        # Add counterparty if present
        if counterparty_id:
            counterparty_entity = KnowledgeGraphEntity(
                entity_id=f"counterparty_{counterparty_id}",
                entity_type="counterparty",
                name=counterparty_id,
                attributes={"id": counterparty_id},
            )
            entities.append(counterparty_entity)
            
            counterparty_rel = KnowledgeGraphRelation(
                relation_id=f"rel_{len(relations)}",
                source_entity_id=action_entity.entity_id,
                target_entity_id=counterparty_entity.entity_id,
                relation_type="involves",
                properties={"confidence": 1.0},
            )
            relations.append(counterparty_rel)
        
        # Build graph
        graph = KnowledgeGraph(
            graph_id=f"risk_graph_{uuid.uuid4().hex[:8]}",
            source_type="risk_request",
            source_id=subject_id,
            entities=entities,
            relations=relations,
            metadata={
                "subject_id": subject_id,
                "action_type": action_type,
                "has_trace": trace_context is not None,
            }
        )
        
        return graph

    def visualize_for_ui(self, graph: KnowledgeGraph) -> Dict:
        """
        Convert graph to UI-friendly format (D3.js/Cytoscape compatible).
        
        Args:
            graph: KnowledgeGraph to visualize
            
        Returns:
            Dict with nodes and links for visualization
        """
        nodes = []
        links = []
        
        # Convert entities to nodes
        for entity in graph.entities:
            node = {
                "id": entity.entity_id,
                "label": entity.name,
                "type": entity.entity_type,
                "attributes": entity.attributes,
            }
            nodes.append(node)
        
        # Convert relations to links
        for relation in graph.relations:
            link = {
                "source": relation.source_entity_id,
                "target": relation.target_entity_id,
                "type": relation.relation_type,
                "properties": relation.properties,
            }
            links.append(link)
        
        return {
            "nodes": nodes,
            "links": links,
            "graph_id": graph.graph_id,
            "source_type": graph.source_type,
        }

