"""Data models for Graph RAG functionality."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class GraphNode:
    """Represents a node (entity) in the knowledge graph.

    This model wraps Graphiti's EntityNode for use within Insta RAG.
    """

    uuid: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    labels: List[str] = field(default_factory=list)
    summary: str = ""
    group_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "group_id": self.group_id,
            "properties": self.properties,
        }


@dataclass
class GraphEdge:
    """Represents an edge (relationship/fact) in the knowledge graph.

    This model wraps Graphiti's EntityEdge for use within Insta RAG.
    """

    uuid: str = field(default_factory=lambda: str(uuid4()))
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    fact: str = ""
    relationship_type: str = ""
    group_id: str = ""
    score: float = 0.0
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    created_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "uuid": self.uuid,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "fact": self.fact,
            "relationship_type": self.relationship_type,
            "group_id": self.group_id,
            "score": self.score,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "created_at": self.created_at,
            "episodes": self.episodes,
            "properties": self.properties,
        }


@dataclass
class GraphRetrievalResult:
    """Result of a graph retrieval operation."""

    edges: List[GraphEdge] = field(default_factory=list)
    nodes: List[GraphNode] = field(default_factory=list)
    total_count: int = 0
    query: str = ""
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "nodes": [node.to_dict() for node in self.nodes],
            "total_count": self.total_count,
            "query": self.query,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass
class GraphAddResult:
    """Result of adding an episode/document to the graph."""

    episode_uuid: str = ""
    nodes_created: int = 0
    edges_created: int = 0
    group_id: str = ""
    processing_time_ms: float = 0.0
    extracted_nodes: List[GraphNode] = field(default_factory=list)
    extracted_edges: List[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "episode_uuid": self.episode_uuid,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "group_id": self.group_id,
            "processing_time_ms": self.processing_time_ms,
            "extracted_nodes": [n.to_dict() for n in self.extracted_nodes],
            "extracted_edges": [e.to_dict() for e in self.extracted_edges],
        }
