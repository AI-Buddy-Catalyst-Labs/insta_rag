"""Graph builder for constructing knowledge graphs from documents."""

import time
from datetime import datetime, timezone
from typing import List

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..models.document import DocumentInput, SourceType
from ..models.chunk import Chunk
from .models import GraphAddResult, GraphNode, GraphEdge


class GraphBuilder:
    """Builds and manages knowledge graphs from documents using Graphiti.

    This class handles the conversion of documents to graph episodes,
    entity extraction, and relationship building.
    """

    def __init__(self, graphiti_client: Graphiti, group_id: str = "default"):
        """Initialize GraphBuilder.

        Args:
            graphiti_client: Initialized Graphiti instance
            group_id: Group ID for organizing graph data
        """
        self.graphiti = graphiti_client
        self.group_id = group_id

    async def add_documents(
        self,
        documents: List[DocumentInput],
        collection_name: str = "default",
    ) -> List[GraphAddResult]:
        """Add documents to the knowledge graph.

        Converts documents to episodes and extracts entities/relationships.

        Args:
            documents: List of documents to add
            collection_name: Collection name for grouping

        Returns:
            List of results for each document

        Raises:
            ValueError: If documents are invalid
            RuntimeError: If graph operation fails
        """
        if not documents:
            return []

        results = []
        group_id = f"{self.group_id}_{collection_name}"

        for doc in documents:
            try:
                # Extract content based on source type
                content = self._extract_content(doc)
                if not content:
                    continue

                start_time = time.time()

                # Add episode to graph - Graphiti automatically extracts entities
                episode_result = await self.graphiti.add_episode(
                    name=f"{collection_name}:{doc.metadata.get('source', 'unknown')}",
                    episode_body=content,
                    source=EpisodeType.text,
                    source_description=f"Document from {collection_name}",
                    reference_time=datetime.now(timezone.utc),
                    group_id=group_id,
                )

                processing_time_ms = (time.time() - start_time) * 1000

                # Convert Graphiti results to our models
                result = self._convert_episode_result(
                    episode_result, group_id, processing_time_ms
                )
                results.append(result)

            except Exception as e:
                # Log error but continue with other documents
                print(f"Error adding document to graph: {str(e)}")
                continue

        return results

    async def add_chunk(
        self,
        chunk: Chunk,
        collection_name: str = "default",
    ) -> GraphAddResult:
        """Add a single chunk to the knowledge graph.

        Args:
            chunk: Chunk to add
            collection_name: Collection name for grouping

        Returns:
            Result of the operation
        """
        start_time = time.time()
        group_id = f"{self.group_id}_{collection_name}"

        # Create episode from chunk
        episode_result = await self.graphiti.add_episode(
            name=f"chunk:{chunk.id}",
            episode_body=chunk.content,
            source=EpisodeType.text,
            source_description=f"Chunk from {collection_name}",
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        return self._convert_episode_result(
            episode_result, group_id, processing_time_ms
        )

    def _extract_content(self, document: DocumentInput) -> str:
        """Extract text content from document based on source type.

        Args:
            document: Document to extract from

        Returns:
            Extracted text content
        """
        if document.source_type == SourceType.TEXT:
            return document.get_source_text() or ""
        elif document.source_type == SourceType.FILE:
            # File content should be read and extracted by upstream processing
            # For now, we return empty string - user should pass as TEXT type
            return ""
        elif document.source_type == SourceType.BINARY:
            # Binary content is not suitable for graph extraction
            return ""
        else:
            return ""

    def _convert_episode_result(
        self,
        episode_result,
        group_id: str,
        processing_time_ms: float,
    ) -> GraphAddResult:
        """Convert Graphiti episode result to GraphAddResult.

        Args:
            episode_result: Result from Graphiti.add_episode
            group_id: Group ID
            processing_time_ms: Processing time in milliseconds

        Returns:
            GraphAddResult object
        """
        # Extract nodes from episode result
        nodes = []
        if hasattr(episode_result, "nodes") and episode_result.nodes:
            for node in episode_result.nodes:
                graph_node = GraphNode(
                    uuid=node.uuid if hasattr(node, "uuid") else "",
                    name=node.name if hasattr(node, "name") else "",
                    labels=node.labels if hasattr(node, "labels") else [],
                    summary=node.summary if hasattr(node, "summary") else "",
                    group_id=group_id,
                )
                nodes.append(graph_node)

        # Extract edges from episode result
        edges = []
        if hasattr(episode_result, "edges") and episode_result.edges:
            for edge in episode_result.edges:
                graph_edge = GraphEdge(
                    uuid=edge.uuid if hasattr(edge, "uuid") else "",
                    source_node_uuid=(
                        edge.source_node_uuid
                        if hasattr(edge, "source_node_uuid")
                        else ""
                    ),
                    target_node_uuid=(
                        edge.target_node_uuid
                        if hasattr(edge, "target_node_uuid")
                        else ""
                    ),
                    fact=edge.fact if hasattr(edge, "fact") else "",
                    relationship_type=(
                        edge.name if hasattr(edge, "name") else "RELATED_TO"
                    ),
                    group_id=group_id,
                    created_at=(
                        edge.created_at.isoformat()
                        if hasattr(edge, "created_at") and edge.created_at
                        else None
                    ),
                )
                edges.append(graph_edge)

        episode_uuid = (
            episode_result.episode.uuid
            if hasattr(episode_result, "episode")
            and hasattr(episode_result.episode, "uuid")
            else ""
        )

        return GraphAddResult(
            episode_uuid=episode_uuid,
            nodes_created=len(nodes),
            edges_created=len(edges),
            group_id=group_id,
            processing_time_ms=processing_time_ms,
            extracted_nodes=nodes,
            extracted_edges=edges,
        )

    async def delete_collection(self, collection_name: str) -> int:
        """Delete all episodes in a collection from the graph.

        Args:
            collection_name: Collection name to delete

        Returns:
            Number of episodes deleted

        Raises:
            RuntimeError: If operation fails
        """
        # Note: Graphiti doesn't have a built-in delete by collection method
        # This is a limitation we document for users
        raise NotImplementedError(
            "Collection deletion not yet implemented. "
            "Please use Neo4j directly or reset the database."
        )

    async def search_graph(
        self,
        query: str,
        collection_name: str = "default",
        num_results: int = 10,
    ) -> List[GraphEdge]:
        """Search the graph for relevant facts.

        Args:
            query: Search query
            collection_name: Collection to search in
            num_results: Number of results to return

        Returns:
            List of relevant GraphEdge objects
        """
        group_id = f"{self.group_id}:{collection_name}"

        results = await self.graphiti.search(
            query=query,
            group_ids=[group_id],
            num_results=num_results,
        )

        # Convert results to GraphEdge objects
        edges = []
        for result in results:
            edge = GraphEdge(
                uuid=result.uuid if hasattr(result, "uuid") else "",
                source_node_uuid=(
                    result.source_node_uuid
                    if hasattr(result, "source_node_uuid")
                    else ""
                ),
                target_node_uuid=(
                    result.target_node_uuid
                    if hasattr(result, "target_node_uuid")
                    else ""
                ),
                fact=result.fact if hasattr(result, "fact") else "",
                score=getattr(result, "score", 0.0),
                group_id=group_id,
            )
            edges.append(edge)

        return edges
