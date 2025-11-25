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

    def __init__(self, graphiti_client: Graphiti, group_id: str = "default", neo4j_driver=None):
        """Initialize GraphBuilder.

        Args:
            graphiti_client: Initialized Graphiti instance
            group_id: Group ID for organizing graph data
            neo4j_driver: Optional Neo4j driver for raw query execution
        """
        self.graphiti = graphiti_client
        self.group_id = group_id
        self.neo4j_driver = neo4j_driver

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

        return self._convert_episode_result(episode_result, group_id, processing_time_ms)

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

    async def delete_node(
        self,
        node_uuid: str,
        collection_name: str,
    ) -> dict:
        """Delete a single node (entity) and all its connected edges.

        Args:
            node_uuid: UUID of the entity node to delete
            collection_name: Collection context (for reference)

        Returns:
            Dict with keys:
                - success: bool
                - node_uuid: The deleted node UUID
                - edges_deleted: Number of connected edges removed
                - error: Optional error message

        Raises:
            RuntimeError: If operation fails
        """
        try:
            if not self.neo4j_driver:
                return {"success": False, "error": "Neo4j driver not initialized"}

            driver = self.neo4j_driver

            # Verify node exists
            node_query = "MATCH (n:Entity {uuid: $uuid}) RETURN n.uuid AS uuid"
            records, _, _ = await driver.execute_query(node_query, uuid=node_uuid)

            if not records:
                return {
                    "success": False,
                    "error": f"Node {node_uuid} not found",
                }

            # Count connected edges before deletion
            edge_count_query = (
                "MATCH (n:Entity {uuid: $uuid})-[e]-(m) RETURN COUNT(e) AS edge_count"
            )
            edge_records, _, _ = await driver.execute_query(
                edge_count_query, uuid=node_uuid
            )
            edge_count = edge_records[0]["edge_count"] if edge_records else 0

            # Delete the node (DETACH DELETE removes all connected edges)
            delete_query = "MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n"
            await driver.execute_query(delete_query, uuid=node_uuid)

            return {
                "success": True,
                "node_uuid": node_uuid,
                "edges_deleted": edge_count,
                "message": f"Deleted node {node_uuid} and {edge_count} connected edges",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to delete node: {str(e)}"}

    async def delete_edge(
        self,
        edge_uuid: str,
        collection_name: str,
    ) -> dict:
        """Delete a single edge (relationship/fact).

        Args:
            edge_uuid: UUID of the edge to delete
            collection_name: Collection context (for reference)

        Returns:
            Dict with deletion status:
                - success: bool
                - edge_uuid: The deleted edge UUID
                - error: Optional error message

        Raises:
            RuntimeError: If operation fails
        """
        try:
            if not self.neo4j_driver:
                return {"success": False, "error": "Neo4j driver not initialized"}

            driver = self.neo4j_driver

            # Verify edge exists (try multiple relationship types)
            edge_query = "MATCH ()-[e {uuid: $uuid}]-() RETURN e.uuid AS uuid"
            records, _, _ = await driver.execute_query(edge_query, uuid=edge_uuid)

            if not records:
                return {"success": False, "error": f"Edge {edge_uuid} not found"}

            # Delete the edge
            delete_query = "MATCH ()-[e {uuid: $uuid}]-() DELETE e"
            await driver.execute_query(delete_query, uuid=edge_uuid)

            return {
                "success": True,
                "edge_uuid": edge_uuid,
                "message": f"Deleted edge {edge_uuid}",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to delete edge: {str(e)}"}

    async def delete_episode(
        self,
        episode_uuid: str,
        collection_name: str,
    ) -> dict:
        """Delete an episode (document) and all its extracted entities/relationships.

        Removes all edges belonging to this episode first, then deletes any
        orphaned nodes (nodes with no remaining connections).

        Args:
            episode_uuid: UUID of the episode to delete
            collection_name: Collection context

        Returns:
            Dict with deletion statistics:
                - success: bool
                - episode_uuid: The deleted episode UUID
                - edges_deleted: Number of edges removed
                - orphan_nodes_deleted: Number of orphaned nodes removed
                - error: Optional error message

        Raises:
            RuntimeError: If operation fails
        """
        try:
            if not self.neo4j_driver:
                return {"success": False, "error": "Neo4j driver not initialized"}

            driver = self.neo4j_driver
            group_id = f"{self.group_id}_{collection_name}"

            # Count edges to delete (edges with this episode in their episodes list)
            edge_count_query = """
            MATCH ()-[e {group_id: $group_id}]-()
            WHERE $episode_uuid IN e.episodes
            RETURN COUNT(e) AS edge_count
            """
            edge_records, _, _ = await driver.execute_query(
                edge_count_query,
                group_id=group_id,
                episode_uuid=episode_uuid,
            )
            edge_count = edge_records[0]["edge_count"] if edge_records else 0

            # Delete edges from this episode
            delete_edges_query = """
            MATCH ()-[e {group_id: $group_id}]-()
            WHERE $episode_uuid IN e.episodes
            DELETE e
            """
            await driver.execute_query(
                delete_edges_query,
                group_id=group_id,
                episode_uuid=episode_uuid,
            )

            # Find orphaned nodes (nodes with no connected edges)
            orphan_query = """
            MATCH (n:Entity {group_id: $group_id})
            WHERE NOT (n)-[]-()
            RETURN COUNT(n) AS orphan_count
            """
            orphan_records, _, _ = await driver.execute_query(
                orphan_query, group_id=group_id
            )
            orphan_count = orphan_records[0]["orphan_count"] if orphan_records else 0

            # Delete orphaned nodes
            delete_orphans_query = """
            MATCH (n:Entity {group_id: $group_id})
            WHERE NOT (n)-[]-()
            DELETE n
            """
            await driver.execute_query(delete_orphans_query, group_id=group_id)

            return {
                "success": True,
                "episode_uuid": episode_uuid,
                "edges_deleted": edge_count,
                "orphan_nodes_deleted": orphan_count,
                "message": f"Deleted episode {episode_uuid}, {edge_count} edges, {orphan_count} orphaned nodes",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to delete episode: {str(e)}"}

    async def delete_collection(self, collection_name: str) -> dict:
        """Delete entire collection with all documents, entities, and relationships.

        Removes all edges first, then all nodes in the collection. This is a
        destructive operation that cannot be undone.

        Args:
            collection_name: Collection to delete

        Returns:
            Dict with deletion statistics:
                - success: bool
                - collection_name: Collection that was deleted
                - edges_deleted: Number of edges removed
                - nodes_deleted: Number of nodes removed
                - error: Optional error message

        Raises:
            RuntimeError: If operation fails
        """
        try:
            if not self.neo4j_driver:
                return {"success": False, "error": "Neo4j driver not initialized"}

            driver = self.neo4j_driver
            group_id = f"{self.group_id}_{collection_name}"

            # Delete all edges in this collection first
            delete_edges_query = "MATCH ()-[e {group_id: $group_id}]-() DELETE e"
            result_edges = await driver.execute_query(
                delete_edges_query, group_id=group_id
            )
            edges_affected = result_edges.summary.counters.relationships_deleted

            # Delete all nodes in this collection
            delete_nodes_query = "MATCH (n:Entity {group_id: $group_id}) DELETE n"
            result_nodes = await driver.execute_query(
                delete_nodes_query, group_id=group_id
            )
            nodes_affected = result_nodes.summary.counters.nodes_deleted

            return {
                "success": True,
                "collection_name": collection_name,
                "edges_deleted": edges_affected,
                "nodes_deleted": nodes_affected,
                "message": f"Deleted collection {collection_name}: {edges_affected} edges, {nodes_affected} nodes",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete collection: {str(e)}",
            }

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
