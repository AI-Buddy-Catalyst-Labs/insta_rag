"""Graph retriever for searching knowledge graphs."""

import time
from typing import Optional

from graphiti_core import Graphiti

from .models import GraphEdge, GraphRetrievalResult


class GraphRetriever:
    """Retrieves information from knowledge graphs using Graphiti.

    Performs hybrid search combining semantic and BM25 keyword search
    on the knowledge graph to find relevant facts and entities.
    """

    def __init__(self, graphiti_client: Graphiti, group_id: str = "default"):
        """Initialize GraphRetriever.

        Args:
            graphiti_client: Initialized Graphiti instance
            group_id: Default group ID for queries
        """
        self.graphiti = graphiti_client
        self.group_id = group_id

    async def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        k: int = 10,
        use_filters: bool = False,
        **filter_kwargs,
    ) -> GraphRetrievalResult:
        """Retrieve relevant facts from the knowledge graph.

        Args:
            query: Search query
            collection_name: Collection to search in
            k: Number of results to return
            use_filters: Whether to apply filters
            **filter_kwargs: Filter arguments (entity_labels, valid_after, etc.)

        Returns:
            GraphRetrievalResult with edges and nodes

        Raises:
            RuntimeError: If retrieval fails
        """
        start_time = time.time()
        group_id = f"{self.group_id}_{collection_name}"

        try:
            # Perform hybrid search using Graphiti
            # This combines semantic and BM25 search automatically
            results = await self.graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=k,
            )

            # Convert results to GraphEdge objects
            edges = []
            node_uuids = set()

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
                    relationship_type=(
                        result.name if hasattr(result, "name") else "RELATED_TO"
                    ),
                    score=getattr(result, "score", 0.0),
                    valid_at=(
                        result.valid_at.isoformat()
                        if hasattr(result, "valid_at") and result.valid_at
                        else None
                    ),
                    created_at=(
                        result.created_at.isoformat()
                        if hasattr(result, "created_at") and result.created_at
                        else None
                    ),
                    group_id=group_id,
                )
                edges.append(edge)

                # Track node UUIDs for retrieval
                if edge.source_node_uuid:
                    node_uuids.add(edge.source_node_uuid)
                if edge.target_node_uuid:
                    node_uuids.add(edge.target_node_uuid)

            # Note: Graphiti search returns edges (facts) not nodes directly
            # For full node information, you would need separate node queries
            # This is a limitation we document

            retrieval_time_ms = (time.time() - start_time) * 1000

            return GraphRetrievalResult(
                edges=edges,
                nodes=[],  # Graphiti returns facts (edges), not nodes
                total_count=len(edges),
                query=query,
                retrieval_time_ms=retrieval_time_ms,
            )

        except Exception as e:
            raise RuntimeError(f"Graph retrieval failed: {str(e)}")

    async def retrieve_with_reranking(
        self,
        query: str,
        collection_name: str = "default",
        k: int = 10,
        center_node_uuid: Optional[str] = None,
    ) -> GraphRetrievalResult:
        """Retrieve facts with distance-based reranking.

        Results are reranked based on their graph distance from a center node.

        Args:
            query: Search query
            collection_name: Collection to search in
            k: Number of results to return
            center_node_uuid: UUID of node to use as reranking center

        Returns:
            GraphRetrievalResult with reranked edges

        Raises:
            RuntimeError: If retrieval fails
        """
        start_time = time.time()
        group_id = f"{self.group_id}_{collection_name}"

        try:
            # If center node not provided, do initial search to find one
            if not center_node_uuid:
                initial_results = await self.graphiti.search(
                    query=query,
                    group_ids=[group_id],
                    num_results=1,
                )
                if initial_results:
                    center_node_uuid = (
                        initial_results[0].source_node_uuid
                        if hasattr(initial_results[0], "source_node_uuid")
                        else None
                    )

            # Perform reranked search using center node
            results = await self.graphiti.search(
                query=query,
                group_ids=[group_id],
                center_node_uuid=center_node_uuid,
                num_results=k,
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

            retrieval_time_ms = (time.time() - start_time) * 1000

            return GraphRetrievalResult(
                edges=edges,
                nodes=[],
                total_count=len(edges),
                query=query,
                retrieval_time_ms=retrieval_time_ms,
            )

        except Exception as e:
            raise RuntimeError(f"Reranked graph retrieval failed: {str(e)}")

    async def get_entity_context(
        self,
        entity_name: str,
        collection_name: str = "default",
        depth: int = 1,
    ) -> GraphRetrievalResult:
        """Get context around an entity in the knowledge graph.

        Retrieves the entity and its connected relationships.

        Args:
            entity_name: Name of entity to find
            collection_name: Collection to search in
            depth: Depth of relationships to traverse (1-3)

        Returns:
            GraphRetrievalResult with entity and related facts
        """
        # This is a limitation of the Graphiti API - it doesn't have
        # direct entity lookup. We use search as a workaround.
        return await self.retrieve(
            query=entity_name,
            collection_name=collection_name,
            k=max(5 * depth, 20),
        )
