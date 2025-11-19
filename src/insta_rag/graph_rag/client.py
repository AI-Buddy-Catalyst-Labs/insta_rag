"""Graph RAG Client - main entry point for Graph RAG operations."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from openai import AsyncOpenAI

from ..models.document import DocumentInput
from ..models.chunk import Chunk
from .neo4j_driver import Neo4jGraphDriver
from .graph_builder import GraphBuilder
from .graph_retriever import GraphRetriever
from .models import GraphAddResult, GraphRetrievalResult

# Import Graphiti's Azure OpenAI client classes
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient


class GraphRAGClient:
    """Main Graph RAG client for knowledge graph operations.

    This client orchestrates all graph-based RAG operations including:
    - Knowledge graph construction from documents
    - Graph-based retrieval with hybrid search
    - Entity and relationship extraction
    - Temporal awareness of facts

    This client operates independently from the regular RAGClient (Qdrant-based).
    Both can coexist in the same application.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        group_id: str = "insta_rag",
        use_azure_openai: bool = True,
    ):
        """Initialize Graph RAG client.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            group_id: Group ID for organizing graph data
            use_azure_openai: Whether to use Azure OpenAI (default: True)

        Raises:
            ValueError: If Neo4j configuration is invalid
        """
        self.group_id = group_id

        # Initialize Azure OpenAI clients if requested
        llm_client = None
        embedder = None

        if use_azure_openai:
            llm_client, embedder = self._create_azure_openai_clients()

        self.driver = Neo4jGraphDriver(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
            llm_client=llm_client,
            embedder=embedder,
        )

        # These will be initialized after Neo4j connection
        self._graphiti = None
        self._builder = None
        self._retriever = None

    @staticmethod
    def _create_azure_openai_clients() -> Tuple[Optional[Any], Optional[Any]]:
        """Create Azure OpenAI LLM and embedder clients using Graphiti's Azure classes.

        Uses Graphiti's dedicated AzureOpenAILLMClient and AzureOpenAIEmbedderClient
        with AsyncOpenAI configured for Azure's v1 API endpoint.

        Returns:
            Tuple of (llm_client, embedder_client) or (None, None) if not configured

        Raises:
            RuntimeError: If Azure OpenAI configuration is incomplete
        """
        # Get Azure OpenAI configuration
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # Try both environment variable names for backward compatibility
        azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        azure_llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT") or os.getenv("GRAPHITI_LLM_MODEL", "gpt-4.1")

        # If credentials not set, return None (Graphiti will use defaults)
        if not azure_endpoint or not azure_api_key:
            return None, None

        try:
            # Format Azure endpoint for OpenAI v1 API
            # Remove trailing slash if present, then append /openai/v1/
            if azure_endpoint.endswith("/"):
                azure_endpoint = azure_endpoint[:-1]
            base_url = f"{azure_endpoint}/openai/v1/"

            # Create single AsyncOpenAI client configured for Azure
            # This client will be used for both LLM and embeddings
            azure_client = AsyncOpenAI(
                base_url=base_url,
                api_key=azure_api_key,
            )

            # Create Graphiti's Azure LLM client
            # model parameter should be your Azure deployment name (e.g., "gpt-4.1")
            llm_client = AzureOpenAILLMClient(
                azure_client=azure_client,
                config=LLMConfig(
                    model=azure_llm_deployment,
                    small_model=azure_llm_deployment,
                ),
            )

            # Create Graphiti's Azure Embedder client
            # model parameter should be your Azure embedding deployment name
            embedder_client = AzureOpenAIEmbedderClient(
                azure_client=azure_client,
                model=azure_embedding_deployment,
            )

            return llm_client, embedder_client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI clients: {str(e)}")

    async def initialize(self) -> "GraphRAGClient":
        """Initialize Neo4j connection and components.

        Must be called before using the client.

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If initialization fails
        """
        self._graphiti = await self.driver.initialize()
        self._builder = GraphBuilder(self._graphiti, self.group_id)
        self._retriever = GraphRetriever(self._graphiti, self.group_id)
        return self

    async def close(self) -> None:
        """Close Neo4j connection."""
        await self.driver.close()

    # ======================== Document Operations ========================

    async def add_documents(
        self,
        documents: List[DocumentInput],
        collection_name: str = "default",
    ) -> List[GraphAddResult]:
        """Add documents to the knowledge graph.

        Converts documents to episodes and automatically extracts entities
        and relationships using Graphiti's LLM-based extraction.

        Args:
            documents: List of DocumentInput objects to add
            collection_name: Name of collection to add to

        Returns:
            List of GraphAddResult objects with extraction results

        Raises:
            RuntimeError: If not initialized
            ValueError: If documents are invalid
        """
        if not self._builder:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await self._builder.add_documents(documents, collection_name)

    async def add_chunk(
        self,
        chunk: Chunk,
        collection_name: str = "default",
    ) -> GraphAddResult:
        """Add a single chunk to the knowledge graph.

        Args:
            chunk: Chunk object to add
            collection_name: Name of collection to add to

        Returns:
            GraphAddResult with extraction results

        Raises:
            RuntimeError: If not initialized
        """
        if not self._builder:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await self._builder.add_chunk(chunk, collection_name)

    # ======================== Retrieval Operations ========================

    async def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        k: int = 10,
    ) -> GraphRetrievalResult:
        """Retrieve relevant facts from the knowledge graph.

        Performs hybrid search combining semantic and BM25 keyword search
        to find facts relevant to the query.

        Args:
            query: Query string
            collection_name: Collection to search in
            k: Number of top facts to return

        Returns:
            GraphRetrievalResult with retrieved facts

        Raises:
            RuntimeError: If not initialized or retrieval fails
        """
        if not self._retriever:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await self._retriever.retrieve(
            query=query,
            collection_name=collection_name,
            k=k,
        )

    async def retrieve_with_reranking(
        self,
        query: str,
        collection_name: str = "default",
        k: int = 10,
        center_node_uuid: Optional[str] = None,
    ) -> GraphRetrievalResult:
        """Retrieve facts with distance-based reranking.

        Results are reranked based on their graph distance from a center node,
        which improves relevance for interconnected facts.

        Args:
            query: Query string
            collection_name: Collection to search in
            k: Number of results to return
            center_node_uuid: Optional center node for reranking

        Returns:
            GraphRetrievalResult with reranked facts

        Raises:
            RuntimeError: If not initialized or retrieval fails
        """
        if not self._retriever:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await self._retriever.retrieve_with_reranking(
            query=query,
            collection_name=collection_name,
            k=k,
            center_node_uuid=center_node_uuid,
        )

    async def get_entity_context(
        self,
        entity_name: str,
        collection_name: str = "default",
        depth: int = 1,
    ) -> GraphRetrievalResult:
        """Get context and relationships for an entity.

        Retrieves all facts related to the specified entity.

        Args:
            entity_name: Name of entity to find
            collection_name: Collection to search in
            depth: Depth of relationships to traverse (1-3)

        Returns:
            GraphRetrievalResult with entity context

        Raises:
            RuntimeError: If not initialized or retrieval fails
        """
        if not self._retriever:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await self._retriever.get_entity_context(
            entity_name=entity_name,
            collection_name=collection_name,
            depth=depth,
        )

    # ======================== Context Manager Support ========================

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit."""
        await self.close()
