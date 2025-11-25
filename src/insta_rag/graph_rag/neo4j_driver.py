"""Neo4j driver wrapper for Graph RAG."""

import os
from typing import Any, Optional

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver


class Neo4jGraphDriver:
    """Wrapper for Neo4j connection using Graphiti.

    This class manages the connection to Neo4j and provides a clean interface
    for initializing Graphiti with proper configuration.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        llm_client: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ):
        """Initialize Neo4j driver.

        Args:
            uri: Neo4j Bolt URI (defaults to NEO4J_URI env var)
            user: Neo4j username (defaults to NEO4J_USER env var)
            password: Neo4j password (defaults to NEO4J_PASSWORD env var)
            database: Neo4j database name (defaults to NEO4J_DATABASE env var)
            llm_client: Optional Graphiti LLM client (for Azure OpenAI or custom LLM)
            embedder: Optional Graphiti embedder client (for Azure OpenAI or custom embedder)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "insta_rag_graph")
        self.llm_client = llm_client
        self.embedder = embedder

        self._graphiti: Optional[Graphiti] = None
        self._neo4j_driver: Optional[Any] = None  # Store Neo4jDriver reference for deletion operations

    async def initialize(self) -> Graphiti:
        """Initialize and return Graphiti instance.

        Raises:
            RuntimeError: If initialization fails.

        Returns:
            Graphiti instance.
        """
        try:
            # Create Neo4jDriver with database parameter
            driver = Neo4jDriver(
                uri=self.uri,
                user=self.user,
                password=self.password,
                database=self.database,
            )

            # Store reference for use in deletion operations
            self._neo4j_driver = driver

            # Create Graphiti instance with the driver and optional LLM/embedder clients
            graphiti_kwargs = {"graph_driver": driver}
            if self.llm_client:
                graphiti_kwargs["llm_client"] = self.llm_client
            if self.embedder:
                graphiti_kwargs["embedder"] = self.embedder

            self._graphiti = Graphiti(**graphiti_kwargs)

            # Build indices and constraints
            await self._graphiti.build_indices_and_constraints()
            return self._graphiti
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Neo4j connection: {str(e)}")

    async def close(self) -> None:
        """Close the Graphiti connection."""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None

    def get_graphiti(self) -> Optional[Graphiti]:
        """Get the Graphiti instance.

        Returns:
            Graphiti instance if initialized, None otherwise.
        """
        return self._graphiti

    def get_neo4j_driver(self) -> Optional[Any]:
        """Get the Neo4j driver instance for raw query execution.

        Returns:
            Neo4jDriver instance if initialized, None otherwise.
        """
        return self._neo4j_driver

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
