"""Tests for Graph RAG functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from insta_rag.graph_rag import (
    GraphRAGClient,
    GraphRetrievalResult,
    GraphNode,
    GraphEdge,
)
from insta_rag.core.config import GraphRAGConfig
from insta_rag import DocumentInput


class TestGraphRAGConfig:
    """Test Graph RAG configuration."""

    def test_config_from_env_defaults(self):
        """Test creating config from environment with defaults."""
        with patch.dict(
            "os.environ",
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "test_password",
            },
        ):
            config = GraphRAGConfig.from_env()
            assert config.neo4j_uri == "bolt://localhost:7687"
            assert config.neo4j_user == "neo4j"
            assert config.neo4j_password == "test_password"
            assert config.neo4j_database == "insta_rag_graph"

    def test_config_validates_required_fields(self):
        """Test that config validation catches missing fields."""
        config = GraphRAGConfig(
            neo4j_uri="",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        with pytest.raises(Exception):
            config.validate()

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = GraphRAGConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="test_db",
        )
        config_dict = config.to_dict()
        assert config_dict["neo4j_uri"] == "bolt://localhost:7687"
        assert config_dict["neo4j_database"] == "test_db"
        assert "neo4j_password" not in config_dict  # Sensitive data excluded


class TestGraphNode:
    """Test GraphNode model."""

    def test_node_creation(self):
        """Test creating a graph node."""
        node = GraphNode(
            name="Alice",
            labels=["Person"],
            summary="Senior Engineer",
            group_id="employees",
        )
        assert node.name == "Alice"
        assert "Person" in node.labels
        assert node.uuid  # Should have auto-generated UUID

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        node = GraphNode(
            name="TechCorp",
            labels=["Organization"],
            group_id="companies",
        )
        node_dict = node.to_dict()
        assert node_dict["name"] == "TechCorp"
        assert node_dict["labels"] == ["Organization"]


class TestGraphEdge:
    """Test GraphEdge model."""

    def test_edge_creation(self):
        """Test creating a graph edge."""
        edge = GraphEdge(
            source_node_uuid="alice-uuid",
            target_node_uuid="techcorp-uuid",
            fact="Alice works at TechCorp",
            relationship_type="WORKS_AT",
        )
        assert edge.fact == "Alice works at TechCorp"
        assert edge.relationship_type == "WORKS_AT"

    def test_edge_to_dict(self):
        """Test converting edge to dictionary."""
        edge = GraphEdge(
            source_node_uuid="source",
            target_node_uuid="target",
            fact="Test fact",
            score=0.95,
        )
        edge_dict = edge.to_dict()
        assert edge_dict["fact"] == "Test fact"
        assert edge_dict["score"] == 0.95


class TestGraphRetrievalResult:
    """Test GraphRetrievalResult model."""

    def test_result_creation(self):
        """Test creating retrieval result."""
        edge = GraphEdge(fact="Test fact")
        result = GraphRetrievalResult(
            edges=[edge],
            query="test query",
            total_count=1,
        )
        assert len(result.edges) == 1
        assert result.query == "test query"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        edge = GraphEdge(fact="Test fact")
        result = GraphRetrievalResult(edges=[edge], query="test")
        result_dict = result.to_dict()
        assert len(result_dict["edges"]) == 1
        assert result_dict["query"] == "test"


class TestGraphRAGClient:
    """Test GraphRAGClient functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test initializing Graph RAG client."""
        with patch(
            "insta_rag.graph_rag.neo4j_driver.Neo4jGraphDriver.initialize",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_graphiti = MagicMock()
            mock_init.return_value = mock_graphiti

            client = GraphRAGClient(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
            )

            await client.initialize()

            assert client._graphiti is not None
            assert client._builder is not None
            assert client._retriever is not None

    @pytest.mark.asyncio
    async def test_add_documents_not_initialized(self):
        """Test that add_documents raises error if not initialized."""
        client = GraphRAGClient()
        with pytest.raises(RuntimeError):
            await client.add_documents([])

    @pytest.mark.asyncio
    async def test_retrieve_not_initialized(self):
        """Test that retrieve raises error if not initialized."""
        client = GraphRAGClient()
        with pytest.raises(RuntimeError):
            await client.retrieve("test query")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        with patch(
            "insta_rag.graph_rag.neo4j_driver.Neo4jGraphDriver.initialize",
            new_callable=AsyncMock,
        ) as mock_init:
            with patch(
                "insta_rag.graph_rag.neo4j_driver.Neo4jGraphDriver.close",
                new_callable=AsyncMock,
            ) as mock_close:
                mock_graphiti = MagicMock()
                mock_init.return_value = mock_graphiti

                async with GraphRAGClient() as client:
                    assert client._graphiti is not None

                mock_close.assert_called_once()


class TestGraphBuilder:
    """Test GraphBuilder functionality."""

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding documents to graph."""
        from insta_rag.graph_rag.graph_builder import GraphBuilder

        mock_graphiti = AsyncMock()
        mock_episode_result = MagicMock()
        mock_episode_result.episode.uuid = "episode-uuid"
        mock_episode_result.nodes = []
        mock_episode_result.edges = []
        mock_graphiti.add_episode.return_value = mock_episode_result

        builder = GraphBuilder(mock_graphiti)
        docs = [DocumentInput.from_text("Test document")]

        results = await builder.add_documents(docs, collection_name="test")

        assert len(results) == 1
        assert results[0].episode_uuid == "episode-uuid"
        mock_graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_empty_list(self):
        """Test adding empty document list."""
        from insta_rag.graph_rag.graph_builder import GraphBuilder

        mock_graphiti = AsyncMock()
        builder = GraphBuilder(mock_graphiti)

        results = await builder.add_documents([])

        assert results == []
        mock_graphiti.add_episode.assert_not_called()


class TestGraphRetriever:
    """Test GraphRetriever functionality."""

    @pytest.mark.asyncio
    async def test_retrieve(self):
        """Test retrieving from graph."""
        from insta_rag.graph_rag.graph_retriever import GraphRetriever

        mock_graphiti = AsyncMock()
        mock_edge = MagicMock()
        mock_edge.uuid = "edge-uuid"
        mock_edge.fact = "Test fact"
        mock_edge.source_node_uuid = "source"
        mock_edge.target_node_uuid = "target"
        mock_edge.score = 0.95
        mock_graphiti.search.return_value = [mock_edge]

        retriever = GraphRetriever(mock_graphiti)
        result = await retriever.retrieve("test query", k=10)

        assert result.query == "test query"
        assert len(result.edges) == 1
        assert result.edges[0].score == 0.95
        mock_graphiti.search.assert_called_once()


class TestGraphRAGIntegration:
    """Integration tests for Graph RAG."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_mock(self):
        """Test complete workflow with mocked Neo4j."""
        with patch(
            "insta_rag.graph_rag.neo4j_driver.Neo4jGraphDriver.initialize",
            new_callable=AsyncMock,
        ) as mock_init:
            # Setup mocks
            mock_graphiti = AsyncMock()
            mock_init.return_value = mock_graphiti

            # Mock add_episode
            mock_episode_result = MagicMock()
            mock_episode_result.episode.uuid = "episode-1"
            mock_episode_result.nodes = []
            mock_episode_result.edges = []
            mock_graphiti.add_episode.return_value = mock_episode_result

            # Mock search
            mock_edge = MagicMock()
            mock_edge.uuid = "edge-1"
            mock_edge.fact = "Test fact"
            mock_edge.source_node_uuid = "source"
            mock_edge.target_node_uuid = "target"
            mock_edge.score = 0.85
            mock_graphiti.search.return_value = [mock_edge]

            # Run workflow
            async with GraphRAGClient() as client:
                # Add document
                docs = [DocumentInput.from_text("Alice works at TechCorp")]
                results = await client.add_documents(docs, collection_name="test")
                assert len(results) == 1

                # Retrieve
                retrieval_result = await client.retrieve(
                    "Who is Alice?", collection_name="test", k=5
                )
                assert len(retrieval_result.edges) == 1
                assert retrieval_result.edges[0].fact == "Test fact"


class TestGraphRAGErrorHandling:
    """Test error handling in Graph RAG."""

    @pytest.mark.asyncio
    async def test_add_documents_with_error_recovery(self):
        """Test that errors in one document don't stop processing."""
        from insta_rag.graph_rag.graph_builder import GraphBuilder

        mock_graphiti = AsyncMock()
        # First call raises error, second succeeds
        mock_episode_result = MagicMock()
        mock_episode_result.episode.uuid = "episode-2"
        mock_episode_result.nodes = []
        mock_episode_result.edges = []
        mock_graphiti.add_episode.side_effect = [
            Exception("Test error"),
            mock_episode_result,
        ]

        builder = GraphBuilder(mock_graphiti)
        docs = [
            DocumentInput.from_text("First doc"),
            DocumentInput.from_text("Second doc"),
        ]

        results = await builder.add_documents(docs)

        # Should process both, one failed but second succeeded
        assert len(results) == 1  # Only successful one


class TestGraphRAGPerformance:
    """Test performance aspects of Graph RAG."""

    @pytest.mark.asyncio
    async def test_retrieval_timing(self):
        """Test that retrieval timing is captured."""
        from insta_rag.graph_rag.graph_retriever import GraphRetriever

        mock_graphiti = AsyncMock()
        mock_edge = MagicMock()
        mock_edge.uuid = "edge-1"
        mock_edge.fact = "Test"
        mock_edge.source_node_uuid = "s"
        mock_edge.target_node_uuid = "t"
        mock_edge.score = 0.9
        mock_graphiti.search.return_value = [mock_edge]

        retriever = GraphRetriever(mock_graphiti)
        result = await retriever.retrieve("query")

        assert result.retrieval_time_ms > 0
