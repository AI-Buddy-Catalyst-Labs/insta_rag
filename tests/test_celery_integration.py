"""Tests for Celery async task integration."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insta_rag.core.config import CeleryConfig
from insta_rag.graph_rag.client import GraphRAGClient
from insta_rag.models.document import DocumentInput
from insta_rag.task_monitoring import TaskMonitoring, get_task_monitoring
from insta_rag.tasks.graph_rag_tasks import add_documents_task


class TestCeleryConfig:
    """Tests for CeleryConfig dataclass."""

    def test_celery_config_defaults(self):
        """Test CeleryConfig with default values."""
        config = CeleryConfig()
        assert config.enabled is True
        assert config.task_serializer == "json"
        assert config.task_track_started is True
        assert config.task_time_limit == 3600
        assert config.task_soft_time_limit == 3300

    def test_celery_config_from_env(self, monkeypatch):
        """Test CeleryConfig.from_env() with environment variables."""
        monkeypatch.setenv("CELERY_ENABLED", "false")
        monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", "7200")

        config = CeleryConfig.from_env()
        assert config.enabled is False
        assert config.task_time_limit == 7200

    def test_celery_config_validate(self):
        """Test CeleryConfig validation."""
        config = CeleryConfig(broker_url="", result_backend="redis://")

        with pytest.raises(Exception):  # ConfigurationError
            config.validate()

    def test_celery_config_to_dict(self):
        """Test CeleryConfig.to_dict() method."""
        config = CeleryConfig()
        config_dict = config.to_dict()

        assert "enabled" in config_dict
        assert "task_serializer" in config_dict
        assert "result_expires" in config_dict
        assert config_dict["enabled"] is True


class TestGraphRAGClientAsyncMethods:
    """Tests for GraphRAGClient async submission methods."""

    def test_submit_add_documents_async_returns_task_id(self):
        """Test that submit_add_documents_async returns task ID."""
        client = GraphRAGClient()

        documents = [
            DocumentInput.from_text("Test document 1"),
            DocumentInput.from_text("Test document 2"),
        ]

        with patch("insta_rag.tasks.graph_rag_tasks.add_documents_task") as mock_task:
            # Mock the async result
            mock_result = MagicMock()
            mock_result.id = "test-task-id-123"
            mock_task.apply_async.return_value = mock_result

            result = client.submit_add_documents_async(
                documents=documents,
                collection_name="test_collection",
            )

            # Verify return format
            assert "task_id" in result
            assert result["task_id"] == "test-task-id-123"
            assert result["status"] == "submitted"
            assert result["num_documents"] == 2
            assert result["collection_name"] == "test_collection"

            # Verify task was submitted with correct args
            mock_task.apply_async.assert_called_once()

    def test_submit_add_documents_async_empty_list_raises(self):
        """Test that submitting empty document list raises ValueError."""
        client = GraphRAGClient()

        with pytest.raises(ValueError, match="documents list cannot be empty"):
            client.submit_add_documents_async(documents=[], collection_name="test")

    def test_submit_add_documents_async_missing_celery_raises(self):
        """Test that missing Celery raises ImportError."""
        client = GraphRAGClient()

        documents = [DocumentInput.from_text("Test")]

        with patch("insta_rag.graph_rag.client.add_documents_task", side_effect=ImportError):
            with pytest.raises(ImportError, match="Celery is required"):
                client.submit_add_documents_async(documents, "test")

    def test_submit_add_documents_async_with_custom_neo4j_params(self):
        """Test submitting with custom Neo4j parameters."""
        client = GraphRAGClient()

        documents = [DocumentInput.from_text("Test")]

        with patch("insta_rag.tasks.graph_rag_tasks.add_documents_task") as mock_task:
            mock_result = MagicMock()
            mock_result.id = "task-id"
            mock_task.apply_async.return_value = mock_result

            result = client.submit_add_documents_async(
                documents=documents,
                collection_name="test",
                neo4j_uri="bolt://custom:7687",
                neo4j_user="custom_user",
                neo4j_password="custom_pass",
                neo4j_database="custom_db",
            )

            # Verify custom params were passed
            call_kwargs = mock_task.apply_async.call_args[1]["kwargs"]
            assert call_kwargs["neo4j_uri"] == "bolt://custom:7687"
            assert call_kwargs["neo4j_user"] == "custom_user"
            assert call_kwargs["neo4j_password"] == "custom_pass"
            assert call_kwargs["neo4j_database"] == "custom_db"

    def test_submit_add_chunk_async(self):
        """Test submitting a chunk asynchronously."""
        from insta_rag.models.chunk import Chunk, ChunkMetadata

        client = GraphRAGClient()

        metadata = ChunkMetadata(
            document_id="doc1",
            source="test.txt",
            chunk_index=0,
            total_chunks=1,
        )
        chunk = Chunk(
            chunk_id="chunk1",
            content="Test chunk content",
            metadata=metadata,
        )

        with patch("insta_rag.tasks.graph_rag_tasks.add_documents_task") as mock_task:
            mock_result = MagicMock()
            mock_result.id = "task-id"
            mock_task.apply_async.return_value = mock_result

            result = client.submit_add_chunk_async(chunk, "test_collection")

            assert result["task_id"] == "task-id"
            assert result["status"] == "submitted"
            assert result["num_documents"] == 1


class TestTaskMonitoring:
    """Tests for TaskMonitoring service."""

    def test_task_monitoring_initialization(self):
        """Test TaskMonitoring initialization."""
        monitor = TaskMonitoring()
        assert monitor.celery_app is not None

    def test_get_task_monitoring_singleton(self):
        """Test get_task_monitoring returns singleton."""
        monitor1 = get_task_monitoring()
        monitor2 = get_task_monitoring()
        assert monitor1 is monitor2

    @patch("insta_rag.task_monitoring.AsyncResult")
    def test_get_task_status(self, mock_async_result):
        """Test get_task_status method."""
        monitor = TaskMonitoring()

        # Mock AsyncResult
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False
        mock_result.result = {"status": "success"}

        mock_async_result.return_value = mock_result

        status = monitor.get_task_status("task-id-123")

        assert status["task_id"] == "task-id-123"
        assert status["state"] == "SUCCESS"
        assert status["ready"] is True
        assert status["successful"] is True
        assert status["failed"] is False

    @patch("insta_rag.task_monitoring.AsyncResult")
    def test_is_task_ready(self, mock_async_result):
        """Test is_task_ready method."""
        monitor = TaskMonitoring()

        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_async_result.return_value = mock_result

        assert monitor.is_task_ready("task-id") is True

        mock_result.ready.return_value = False
        assert monitor.is_task_ready("task-id") is False

    @patch("insta_rag.task_monitoring.AsyncResult")
    def test_is_task_successful(self, mock_async_result):
        """Test is_task_successful method."""
        monitor = TaskMonitoring()

        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_async_result.return_value = mock_result

        assert monitor.is_task_successful("task-id") is True

    @patch("insta_rag.task_monitoring.AsyncResult")
    def test_is_task_failed(self, mock_async_result):
        """Test is_task_failed method."""
        monitor = TaskMonitoring()

        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.failed.return_value = True
        mock_async_result.return_value = mock_result

        assert monitor.is_task_failed("task-id") is True

    @patch("insta_rag.task_monitoring.AsyncResult")
    def test_get_task_result_success(self, mock_async_result):
        """Test get_task_result with successful task."""
        monitor = TaskMonitoring()

        mock_result = MagicMock()
        mock_result.get.return_value = {"status": "success", "data": "result"}
        mock_async_result.return_value = mock_result

        result = monitor.get_task_result("task-id", timeout=10)

        assert result == {"status": "success", "data": "result"}
        mock_result.get.assert_called_once_with(timeout=10, propagate=True)

    def test_get_queue_length(self):
        """Test get_queue_length method."""
        monitor = TaskMonitoring()

        with patch.object(monitor, "get_active_tasks") as mock_active:
            with patch.object(monitor, "get_reserved_tasks") as mock_reserved:
                mock_active.return_value = {
                    "worker1": [{"id": "1"}, {"id": "2"}],
                    "worker2": [{"id": "3"}],
                }
                mock_reserved.return_value = {
                    "worker1": [{"id": "4"}, {"id": "5"}, {"id": "6"}],
                }

                length = monitor.get_queue_length()
                assert length == 6  # 3 active + 3 reserved

    @patch("insta_rag.task_monitoring.AsyncResult")
    def test_cancel_task(self, mock_async_result):
        """Test cancel_task method."""
        monitor = TaskMonitoring()

        with patch.object(monitor.celery_app.control, "revoke") as mock_revoke:
            result = monitor.cancel_task("task-id", terminate=False)

            assert result is True
            mock_revoke.assert_called_once_with("task-id", terminate=False)

    def test_get_active_tasks(self):
        """Test get_active_tasks method."""
        monitor = TaskMonitoring()

        with patch.object(monitor.celery_app.control, "inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.active.return_value = {
                "worker1": [{"name": "task1", "id": "id1"}],
                "worker2": [{"name": "task2", "id": "id2"}],
            }
            mock_inspect.return_value = mock_inspector

            active = monitor.get_active_tasks()

            assert "worker1" in active
            assert len(active["worker1"]) == 1

    def test_get_worker_stats(self):
        """Test get_worker_stats method."""
        monitor = TaskMonitoring()

        with patch.object(monitor.celery_app.control, "inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.stats.return_value = {
                "worker1": {"pool": {"max-concurrency": 4}, "total": 100},
            }
            mock_inspect.return_value = mock_inspector

            with patch.object(monitor, "get_active_tasks") as mock_active:
                with patch.object(monitor, "get_reserved_tasks") as mock_reserved:
                    mock_active.return_value = {
                        "worker1": [{"id": "1"}],
                    }
                    mock_reserved.return_value = {
                        "worker1": [{"id": "2"}, {"id": "3"}],
                    }

                    stats = monitor.get_worker_stats()

                    assert "worker1" in stats
                    assert stats["worker1"]["pool_size"] == 4
                    assert stats["worker1"]["active_tasks"] == 1
                    assert stats["worker1"]["reserved_tasks"] == 2


class TestAddDocumentsTask:
    """Tests for add_documents_task Celery task."""

    @pytest.mark.asyncio
    async def test_add_documents_task_structure(self):
        """Test add_documents_task structure and properties."""
        assert hasattr(add_documents_task, "apply_async")
        assert hasattr(add_documents_task, "delay")
        assert add_documents_task.name == "insta_rag.tasks.add_documents_task"

    def test_document_dict_conversion(self):
        """Test document serialization to dict format."""
        doc = DocumentInput.from_text("Test content")

        doc_dict = {
            "source": doc.source,
            "source_type": doc.source_type.value,
            "metadata": doc.metadata or {},
        }

        assert doc_dict["source"] == "Test content"
        assert doc_dict["source_type"] == "TEXT"
        assert isinstance(doc_dict["metadata"], dict)


class TestAsyncIntegration:
    """Integration tests for async document submission flow."""

    def test_submit_and_track_flow(self):
        """Test complete flow: submit -> check status -> get result."""
        client = GraphRAGClient()
        monitor = TaskMonitoring()

        documents = [DocumentInput.from_text("Test document")]

        with patch("insta_rag.tasks.graph_rag_tasks.add_documents_task") as mock_task:
            # Setup mock task
            mock_result = MagicMock()
            task_id = "test-task-123"
            mock_result.id = task_id
            mock_task.apply_async.return_value = mock_result

            # Submit
            submit_result = client.submit_add_documents_async(documents, "test")
            assert submit_result["task_id"] == task_id

            # Status check
            with patch("insta_rag.task_monitoring.AsyncResult") as mock_async:
                mock_async_instance = MagicMock()
                mock_async_instance.state = "SUCCESS"
                mock_async_instance.ready.return_value = True
                mock_async_instance.successful.return_value = True
                mock_async.return_value = mock_async_instance

                status = monitor.get_task_status(task_id)
                assert status["state"] == "SUCCESS"
                assert status["ready"] is True


# Fixtures for integration testing
@pytest.fixture
def celery_config():
    """Fixture providing CeleryConfig."""
    return CeleryConfig()


@pytest.fixture
def graph_rag_client():
    """Fixture providing GraphRAGClient."""
    return GraphRAGClient()


@pytest.fixture
def task_monitoring():
    """Fixture providing TaskMonitoring service."""
    return TaskMonitoring()
