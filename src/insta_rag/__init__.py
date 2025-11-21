"""insta_rag - A modular RAG library for document processing and retrieval."""

import importlib.metadata

__version__ = importlib.metadata.version("insta_rag")

from .core.client import RAGClient
from .core.config import RAGConfig
from .models.document import DocumentInput
from .models.response import AddDocumentsResponse
from .worker_manager import start_worker, stop_worker, is_worker_running, get_worker_pid
from .worker_pool import (
    start_worker_pool,
    stop_worker_pool,
    get_pool_status,
    get_active_worker_count,
    is_pool_healthy,
    scale_pool,
    get_queue_depth,
    auto_scale_if_needed,
)

__all__ = [
    "RAGClient",
    "RAGConfig",
    "DocumentInput",
    "AddDocumentsResponse",
    "start_worker",
    "stop_worker",
    "is_worker_running",
    "get_worker_pid",
    "start_worker_pool",
    "stop_worker_pool",
    "get_pool_status",
    "get_active_worker_count",
    "is_pool_healthy",
    "scale_pool",
    "get_queue_depth",
    "auto_scale_if_needed",
]
