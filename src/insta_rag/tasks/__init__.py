"""Celery tasks for async operations in Insta RAG."""

from .graph_rag_tasks import add_documents_task

__all__ = ["add_documents_task"]
