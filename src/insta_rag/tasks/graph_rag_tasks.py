"""Celery tasks for Graph RAG document ingestion operations."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from insta_rag.celery_app import app
from insta_rag.graph_rag.client import GraphRAGClient
from insta_rag.models.document import DocumentInput

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="insta_rag.tasks.add_documents_task",
    track_started=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    retry_backoff=True,
    retry_backoff_max=600,  # Max retry delay: 10 minutes
    retry_jitter=True,
)
def add_documents_task(
    self,
    documents_data: List[Dict[str, Any]],
    collection_name: str = "default",
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    neo4j_database: Optional[str] = None,
    group_id: str = "insta_rag",
) -> Dict[str, Any]:
    """Add documents to Graph RAG knowledge graph asynchronously.

    This task runs as a Celery task in a background worker, allowing document
    ingestion to proceed without blocking the caller. The task returns immediately
    with a task ID, and results can be retrieved using the task monitoring service.

    Args:
        documents_data: List of document dictionaries with keys:
            - 'source': The document text or content
            - 'source_type': 'TEXT', 'FILE', or 'BINARY'
            - 'metadata': Optional dict of custom metadata
        collection_name: Name of the collection in knowledge graph (default: 'default')
        neo4j_uri: Neo4j connection URI (optional, uses env var if not provided)
        neo4j_user: Neo4j username (optional, uses env var if not provided)
        neo4j_password: Neo4j password (optional, uses env var if not provided)
        neo4j_database: Neo4j database name (optional, uses env var if not provided)
        group_id: Group ID for organizing graph data (default: 'insta_rag')

    Returns:
        Dict with keys:
            - 'status': 'success' or 'failure'
            - 'task_id': The Celery task ID
            - 'collection_name': The collection name used
            - 'total_documents': Number of documents processed
            - 'succeeded': Number of successfully added documents
            - 'failed': Number of failed documents
            - 'results': List of GraphAddResult dicts
            - 'error': Error message if status is 'failure'
            - 'processing_time_ms': Total processing time in milliseconds

    Raises:
        ValueError: If documents_data is empty or invalid
        Exception: If Neo4j connection or document processing fails
    """
    if not documents_data:
        raise ValueError("documents_data cannot be empty")

    self.update_state(state="PROGRESS", meta={"current": 0, "total": len(documents_data)})

    try:
        # Convert document dicts to DocumentInput objects
        documents = []
        for i, doc_data in enumerate(documents_data):
            try:
                source = doc_data.get("source", "")
                source_type = doc_data.get("source_type", "TEXT").upper()
                metadata = doc_data.get("metadata", {})

                # Create DocumentInput based on source type
                if source_type == "TEXT":
                    doc = DocumentInput.from_text(source)
                    if metadata:
                        doc.metadata = metadata
                elif source_type == "FILE":
                    doc = DocumentInput.from_file(source)
                    if metadata:
                        doc.metadata = metadata
                elif source_type == "BINARY":
                    doc = DocumentInput.from_binary(source)
                    if metadata:
                        doc.metadata = metadata
                else:
                    raise ValueError(f"Invalid source_type: {source_type}")

                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse document {i}: {e}")
                continue

        if not documents:
            raise ValueError("No valid documents could be parsed from documents_data")

        # Run async GraphRAGClient in sync context using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _add_documents_async(
                    documents=documents,
                    collection_name=collection_name,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    neo4j_database=neo4j_database,
                    group_id=group_id,
                    task_id=self.request.id,
                    update_progress=self.update_state,
                )
            )
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Document ingestion task failed: {e}", exc_info=True)
        raise


async def _add_documents_async(
    documents: List[DocumentInput],
    collection_name: str,
    neo4j_uri: Optional[str],
    neo4j_user: Optional[str],
    neo4j_password: Optional[str],
    neo4j_database: Optional[str],
    group_id: str,
    task_id: str,
    update_progress,
) -> Dict[str, Any]:
    """Internal async function to add documents using GraphRAGClient.

    Args:
        documents: List of DocumentInput objects
        collection_name: Collection name
        neo4j_uri: Neo4j URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database
        group_id: Group ID
        task_id: Celery task ID (for logging/tracking)
        update_progress: Function to update task progress

    Returns:
        Dict with success/failure status and results
    """
    try:
        # Initialize GraphRAGClient with optional parameters
        client = GraphRAGClient(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            group_id=group_id,
        )

        # Initialize the client
        await client.initialize()

        try:
            logger.info(
                f"[Task {task_id}] Starting document ingestion: {len(documents)} documents "
                f"to collection '{collection_name}'"
            )

            # Add documents with progress tracking
            results = await client.add_documents(documents, collection_name)

            logger.info(
                f"[Task {task_id}] Document ingestion completed: {len(results)} results"
            )

            # Convert results to serializable dicts
            results_dicts = []
            total_nodes = 0
            total_edges = 0

            for result in results:
                total_nodes += result.nodes_created
                total_edges += result.edges_created
                results_dicts.append(
                    {
                        "episode_uuid": result.episode_uuid,
                        "nodes_created": result.nodes_created,
                        "edges_created": result.edges_created,
                        "group_id": result.group_id,
                        "processing_time_ms": result.processing_time_ms,
                    }
                )

            return {
                "status": "success",
                "task_id": task_id,
                "collection_name": collection_name,
                "total_documents": len(documents),
                "succeeded": len(results),
                "failed": len(documents) - len(results),
                "total_nodes_created": total_nodes,
                "total_edges_created": total_edges,
                "results": results_dicts,
            }

        finally:
            await client.close()

    except Exception as e:
        logger.error(f"[Task {task_id}] Document ingestion failed: {e}", exc_info=True)
        return {
            "status": "failure",
            "task_id": task_id,
            "collection_name": collection_name,
            "total_documents": len(documents),
            "succeeded": 0,
            "failed": len(documents),
            "error": str(e),
        }
