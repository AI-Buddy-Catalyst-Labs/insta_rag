"""Example: Async document ingestion using Celery for Graph RAG.

This example demonstrates how to submit document ingestion tasks to Celery,
allowing documents to be processed asynchronously without blocking the application.
The task ID is returned immediately, and you can track progress using the
TaskMonitoring service.

Prerequisites:
    - Redis server running at 52.140.76.45:6379
    - Celery worker running: celery -A insta_rag.celery_app worker -l info
    - Neo4j database configured and running
"""

import asyncio
import logging
import time
from insta_rag.graph_rag.client import GraphRAGClient
from insta_rag.models.document import DocumentInput
from insta_rag.task_monitoring import get_task_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_async_submission():
    """Example 1: Submit documents asynchronously and immediately return task ID.

    This demonstrates the key benefit of Celery integration - the API call
    returns immediately with a task ID, while the actual processing happens
    in the background.
    """
    logger.info("\n=== Example 1: Basic Async Submission ===")

    # Create documents for ingestion
    documents = [
        DocumentInput.from_text(
            "Alice works at TechCorp as a senior engineer. "
            "She specializes in machine learning and distributed systems."
        ),
        DocumentInput.from_text(
            "TechCorp is an AI company founded in 2015. "
            "It focuses on building cutting-edge AI products."
        ),
        DocumentInput.from_text(
            "Bob is the CEO of TechCorp. He has 20 years of experience in tech."
        ),
    ]

    # Initialize client (no async required for submission)
    client = GraphRAGClient()

    # Submit documents for async ingestion
    # This returns IMMEDIATELY with a task ID
    result = client.submit_add_documents_async(
        documents=documents,
        collection_name="company_info",
    )

    logger.info(f"Task submitted successfully!")
    logger.info(f"  Task ID: {result['task_id']}")
    logger.info(f"  Status: {result['status']}")
    logger.info(f"  Message: {result['message']}")
    logger.info(f"  Documents: {result['num_documents']}")

    return result["task_id"]


def example_2_monitor_task_status(task_id: str):
    """Example 2: Monitor task status and progress.

    After submitting a task, you can check its status, progress, and final results.
    """
    logger.info("\n=== Example 2: Monitor Task Status ===")

    monitor = get_task_monitoring()

    # Initial status check
    status = monitor.get_task_status(task_id)
    logger.info(f"Initial task status: {status['state']}")

    # Monitor until completion
    max_wait_time = 300  # 5 minutes max wait
    start_time = time.time()

    while not status["ready"] and (time.time() - start_time) < max_wait_time:
        logger.info(f"Task state: {status['state']}")

        if "progress" in status:
            logger.info(f"  Progress: {status['progress']}")

        time.sleep(2)  # Check every 2 seconds
        status = monitor.get_task_status(task_id)

    logger.info(f"Final task state: {status['state']}")

    if status["successful"]:
        logger.info("✓ Task completed successfully!")
        logger.info(f"Result: {status['result']}")
    elif status["failed"]:
        logger.info("✗ Task failed!")
        logger.info(f"Error: {status['error']}")

    return status


def example_3_wait_for_result(task_id: str):
    """Example 3: Wait for task result with timeout.

    This is a more direct approach - wait for the task to complete and get results.
    """
    logger.info("\n=== Example 3: Wait for Result ===")

    monitor = get_task_monitoring()

    try:
        # Wait up to 5 minutes for the result
        result = monitor.get_task_result(task_id, timeout=300)
        logger.info("✓ Task completed successfully!")
        logger.info(f"Result: {result}")
        return result
    except TimeoutError:
        logger.warning("Task did not complete within timeout period")
        return None
    except Exception as e:
        logger.error(f"Task failed with error: {e}")
        return None


def example_4_queue_monitoring():
    """Example 4: Monitor queued and active tasks.

    See what tasks are currently queued or being processed by workers.
    """
    logger.info("\n=== Example 4: Queue Monitoring ===")

    monitor = get_task_monitoring()

    # Get queue statistics
    queue_length = monitor.get_queue_length()
    logger.info(f"Total tasks in queue: {queue_length}")

    # Get active tasks (currently executing)
    active = monitor.get_active_tasks()
    if active:
        logger.info("Active tasks:")
        for worker, tasks in active.items():
            logger.info(f"  {worker}: {len(tasks)} tasks")
            for task in tasks[:2]:  # Show first 2 tasks
                logger.info(f"    - {task.get('name')}: {task.get('id')[:8]}...")
    else:
        logger.info("No active tasks")

    # Get reserved tasks (pending execution)
    reserved = monitor.get_reserved_tasks()
    if reserved:
        logger.info("Reserved (pending) tasks:")
        for worker, tasks in reserved.items():
            logger.info(f"  {worker}: {len(tasks)} tasks")
    else:
        logger.info("No reserved tasks")

    # Get worker statistics
    stats = monitor.get_worker_stats()
    if stats:
        logger.info("Worker statistics:")
        for worker_name, worker_stats in stats.items():
            logger.info(f"  {worker_name}:")
            logger.info(f"    Pool size: {worker_stats['pool_size']}")
            logger.info(f"    Active: {worker_stats['active_tasks']}")
            logger.info(f"    Reserved: {worker_stats['reserved_tasks']}")
    else:
        logger.info("No workers available")


def example_5_batch_ingestion():
    """Example 5: Submit multiple ingestion tasks concurrently.

    Submit multiple document batches to be processed in parallel by workers.
    """
    logger.info("\n=== Example 5: Batch Concurrent Ingestion ===")

    client = GraphRAGClient()
    monitor = get_task_monitoring()

    # Define multiple document batches
    batches = [
        {
            "collection": "company_info",
            "docs": [
                DocumentInput.from_text("Alice is a senior engineer at TechCorp."),
                DocumentInput.from_text("Bob is the CEO of TechCorp."),
            ],
        },
        {
            "collection": "products",
            "docs": [
                DocumentInput.from_text("TechCorp develops AI-powered analytics platform."),
                DocumentInput.from_text("The platform uses machine learning for predictions."),
            ],
        },
        {
            "collection": "partnerships",
            "docs": [
                DocumentInput.from_text("TechCorp partners with Google Cloud for infrastructure."),
            ],
        },
    ]

    # Submit all batches concurrently
    task_ids = []
    for batch in batches:
        result = client.submit_add_documents_async(
            documents=batch["docs"],
            collection_name=batch["collection"],
        )
        task_ids.append(result["task_id"])
        logger.info(
            f"Submitted batch to '{batch['collection']}': "
            f"Task ID = {result['task_id'][:8]}..."
        )

    logger.info(f"\nSubmitted {len(task_ids)} batches for parallel processing")

    # Monitor all tasks
    pending = set(task_ids)
    completed = set()

    while pending:
        for task_id in list(pending):
            status = monitor.get_task_status(task_id)

            if status["ready"]:
                if status["successful"]:
                    logger.info(f"✓ Task {task_id[:8]}... completed successfully")
                    completed.add(task_id)
                else:
                    logger.warning(f"✗ Task {task_id[:8]}... failed")
                    completed.add(task_id)

                pending.discard(task_id)

        if pending:
            logger.info(f"Waiting for {len(pending)} tasks to complete...")
            time.sleep(3)

    logger.info(f"All {len(completed)} batches completed!")


def main():
    """Run all examples."""
    logger.info("Celery Async Ingestion Examples for Graph RAG")
    logger.info("=" * 50)

    try:
        # Example 1: Submit async
        logger.info("\nStarting Example 1...")
        task_id = example_1_basic_async_submission()

        # Example 2: Monitor status
        logger.info("\nStarting Example 2...")
        time.sleep(2)  # Give task a moment to start
        example_2_monitor_task_status(task_id)

        # Example 4: Queue monitoring
        logger.info("\nStarting Example 4...")
        example_4_queue_monitoring()

        # Example 5: Batch ingestion
        logger.info("\nStarting Example 5...")
        example_5_batch_ingestion()

        logger.info("\n" + "=" * 50)
        logger.info("All examples completed!")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
