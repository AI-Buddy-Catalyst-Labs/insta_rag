"""Celery application configuration for Insta RAG async task processing."""

import os

from celery import Celery

# Initialize Celery app
app = Celery("insta_rag")

# Get broker and backend URLs from environment
# IMPORTANT: These MUST be set via environment variables (e.g., from .env file)
# They are required for async task processing to work
celery_broker_url = os.getenv("CELERY_BROKER_URL")
celery_result_backend = os.getenv("CELERY_RESULT_BACKEND")

# Validate that required Redis configuration is present
if not celery_broker_url:
    print(
        "[Celery] WARNING: CELERY_BROKER_URL not set in environment. "
        "Async tasks may not work properly. "
        "Set CELERY_BROKER_URL in your .env file or environment."
    )
if not celery_result_backend:
    print(
        "[Celery] WARNING: CELERY_RESULT_BACKEND not set in environment. "
        "Task results may not be stored. "
        "Set CELERY_RESULT_BACKEND in your .env file or environment."
    )

# Celery configuration
app.conf.update(
    broker_url=celery_broker_url,
    result_backend=celery_result_backend,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,  # Track task started state
    task_time_limit=3600,  # 1 hour hard time limit
    task_soft_time_limit=3300,  # 55 minutes soft time limit
    worker_prefetch_multiplier=1,  # Don't prefetch tasks (better for long-running)
    worker_max_tasks_per_child=1000,  # Recycle worker after 1000 tasks
    result_expires=86400,  # Results expire after 24 hours
)

# Import tasks to register them with Celery
# This must happen AFTER app configuration to avoid circular imports
try:
    print("[Celery] Attempting to import tasks...")
    from insta_rag.tasks import graph_rag_tasks  # noqa: F401
    print("[Celery] ✓ Tasks imported successfully")
except ImportError as e:
    print(f"[Celery] ✗ Failed to import tasks: {e}")
except Exception as e:
    print(f"[Celery] ✗ Unexpected error importing tasks: {e}")


def get_celery_app():
    """Get the Celery app instance.

    Returns:
        Celery: The configured Celery application instance.
    """
    return app
