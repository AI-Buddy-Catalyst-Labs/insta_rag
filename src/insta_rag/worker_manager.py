"""Celery worker lifecycle management for Insta RAG.

This module provides utilities to start and stop Celery workers programmatically,
useful for applications that want to manage workers without manual terminal commands.
"""

import subprocess
import atexit
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global worker process
_celery_worker_process: Optional[subprocess.Popen] = None


def start_worker(
    log_file: str = "/tmp/celery_worker.log",
    queue: str = "default",
    loglevel: str = "info",
    concurrency: int = 4,
) -> Optional[int]:
    """Start Celery worker in a background subprocess.

    This function starts a Celery worker process that handles async document
    ingestion tasks. The worker runs independently and processes tasks from
    the Redis queue.

    Args:
        log_file: Path to write worker logs (default: /tmp/celery_worker.log)
        queue: Queue to listen to (default: 'default')
        loglevel: Logging level: debug, info, warning (default: 'info')
        concurrency: Number of concurrent tasks (default: 4)

    Returns:
        Process ID (PID) of worker if started successfully, None otherwise

    Example:
        >>> from insta_rag.worker_manager import start_worker
        >>> pid = start_worker(queue="default", loglevel="info")
        >>> print(f"Worker started with PID: {pid}")
    """
    global _celery_worker_process

    try:
        # Check if worker is already running
        if _celery_worker_process is not None and _celery_worker_process.poll() is None:
            logger.info("✓ Celery worker is already running")
            return _celery_worker_process.pid

        # Open log file for writing
        worker_log_file = open(log_file, "a")

        # Start the worker process with specified configuration
        _celery_worker_process = subprocess.Popen(
            [
                "celery",
                "-A",
                "insta_rag.celery_app",
                "worker",
                "-l",
                loglevel,
                "-c",
                str(concurrency),
                "-Q",
                queue,
            ],
            stdout=worker_log_file,
            stderr=worker_log_file,
            start_new_session=True,  # Run in new process group (detached)
        )

        logger.info(f"✓ Celery worker started (PID: {_celery_worker_process.pid})")
        logger.info(f"  Queue: {queue}")
        logger.info(f"  Concurrency: {concurrency}")
        logger.info(f"  Log file: {log_file}")

        # Register cleanup handler to stop worker on exit
        atexit.register(stop_worker)

        return _celery_worker_process.pid

    except FileNotFoundError:
        logger.error("✗ Celery not found. Install with: pip install celery")
        return None
    except Exception as e:
        logger.error(f"✗ Failed to start Celery worker: {e}")
        _celery_worker_process = None
        return None


def stop_worker(timeout: int = 5) -> bool:
    """Stop the Celery worker process gracefully.

    This function gracefully shuts down the background Celery worker.
    If graceful shutdown fails, it force-kills the process.

    Args:
        timeout: Seconds to wait for graceful shutdown (default: 5)

    Returns:
        True if stopped successfully, False otherwise

    Example:
        >>> from insta_rag.worker_manager import stop_worker
        >>> success = stop_worker(timeout=5)
        >>> print("Worker stopped" if success else "Failed to stop worker")
    """
    global _celery_worker_process

    if _celery_worker_process is None:
        return True

    try:
        # Check if process is still running
        if _celery_worker_process.poll() is None:
            # Terminate gracefully
            _celery_worker_process.terminate()

            # Wait for graceful shutdown
            try:
                _celery_worker_process.wait(timeout=timeout)
                logger.info("✓ Celery worker stopped gracefully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                _celery_worker_process.kill()
                logger.warning("⚠ Celery worker force-killed (graceful shutdown timeout)")
                return False

        _celery_worker_process = None
        return True

    except Exception as e:
        logger.error(f"✗ Error stopping Celery worker: {e}")
        _celery_worker_process = None
        return False


def is_worker_running() -> bool:
    """Check if Celery worker is currently running.

    Returns:
        True if worker is running, False otherwise

    Example:
        >>> from insta_rag.worker_manager import is_worker_running
        >>> if is_worker_running():
        ...     print("Worker is active")
    """
    global _celery_worker_process

    if _celery_worker_process is None:
        return False

    return _celery_worker_process.poll() is None


def get_worker_pid() -> Optional[int]:
    """Get the PID of the running worker process.

    Returns:
        Process ID if worker is running, None otherwise

    Example:
        >>> from insta_rag.worker_manager import get_worker_pid
        >>> pid = get_worker_pid()
        >>> if pid:
        ...     print(f"Worker PID: {pid}")
    """
    global _celery_worker_process

    if _celery_worker_process is None:
        return None

    if _celery_worker_process.poll() is None:
        return _celery_worker_process.pid

    return None
