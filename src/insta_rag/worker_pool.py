"""Celery worker pool management for horizontal scaling.

This module provides utilities to manage multiple Celery workers for horizontal
scaling across multiple instances or containers.
"""

import subprocess
import logging
import time
from typing import List, Optional, Dict, Any

from .task_monitoring import get_task_monitoring

logger = logging.getLogger(__name__)

# Global worker processes registry
_worker_processes: List[subprocess.Popen] = []
_worker_configs: Dict[str, Dict[str, Any]] = {}


def start_worker_pool(
    num_workers: int = 2,
    queue: str = "default",
    loglevel: str = "info",
    concurrency_per_worker: int = 4,
    log_dir: str = "/tmp",
    auto_scale: bool = False,
) -> List[int]:
    """Start a pool of Celery workers for horizontal scaling.

    This starts multiple worker processes that all connect to the same Redis
    broker, enabling horizontal scaling of document processing tasks.

    Args:
        num_workers: Number of worker processes to start (default: 2)
        queue: Queue to listen to (default: 'default')
        loglevel: Logging level: debug, info, warning (default: 'info')
        concurrency_per_worker: Concurrent tasks per worker (default: 4)
        log_dir: Directory for worker logs (default: '/tmp')
        auto_scale: Enable auto-scaling based on queue depth (default: False)

    Returns:
        List of process IDs (PIDs) of started workers

    Example:
        >>> from insta_rag.worker_pool import start_worker_pool
        >>> pids = start_worker_pool(num_workers=4, concurrency_per_worker=8)
        >>> print(f"Started {len(pids)} workers: {pids}")
    """
    global _worker_processes, _worker_configs

    if num_workers < 1:
        logger.error("num_workers must be >= 1")
        return []

    try:
        # Check if workers are already running
        running_workers = [p for p in _worker_processes if p.poll() is None]
        if running_workers:
            logger.warning(
                f"✓ {len(running_workers)} workers already running. "
                f"Call stop_worker_pool() first to restart."
            )
            return [p.pid for p in running_workers]

        started_pids = []

        for worker_id in range(num_workers):
            worker_name = f"worker{worker_id + 1}"
            log_file_path = f"{log_dir}/celery_{worker_name}.log"

            try:
                log_file = open(log_file_path, "a")

                worker_process = subprocess.Popen(
                    [
                        "celery",
                        "-A",
                        "insta_rag.celery_app",
                        "worker",
                        "-n",
                        worker_name,
                        "-l",
                        loglevel,
                        "-c",
                        str(concurrency_per_worker),
                        "-Q",
                        queue,
                    ],
                    stdout=log_file,
                    stderr=log_file,
                    start_new_session=True,
                )

                _worker_processes.append(worker_process)
                _worker_configs[worker_name] = {
                    "pid": worker_process.pid,
                    "queue": queue,
                    "concurrency": concurrency_per_worker,
                    "log_file": log_file_path,
                    "started_at": time.time(),
                }

                started_pids.append(worker_process.pid)
                logger.info(
                    f"✓ Worker {worker_name} started (PID: {worker_process.pid}, "
                    f"concurrency: {concurrency_per_worker})"
                )

            except Exception as e:
                logger.error(f"✗ Failed to start worker {worker_name}: {e}")
                continue

        if started_pids:
            logger.info(
                f"✓ Started {len(started_pids)} workers. "
                f"Log directory: {log_dir}"
            )
            if auto_scale:
                logger.info("✓ Auto-scaling enabled. Monitoring queue depth...")

        return started_pids

    except Exception as e:
        logger.error(f"✗ Failed to start worker pool: {e}")
        return []


def stop_worker_pool(timeout: int = 10) -> bool:
    """Stop all Celery workers in the pool.

    Args:
        timeout: Seconds to wait for graceful shutdown (default: 10)

    Returns:
        True if all workers stopped successfully, False otherwise

    Example:
        >>> from insta_rag.worker_pool import stop_worker_pool
        >>> success = stop_worker_pool(timeout=10)
        >>> print("All workers stopped" if success else "Some workers remained")
    """
    global _worker_processes, _worker_configs

    if not _worker_processes:
        logger.info("No workers running")
        return True

    all_stopped = True

    for worker_process in _worker_processes:
        if worker_process.poll() is None:  # Still running
            try:
                worker_process.terminate()

                try:
                    worker_process.wait(timeout=timeout)
                    logger.info(f"✓ Worker PID {worker_process.pid} stopped gracefully")
                except subprocess.TimeoutExpired:
                    worker_process.kill()
                    logger.warning(
                        f"⚠ Worker PID {worker_process.pid} force-killed "
                        f"(graceful shutdown timeout)"
                    )
                    all_stopped = False

            except Exception as e:
                logger.error(f"✗ Error stopping worker PID {worker_process.pid}: {e}")
                all_stopped = False

    _worker_processes.clear()
    _worker_configs.clear()

    if all_stopped:
        logger.info("✓ All workers stopped gracefully")
    else:
        logger.warning("⚠ Some workers required force-kill")

    return all_stopped


def get_pool_status() -> Dict[str, Any]:
    """Get the status of the worker pool.

    Returns:
        Dict with pool information and statistics

    Example:
        >>> from insta_rag.worker_pool import get_pool_status
        >>> status = get_pool_status()
        >>> print(f"Active workers: {status['active_workers']}")
        >>> print(f"Queue depth: {status['queue_depth']}")
    """
    global _worker_processes, _worker_configs

    active_workers = [p for p in _worker_processes if p.poll() is None]
    monitoring = get_task_monitoring()

    queue_depth = monitoring.get_queue_length()
    worker_stats = monitoring.get_worker_stats()

    total_concurrency = sum(
        config.get("concurrency", 0) for config in _worker_configs.values()
    )

    return {
        "total_workers": len(_worker_processes),
        "active_workers": len(active_workers),
        "inactive_workers": len(_worker_processes) - len(active_workers),
        "total_concurrency": total_concurrency,
        "queue_depth": queue_depth,
        "workers": list(_worker_configs.keys()),
        "worker_details": _worker_configs,
        "worker_stats": worker_stats,
    }


def get_active_worker_count() -> int:
    """Get number of active workers.

    Returns:
        Number of currently running workers
    """
    global _worker_processes
    return sum(1 for p in _worker_processes if p.poll() is None)


def is_pool_healthy() -> bool:
    """Check if worker pool is healthy.

    Returns:
        True if at least one worker is active, False otherwise
    """
    return get_active_worker_count() > 0


def scale_pool(target_workers: int, timeout: int = 10) -> bool:
    """Scale worker pool to target number of workers.

    This function adjusts the pool size by stopping excess workers or
    starting additional workers as needed.

    Args:
        target_workers: Target number of workers to maintain
        timeout: Seconds to wait for graceful shutdown (default: 10)

    Returns:
        True if scaling succeeded, False otherwise

    Example:
        >>> from insta_rag.worker_pool import scale_pool
        >>> success = scale_pool(target_workers=8)
        >>> print("Scaled to 8 workers" if success else "Scaling failed")
    """
    global _worker_processes

    current_active = get_active_worker_count()

    if current_active == target_workers:
        logger.info(f"✓ Pool already at target size: {target_workers} workers")
        return True

    if current_active > target_workers:
        logger.info(f"Scaling down from {current_active} to {target_workers} workers")
        return stop_worker_pool(timeout=timeout)

    # Scale up
    logger.info(f"Scaling up from {current_active} to {target_workers} workers")
    additional_workers = target_workers - current_active

    # Get config from first worker to maintain consistency
    if _worker_configs:
        first_config = list(_worker_configs.values())[0]
        queue = first_config.get("queue", "default")
        concurrency = first_config.get("concurrency", 4)

        pids = start_worker_pool(
            num_workers=additional_workers,
            queue=queue,
            concurrency_per_worker=concurrency,
        )
        return len(pids) == additional_workers

    return False


def get_queue_depth() -> int:
    """Get current queue depth (number of pending tasks).

    Returns:
        Number of tasks waiting in queue
    """
    monitoring = get_task_monitoring()
    return monitoring.get_queue_length()


def auto_scale_if_needed(
    queue_depth_threshold: int = 50,
    min_workers: int = 2,
    max_workers: int = 8,
) -> Optional[int]:
    """Auto-scale worker pool based on queue depth.

    Increases workers if queue depth exceeds threshold,
    decreases if queue is empty.

    Args:
        queue_depth_threshold: Queue depth to trigger scaling up (default: 50)
        min_workers: Minimum workers to maintain (default: 2)
        max_workers: Maximum workers to allow (default: 8)

    Returns:
        New worker count if scaled, None if no scaling occurred

    Example:
        >>> from insta_rag.worker_pool import auto_scale_if_needed
        >>> new_count = auto_scale_if_needed(
        ...     queue_depth_threshold=30,
        ...     min_workers=2,
        ...     max_workers=10
        ... )
        >>> if new_count:
        ...     print(f"Scaled to {new_count} workers")
    """
    current_active = get_active_worker_count()
    queue_depth = get_queue_depth()

    logger.info(f"Queue depth: {queue_depth}, Active workers: {current_active}")

    # Scale up if queue is backing up
    if queue_depth > queue_depth_threshold and current_active < max_workers:
        new_count = min(current_active + 2, max_workers)
        logger.info(
            f"Queue depth ({queue_depth}) exceeds threshold ({queue_depth_threshold}). "
            f"Scaling up to {new_count} workers"
        )
        if scale_pool(new_count):
            return new_count
        return None

    # Scale down if queue is empty and we have extra workers
    if queue_depth == 0 and current_active > min_workers:
        new_count = max(current_active - 1, min_workers)
        logger.info(
            f"Queue empty. Scaling down to {new_count} workers"
        )
        if scale_pool(new_count):
            return new_count
        return None

    return None
