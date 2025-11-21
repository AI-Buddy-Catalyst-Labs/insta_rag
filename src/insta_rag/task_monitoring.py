"""Task monitoring and tracking service for Celery-based async operations."""

import logging
from typing import Any, Dict, List, Optional

from celery import states
from celery.result import AsyncResult

from insta_rag.celery_app import app

logger = logging.getLogger(__name__)


class TaskMonitoring:
    """Service for monitoring and tracking Celery task execution.

    Provides methods to check task status, retrieve results, monitor queues,
    and get worker statistics for the document ingestion pipeline.
    """

    def __init__(self):
        """Initialize TaskMonitoring service with the Celery app."""
        self.celery_app = app

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task.

        Args:
            task_id: The Celery task ID

        Returns:
            Dict with keys:
                - 'task_id': The task ID
                - 'state': Current state (PENDING, STARTED, SUCCESS, FAILURE, etc.)
                - 'ready': Boolean indicating if task is complete
                - 'successful': Boolean indicating if task succeeded
                - 'failed': Boolean indicating if task failed
                - 'progress': Dict with 'current' and 'total' keys (if available)
                - 'result': Task result if state is SUCCESS
                - 'error': Error message if state is FAILURE
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)

            status_dict = {
                "task_id": task_id,
                "state": result.state,
                "ready": result.ready(),
                "successful": result.successful() if result.ready() else None,
                "failed": result.failed() if result.ready() else None,
            }

            # Add progress info if available
            if result.state == "PROGRESS":
                status_dict["progress"] = result.info
            elif result.ready() and result.state == states.SUCCESS:
                status_dict["result"] = result.result
            elif result.ready() and result.state == states.FAILURE:
                status_dict["error"] = str(result.info)

            return status_dict
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "state": "UNKNOWN",
                "error": str(e),
            }

    def get_task_result(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Any:
        """Get the result of a task, waiting if necessary.

        Args:
            task_id: The Celery task ID
            timeout: Maximum time to wait in seconds (None = infinite)

        Returns:
            The task result (dict with success/failure info)

        Raises:
            TimeoutError: If timeout is exceeded
            Exception: If the task failed
        """
        result = AsyncResult(task_id, app=self.celery_app)
        return result.get(timeout=timeout, propagate=True)

    def is_task_ready(self, task_id: str) -> bool:
        """Check if a task has completed (succeeded or failed).

        Args:
            task_id: The Celery task ID

        Returns:
            True if task is complete, False otherwise
        """
        result = AsyncResult(task_id, app=self.celery_app)
        return result.ready()

    def is_task_successful(self, task_id: str) -> bool:
        """Check if a task completed successfully.

        Args:
            task_id: The Celery task ID

        Returns:
            True if task succeeded, False otherwise
        """
        result = AsyncResult(task_id, app=self.celery_app)
        return result.successful() if result.ready() else False

    def is_task_failed(self, task_id: str) -> bool:
        """Check if a task failed.

        Args:
            task_id: The Celery task ID

        Returns:
            True if task failed, False otherwise
        """
        result = AsyncResult(task_id, app=self.celery_app)
        return result.failed() if result.ready() else False

    def get_all_queued_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all tasks currently in queues (pending and reserved).

        Returns:
            Dict with worker names as keys and lists of task info as values.
            Each task info dict contains: name, id, args, kwargs, eta, etc.
        """
        try:
            inspector = self.celery_app.control.inspect()

            # Get reserved (not yet executed) tasks
            reserved = inspector.reserved() or {}
            # Get active (currently executing) tasks
            active = inspector.active() or {}

            queued_tasks = {}

            # Combine reserved tasks (pending)
            for worker, tasks in reserved.items():
                if tasks:
                    queued_tasks[f"{worker}_reserved"] = tasks

            # Add info that tasks are active
            for worker, tasks in active.items():
                if tasks:
                    queued_tasks[f"{worker}_active"] = tasks

            return queued_tasks

        except Exception as e:
            logger.error(f"Error getting queued tasks: {e}")
            return {}

    def get_active_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all currently executing tasks.

        Returns:
            Dict with worker names as keys and lists of active task info as values.
        """
        try:
            inspector = self.celery_app.control.inspect()
            active = inspector.active() or {}

            return {
                worker: tasks
                for worker, tasks in active.items()
                if tasks
            }

        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return {}

    def get_reserved_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all reserved tasks (pending execution).

        Returns:
            Dict with worker names as keys and lists of reserved task info as values.
        """
        try:
            inspector = self.celery_app.control.inspect()
            reserved = inspector.reserved() or {}

            return {
                worker: tasks
                for worker, tasks in reserved.items()
                if tasks
            }

        except Exception as e:
            logger.error(f"Error getting reserved tasks: {e}")
            return {}

    def get_queue_length(self) -> int:
        """Get total number of tasks in all queues.

        Returns:
            Total count of queued tasks
        """
        try:
            active = self.get_active_tasks()
            reserved = self.get_reserved_tasks()

            active_count = sum(len(tasks) for tasks in active.values())
            reserved_count = sum(len(tasks) for tasks in reserved.values())

            return active_count + reserved_count

        except Exception as e:
            logger.error(f"Error getting queue length: {e}")
            return 0

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about all active workers.

        Returns:
            Dict with worker names as keys and stats as values.
            Stats include: pool size, active tasks, processed tasks, etc.
        """
        try:
            inspector = self.celery_app.control.inspect()

            # Get stats for all workers
            stats = inspector.stats() or {}

            worker_info = {}
            for worker_name, worker_stats in stats.items():
                active_tasks = self.get_active_tasks().get(worker_name, [])
                reserved_tasks = self.get_reserved_tasks().get(worker_name, [])

                worker_info[worker_name] = {
                    "pool_size": worker_stats.get("pool", {}).get("max-concurrency", "N/A"),
                    "active_tasks": len(active_tasks),
                    "reserved_tasks": len(reserved_tasks),
                    "total_tasks": len(active_tasks) + len(reserved_tasks),
                    "processed": worker_stats.get("total", {}),
                    "status": "online",
                }

            return worker_info

        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            return {}

    def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """Cancel a running task.

        Args:
            task_id: The Celery task ID
            terminate: If True, force terminate; if False, wait for graceful shutdown

        Returns:
            True if cancellation was successful
        """
        try:
            if terminate:
                self.celery_app.control.revoke(task_id, terminate=True)
            else:
                self.celery_app.control.revoke(task_id, terminate=False)

            logger.info(f"Task {task_id} cancelled successfully")
            return True

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    def get_task_info_summary(self, task_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of a task's information.

        Args:
            task_id: The Celery task ID

        Returns:
            Dict with comprehensive task information
        """
        status = self.get_task_status(task_id)

        summary = {
            "task_id": task_id,
            "state": status.get("state"),
            "is_ready": status.get("ready"),
            "is_successful": status.get("successful"),
            "is_failed": status.get("failed"),
        }

        if status.get("progress"):
            summary["progress"] = status["progress"]

        if status.get("result"):
            summary["result"] = status["result"]

        if status.get("error"):
            summary["error"] = status["error"]

        return summary


# Global instance
_task_monitoring = None


def get_task_monitoring() -> TaskMonitoring:
    """Get or create the global TaskMonitoring instance.

    Returns:
        TaskMonitoring instance
    """
    global _task_monitoring
    if _task_monitoring is None:
        _task_monitoring = TaskMonitoring()
    return _task_monitoring
