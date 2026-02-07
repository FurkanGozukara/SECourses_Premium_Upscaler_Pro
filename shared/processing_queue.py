"""
Application-level processing queue manager.

This queue enforces a single active processing job across tabs while allowing
additional generate/upscale requests to wait in FIFO order.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Sequence


@dataclass
class QueueTicket:
    """Represents one queued processing request."""

    job_id: str
    tab_name: str
    action_name: str
    submitted_at: float
    start_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class ProcessingQueueManager:
    """Thread-safe FIFO queue with one active slot and cancelable waiting jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._id_counter = itertools.count(1)
        self._active: Optional[QueueTicket] = None
        self._waiting: Deque[QueueTicket] = deque()

    def submit(self, tab_name: str, action_name: str) -> QueueTicket:
        """Submit a new job to queue or start immediately if idle."""
        ticket = QueueTicket(
            job_id=f"Q{next(self._id_counter):06d}",
            tab_name=str(tab_name or "Unknown"),
            action_name=str(action_name or "Run"),
            submitted_at=time.time(),
        )
        with self._lock:
            if self._active is None:
                self._active = ticket
                ticket.start_event.set()
            else:
                self._waiting.append(ticket)
        return ticket

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._active and self._active.job_id == job_id)

    def waiting_position(self, job_id: str) -> int:
        """Return 1-based queue position for waiting jobs, 0 if not waiting."""
        with self._lock:
            for idx, item in enumerate(self._waiting, start=1):
                if item.job_id == job_id:
                    return idx
        return 0

    def complete(self, job_id: str) -> bool:
        """
        Mark an active job completed and promote next waiting job.

        Returns True when the active job was released.
        """
        with self._lock:
            if not self._active or self._active.job_id != job_id:
                return False
            self._active = None
            self._promote_next_locked()
            return True

    def cancel_waiting(self, job_ids: Sequence[str]) -> List[str]:
        """
        Cancel one or more waiting jobs (never cancels active job).

        Returns list of canceled job IDs.
        """
        wanted = {str(j) for j in (job_ids or []) if str(j).strip()}
        if not wanted:
            return []

        canceled: List[str] = []
        with self._lock:
            kept: Deque[QueueTicket] = deque()
            while self._waiting:
                item = self._waiting.popleft()
                if item.job_id in wanted:
                    item.cancel_event.set()
                    canceled.append(item.job_id)
                else:
                    kept.append(item)
            self._waiting = kept
        return canceled

    def cancel_all_waiting(self) -> List[str]:
        """Cancel and remove every waiting job."""
        with self._lock:
            canceled = [item.job_id for item in self._waiting]
            while self._waiting:
                item = self._waiting.popleft()
                item.cancel_event.set()
        return canceled

    def snapshot(self) -> Dict[str, Any]:
        """Get queue snapshot for UI rendering."""
        now = time.time()
        with self._lock:
            active = self._ticket_to_view(self._active, now) if self._active else None
            waiting = [self._ticket_to_view(item, now, idx + 1) for idx, item in enumerate(self._waiting)]
        return {
            "active": active,
            "waiting": waiting,
            "waiting_count": len(waiting),
        }

    def _promote_next_locked(self) -> None:
        while self._waiting:
            next_item = self._waiting.popleft()
            if next_item.cancel_event.is_set():
                continue
            self._active = next_item
            next_item.start_event.set()
            return
        self._active = None

    @staticmethod
    def _ticket_to_view(ticket: QueueTicket, now: float, position: Optional[int] = None) -> Dict[str, Any]:
        waited = max(0.0, now - float(ticket.submitted_at))
        return {
            "job_id": ticket.job_id,
            "tab_name": ticket.tab_name,
            "action_name": ticket.action_name,
            "submitted_at": ticket.submitted_at,
            "submitted_at_text": datetime.fromtimestamp(ticket.submitted_at).strftime("%Y-%m-%d %H:%M:%S"),
            "wait_seconds": waited,
            "wait_seconds_text": f"{waited:.1f}s",
            "position": position or 0,
        }


_QUEUE_MANAGER: Optional[ProcessingQueueManager] = None
_QUEUE_MANAGER_LOCK = threading.Lock()


def get_processing_queue_manager() -> ProcessingQueueManager:
    """Get shared singleton queue manager for the app process."""
    global _QUEUE_MANAGER
    with _QUEUE_MANAGER_LOCK:
        if _QUEUE_MANAGER is None:
            _QUEUE_MANAGER = ProcessingQueueManager()
        return _QUEUE_MANAGER
