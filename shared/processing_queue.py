"""
Application-level processing queue manager.

This queue enforces FIFO ordering per processing resource while allowing
independent GPU selections to run concurrently.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from shared.gpu_utils import expand_cuda_device_spec, get_global_gpu_override


CPU_RESOURCE_KEY = "cpu"
GPU_AUTO_RESOURCE_KEY = "gpu:auto"


def _resource_sort_key(value: str) -> tuple[int, int, str]:
    text = str(value or "").strip().lower()
    if text == CPU_RESOURCE_KEY:
        return (0, -1, text)
    if text == GPU_AUTO_RESOURCE_KEY:
        return (1, -1, text)
    if text.startswith("gpu:"):
        suffix = text.split(":", 1)[1].strip()
        if suffix.isdigit():
            return (1, int(suffix), text)
    return (2, 0, text)


def _normalize_gpu_selection_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1].strip()
    return text


def queue_resource_keys_for_gpu_selection(selection: Any) -> Tuple[str, ...]:
    """
    Convert a GPU selection string into queue resource keys.

    Examples:
    - "0" -> ("gpu:0",)
    - "0,1" -> ("gpu:0", "gpu:1")
    - "cpu" -> ("cpu",)
    - "" / "auto" -> ("gpu:auto",)
    """
    text = _normalize_gpu_selection_text(selection)
    if not text:
        return (GPU_AUTO_RESOURCE_KEY,)
    if text in {"cpu", "none", "off"}:
        return (CPU_RESOURCE_KEY,)
    if text in {"auto"}:
        return (GPU_AUTO_RESOURCE_KEY,)

    expanded = expand_cuda_device_spec(text)
    resource_keys = {
        f"gpu:{int(token)}"
        for token in (part.strip() for part in str(expanded).split(","))
        if token.isdigit()
    }
    if resource_keys:
        return tuple(sorted(resource_keys, key=_resource_sort_key))

    normalized = _normalize_gpu_selection_text(expanded)
    if normalized in {"all", "auto"}:
        return (GPU_AUTO_RESOURCE_KEY,)
    return (GPU_AUTO_RESOURCE_KEY,)


def describe_queue_resources(resource_keys: Sequence[str]) -> str:
    """Human-readable resource label for queue UI."""
    keys = tuple(
        sorted(
            {
                str(key).strip().lower()
                for key in (resource_keys or [])
                if str(key).strip()
            },
            key=_resource_sort_key,
        )
    )
    if not keys:
        return "Auto GPU"
    if keys == (CPU_RESOURCE_KEY,):
        return "CPU"
    if GPU_AUTO_RESOURCE_KEY in keys:
        return "Auto GPU"

    gpu_ids = [
        key.split(":", 1)[1]
        for key in keys
        if key.startswith("gpu:") and key.split(":", 1)[1].isdigit()
    ]
    if gpu_ids:
        return f"GPU {', '.join(gpu_ids)}"
    return ", ".join(keys)


def resolve_queue_gpu_resources(
    state: Optional[Mapping[str, Any]] = None,
    global_settings: Optional[Mapping[str, Any]] = None,
) -> Tuple[Tuple[str, ...], str]:
    """
    Resolve the queue resource set from the captured app state/global settings.

    Uses the same priority as runtime GPU resolution, but preserves a manually
    supplied comma-separated GPU list when present.
    """
    seed_controls = state.get("seed_controls") if isinstance(state, Mapping) else {}
    if not isinstance(seed_controls, Mapping):
        seed_controls = {}

    raw_selection = None
    state_global = seed_controls.get("global_settings")
    if isinstance(state_global, Mapping):
        raw_selection = state_global.get("global_gpu_device")
    if raw_selection is None:
        raw_selection = seed_controls.get("global_gpu_device_val")
    if raw_selection is None and isinstance(global_settings, Mapping):
        raw_selection = global_settings.get("global_gpu_device")
    if raw_selection is None:
        raw_selection = get_global_gpu_override(seed_controls, global_settings)

    resource_keys = queue_resource_keys_for_gpu_selection(raw_selection)
    return resource_keys, describe_queue_resources(resource_keys)


@dataclass
class QueueTicket:
    """Represents one queued processing request."""

    job_id: str
    tab_name: str
    action_name: str
    submitted_at: float
    resource_keys: Tuple[str, ...] = field(default_factory=tuple)
    resource_label: str = ""
    start_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class ProcessingQueueManager:
    """Thread-safe FIFO queue with resource-aware active slots and cancelable waiting jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._id_counter = itertools.count(1)
        self._active: Dict[str, QueueTicket] = {}
        self._waiting: Deque[QueueTicket] = deque()

    def submit(
        self,
        tab_name: str,
        action_name: str,
        resource_keys: Optional[Sequence[str]] = None,
        resource_label: Optional[str] = None,
    ) -> QueueTicket:
        """Submit a new job to queue or start immediately if its resources are idle."""
        normalized_keys = tuple(
            sorted(
                {
                    str(key).strip().lower()
                    for key in (resource_keys or [])
                    if str(key).strip()
                },
                key=_resource_sort_key,
            )
        ) or (GPU_AUTO_RESOURCE_KEY,)
        ticket = QueueTicket(
            job_id=f"Q{next(self._id_counter):06d}",
            tab_name=str(tab_name or "Unknown"),
            action_name=str(action_name or "Run"),
            submitted_at=time.time(),
            resource_keys=normalized_keys,
            resource_label=str(resource_label or describe_queue_resources(normalized_keys)),
        )
        with self._lock:
            if self._can_start_locked(ticket):
                self._activate_locked(ticket)
            else:
                self._waiting.append(ticket)
        return ticket

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._active

    def waiting_position(self, job_id: str) -> int:
        with self._lock:
            return self._waiting_position_locked(job_id)

    def complete(self, job_id: str) -> bool:
        """
        Mark an active job completed and promote newly unblocked waiting jobs.

        Returns True when the active job was released.
        """
        with self._lock:
            if job_id not in self._active:
                return False
            self._active.pop(job_id, None)
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
            active_jobs = [
                self._ticket_to_view(ticket, now)
                for ticket in sorted(self._active.values(), key=lambda item: (item.submitted_at, item.job_id))
            ]
            waiting = [
                self._ticket_to_view(item, now, self._waiting_position_locked(item.job_id))
                for item in self._waiting
            ]
        return {
            "active": active_jobs[0] if len(active_jobs) == 1 else None,
            "active_jobs": active_jobs,
            "active_count": len(active_jobs),
            "waiting": waiting,
            "waiting_count": len(waiting),
        }

    def _can_start_locked(self, ticket: QueueTicket) -> bool:
        return all(
            not self._resource_sets_conflict(ticket.resource_keys, active_ticket.resource_keys)
            for active_ticket in self._active.values()
        )

    def _activate_locked(self, ticket: QueueTicket) -> None:
        self._active[ticket.job_id] = ticket
        ticket.start_event.set()

    def _waiting_position_locked(self, job_id: str) -> int:
        target: Optional[QueueTicket] = None
        ahead: List[QueueTicket] = []
        for item in self._waiting:
            if item.job_id == job_id:
                target = item
                break
            ahead.append(item)
        if target is None:
            return 0

        position = 1
        for item in ahead:
            if self._resource_sets_conflict(target.resource_keys, item.resource_keys):
                position += 1
        return position

    def _promote_next_locked(self) -> None:
        if not self._waiting:
            return
        kept: Deque[QueueTicket] = deque()
        while self._waiting:
            next_item = self._waiting.popleft()
            if next_item.cancel_event.is_set():
                continue
            if self._can_start_locked(next_item):
                self._activate_locked(next_item)
                continue
            kept.append(next_item)
        self._waiting = kept

    @staticmethod
    def _resource_sets_conflict(left: Sequence[str], right: Sequence[str]) -> bool:
        left_keys = {str(item).strip().lower() for item in (left or []) if str(item).strip()}
        right_keys = {str(item).strip().lower() for item in (right or []) if str(item).strip()}
        if not left_keys or not right_keys:
            return False

        if CPU_RESOURCE_KEY in left_keys and CPU_RESOURCE_KEY in right_keys:
            return True

        left_gpu = {key for key in left_keys if key.startswith("gpu:")}
        right_gpu = {key for key in right_keys if key.startswith("gpu:")}
        if not left_gpu or not right_gpu:
            return False
        if GPU_AUTO_RESOURCE_KEY in left_gpu or GPU_AUTO_RESOURCE_KEY in right_gpu:
            return True
        return bool(left_gpu & right_gpu)

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
            "resource_keys": list(ticket.resource_keys),
            "resource_label": ticket.resource_label or describe_queue_resources(ticket.resource_keys),
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
