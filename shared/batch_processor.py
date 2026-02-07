"""
Batch Processing System for Multiple Files

Provides intelligent batch processing with:
- Queue management for multiple files
- Progress tracking across files
- Error handling and partial completion
- Resume capability for interrupted batches
- Resource management and cleanup
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .path_utils import normalize_path, resolve_output_location, detect_input_type
from .logging_utils import RunLogger


@dataclass
class BatchJob:
    """Represents a single file in the batch"""
    input_path: str
    output_path: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed, skipped
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Tracks overall batch progress"""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    current_file: Optional[str] = None
    overall_progress: float = 0.0
    estimated_time_remaining: Optional[float] = None
    start_time: Optional[float] = None
    jobs: List[BatchJob] = field(default_factory=list)


class BatchProcessor:
    """
    Intelligent batch processor for upscaling multiple files.

    Features:
    - Concurrent processing with configurable thread limits
    - Progress tracking and ETA calculation
    - Error handling with partial completion
    - Resume capability for interrupted batches
    - Resource cleanup and memory management
    """

    def __init__(
        self,
        max_workers: int = 1,  # Default to 1 for GPU-bound operations
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        logger: Optional[RunLogger] = None,
        output_dir: Optional[str] = None,
        telemetry_enabled: bool = True
    ):
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.logger = logger or RunLogger(enabled=telemetry_enabled)
        self.output_dir = output_dir
        self.telemetry_enabled = telemetry_enabled
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    def discover_files(self, input_dir: str, supported_extensions: List[str]) -> List[str]:
        """Discover all supported files in input directory"""
        input_path = Path(normalize_path(input_dir))
        if not input_path.exists() or not input_path.is_dir():
            return []

        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f"**/*{ext}"))

        # Sort for consistent processing order
        return sorted([str(f) for f in files])

    def create_batch_jobs(
        self,
        input_files: List[str],
        settings: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> List[BatchJob]:
        """Create batch jobs for each input file"""
        jobs = []

        for input_file in input_files:
            # Predict output path for this file
            predicted_output = resolve_output_location(
                input_path=input_file,
                output_format=settings.get("output_format", "auto"),
                global_output_dir=output_dir or settings.get("output_dir"),
                batch_mode=False,  # Each file gets its own output
                png_padding=settings.get("png_padding", 6),  # Match CLI default
                png_keep_basename=settings.get("png_keep_basename", True),
            )

            job = BatchJob(
                input_path=input_file,
                output_path=predicted_output,
                metadata={
                    "input_type": detect_input_type(input_file),
                    "settings": settings.copy()
                }
            )
            jobs.append(job)

        return jobs

    def process_batch(
        self,
        jobs: List[BatchJob],
        processor_func: Callable[[BatchJob], bool],
        max_concurrent: int = 1
    ) -> BatchProgress:
        """
        Process a batch of jobs with progress tracking.
        
        Uses single processing pipeline - processor_func should handle ONE job
        and reuse the main processing function (no duplicate code).

        Args:
            jobs: List of BatchJob objects to process
            processor_func: Function that takes a BatchJob and returns success bool
                           This function should call the single-file processor
            max_concurrent: Maximum number of concurrent jobs (usually 1 for GPU-bound tasks)
        """
        progress = BatchProgress(
            total_files=len(jobs),
            jobs=jobs,
            start_time=time.time()
        )

        if self.progress_callback:
            self.progress_callback(progress)

        # Sequential processing (GPU-bound operations can't truly parallelize anyway)
        for idx, job in enumerate(jobs):
            if self._cancel_event.is_set():
                # Mark remaining jobs as skipped
                for remaining_job in jobs[idx:]:
                    if remaining_job.status == "pending":
                        remaining_job.status = "skipped"
                        progress.skipped_files += 1
                break

            try:
                job.status = "processing"
                job.start_time = time.time()
                progress.current_file = job.input_path

                # Call the processor function
                success = processor_func(job)

                job.end_time = time.time()
                
                with self._lock:
                    if job.status == "skipped":
                        progress.skipped_files += 1
                    elif success:
                        job.status = "completed"
                        progress.completed_files += 1
                        
                        # Write per-file summary for successful completions
                        if self.logger and job.output_path:
                            try:
                                output_path = Path(job.output_path)
                                self.logger.write_summary(
                                    output_path,
                                    {
                                        "input": job.input_path,
                                        "output": job.output_path,
                                        "status": "completed",
                                        "processing_time": job.end_time - job.start_time if job.start_time else 0,
                                        "batch_processing": True,
                                        "metadata": job.metadata
                                    }
                                )
                            except Exception:
                                pass  # Non-critical metadata write
                    else:
                        job.status = "failed"
                        progress.failed_files += 1

                    processed = progress.completed_files + progress.failed_files + progress.skipped_files
                    progress.overall_progress = processed / progress.total_files

                    # Calculate ETA
                    elapsed = time.time() - (progress.start_time or time.time())
                    if processed > 0:
                        avg_time_per_file = elapsed / processed
                        remaining_files = progress.total_files - processed
                        progress.estimated_time_remaining = avg_time_per_file * remaining_files

            except Exception as e:
                with self._lock:
                    progress.failed_files += 1
                    job.status = "failed"
                    job.error_message = str(e)
                    job.end_time = time.time()
                
                # Log error (RunLogger doesn't have log_error, just write summary with error status)
                if self.logger and job.output_path:
                    try:
                        self.logger.write_summary(
                            Path(job.output_path).parent if job.output_path else Path(self.output_dir or "."),
                            {
                                "input": job.input_path,
                                "status": "failed",
                                "error": str(e),
                                "batch_processing": True
                            }
                        )
                    except Exception:
                        pass

            # Callback after each job
            if self.progress_callback:
                self.progress_callback(progress)

        # Mark final status
        progress.current_file = None
        if self.progress_callback:
            self.progress_callback(progress)
        
        # Write batch summary
        if self.logger and self.output_dir:
            try:
                batch_summary_path = Path(self.output_dir) / "batch_summary.json"
                self.logger.write_summary(
                    batch_summary_path.parent,
                    {
                        "batch_processing": True,
                        "total_files": progress.total_files,
                        "completed_files": progress.completed_files,
                        "failed_files": progress.failed_files,
                        "skipped_files": progress.skipped_files,
                        "overall_progress": progress.overall_progress,
                        "total_time": time.time() - (progress.start_time or time.time()),
                        "jobs": [
                            {
                                "input": job.input_path,
                                "output": job.output_path,
                                "status": job.status,
                                "error": job.error_message,
                                "processing_time": (job.end_time - job.start_time) if (job.start_time and job.end_time) else None
                            }
                            for job in jobs
                        ]
                    }
                )
            except Exception:
                pass  # Non-critical summary write

        return progress

    def _process_single_job(self, job: BatchJob, processor_func: Callable[[BatchJob], bool]) -> bool:
        """Process a single job with error handling"""
        try:
            job.status = "processing"
            job.start_time = time.time()

            success = processor_func(job)

            job.end_time = time.time()
            job.status = "completed" if success else "failed"

            return success

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = time.time()
            return False

    def cancel(self) -> bool:
        """Cancel current batch processing"""
        self._cancel_event.set()
        return True

    def is_cancelled(self) -> bool:
        """Check if batch processing was cancelled"""
        return self._cancel_event.is_set()

    def save_batch_state(self, progress: BatchProgress, state_file: Path) -> None:
        """Save batch progress for resume capability"""
        state_data = {
            "total_files": progress.total_files,
            "completed_files": progress.completed_files,
            "failed_files": progress.failed_files,
            "start_time": progress.start_time,
            "jobs": [
                {
                    "input_path": job.input_path,
                    "output_path": job.output_path,
                    "status": job.status,
                    "error_message": job.error_message,
                    "start_time": job.start_time,
                    "end_time": job.end_time,
                    "progress": job.progress,
                    "metadata": job.metadata
                }
                for job in progress.jobs
            ]
        }

        with state_file.open("w") as f:
            json.dump(state_data, f, indent=2, default=str)

    def load_batch_state(self, state_file: Path) -> Optional[BatchProgress]:
        """Load batch progress for resume"""
        if not state_file.exists():
            return None

        try:
            with state_file.open("r") as f:
                state_data = json.load(f)

            jobs = []
            for job_data in state_data["jobs"]:
                job = BatchJob(
                    input_path=job_data["input_path"],
                    output_path=job_data.get("output_path"),
                    status=job_data.get("status", "pending"),
                    error_message=job_data.get("error_message"),
                    start_time=job_data.get("start_time"),
                    end_time=job_data.get("end_time"),
                    progress=job_data.get("progress", 0.0),
                    metadata=job_data.get("metadata", {})
                )
                jobs.append(job)

            progress = BatchProgress(
                total_files=state_data["total_files"],
                completed_files=state_data["completed_files"],
                failed_files=state_data["failed_files"],
                start_time=state_data.get("start_time"),
                jobs=jobs
            )

            progress.overall_progress = (progress.completed_files + progress.failed_files) / progress.total_files

            return progress

        except Exception:
            return None

    def get_batch_summary(self, progress: BatchProgress) -> Dict[str, Any]:
        """Generate a summary of batch processing results"""
        total_time = time.time() - (progress.start_time or time.time())

        return {
            "total_files": progress.total_files,
            "completed_files": progress.completed_files,
            "failed_files": progress.failed_files,
            "skipped_files": progress.skipped_files,
            "success_rate": progress.completed_files / max(1, progress.total_files),
            "total_time_seconds": total_time,
            "average_time_per_file": total_time / max(1, progress.completed_files),
            "jobs": [
                {
                    "input": job.input_path,
                    "output": job.output_path,
                    "status": job.status,
                    "error": job.error_message,
                    "processing_time": (job.end_time - job.start_time) if job.start_time and job.end_time else None
                }
                for job in progress.jobs
            ]
        }
