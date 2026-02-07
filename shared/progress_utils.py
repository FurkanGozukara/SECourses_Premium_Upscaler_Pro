"""
Progress Tracking Utilities with gr.Progress Integration

Provides helpers for tracking progress with Gradio's Progress component.
"""

from typing import Optional, Callable
import time


class ProgressTracker:
    """
    Progress tracker that works with gr.Progress from Gradio.
    
    Usage:
        def process(input_data, progress=gr.Progress()):
            tracker = ProgressTracker(progress, total=100, desc="Processing")
            
            for i in range(100):
                # Do work
                tracker.update(i + 1)
            
            return result
    """
    
    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        total: Optional[int] = None,
        desc: str = "Processing",
        unit: str = "steps"
    ):
        """
        Initialize progress tracker.
        
        Args:
            progress_callback: gr.Progress() instance or callable
            total: Total number of steps (None for unknown)
            desc: Description to display
            unit: Unit name for steps
        """
        self.progress_callback = progress_callback
        self.total = total
        self.desc = desc
        unit = unit
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1, desc: Optional[str] = None, force: bool = False):
        """
        Update progress by n steps.
        
        Args:
            n: Number to increment (or absolute value if force=True)
            desc: Optional description override
            force: If True, set current to n instead of incrementing
        """
        if force:
            self.current = n
        else:
            self.current += n
        
        if desc:
            self.desc = desc
        
        self._report()
    
    def set_progress(self, current: int, total: Optional[int] = None, desc: Optional[str] = None):
        """
        Set absolute progress.
        
        Args:
            current: Current step number
            total: Optional total override
            desc: Optional description override
        """
        self.current = current
        if total is not None:
            self.total = total
        if desc:
            self.desc = desc
        
        self._report()
    
    def _report(self):
        """Report progress to gr.Progress callback"""
        if not self.progress_callback:
            return
        
        try:
            if self.total and self.total > 0:
                # Known total - use tuple format
                self.progress_callback((self.current, self.total), desc=self.desc)
            else:
                # Unknown total - use None to show indeterminate
                self.progress_callback(None, desc=self.desc)
        except Exception:
            # Graceful degradation if progress reporting fails
            pass
    
    def get_eta(self) -> Optional[float]:
        """
        Get estimated time remaining in seconds.
        
        Returns:
            Seconds remaining, or None if cannot estimate
        """
        if not self.total or self.current <= 0:
            return None
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        
        if rate <= 0:
            return None
        
        remaining = self.total - self.current
        return remaining / rate
    
    def format_eta(self) -> str:
        """Get formatted ETA string"""
        eta = self.get_eta()
        if eta is None:
            return "ETA: Unknown"
        
        if eta < 60:
            return f"ETA: {int(eta)}s"
        elif eta < 3600:
            mins = int(eta / 60)
            secs = int(eta % 60)
            return f"ETA: {mins}m {secs}s"
        else:
            hours = int(eta / 3600)
            mins = int((eta % 3600) / 60)
            return f"ETA: {hours}h {mins}m"
    
    def tqdm(self, iterable, desc: Optional[str] = None):
        """
        Wrap an iterable with progress tracking (like tqdm).
        
        Args:
            iterable: Iterable to wrap
            desc: Optional description
            
        Usage:
            tracker = ProgressTracker(progress, total=100)
            for item in tracker.tqdm(items, desc="Processing items"):
                # do work
        """
        if desc:
            self.desc = desc
        
        # If we can get length, use it
        try:
            length = len(iterable)
            if self.total is None:
                self.total = length
        except (TypeError, AttributeError):
            pass
        
        for i, item in enumerate(iterable):
            self.set_progress(i, desc=self.desc)
            yield item
        
        # Final update
        if self.total:
            self.set_progress(self.total, desc=f"{self.desc} - Complete")


def format_progress_message(current: int, total: int, desc: str = "", eta_seconds: Optional[float] = None) -> str:
    """
    Format a progress message string.
    
    Args:
        current: Current step
        total: Total steps
        desc: Description
        eta_seconds: Estimated time remaining
        
    Returns:
        Formatted string like "Processing: 50/100 (50%) - ETA: 2m 30s"
    """
    if total > 0:
        percent = (current / total) * 100
        msg = f"{desc}: {current}/{total} ({percent:.1f}%)"
    else:
        msg = f"{desc}: {current}"
    
    if eta_seconds is not None:
        if eta_seconds < 60:
            msg += f" - ETA: {int(eta_seconds)}s"
        elif eta_seconds < 3600:
            mins = int(eta_seconds / 60)
            secs = int(eta_seconds % 60)
            msg += f" - ETA: {mins}m {secs}s"
        else:
            hours = int(eta_seconds / 3600)
            mins = int((eta_seconds % 3600) / 60)
            msg += f" - ETA: {hours}h {mins}m"
    
    return msg


class ChunkProgressTracker(ProgressTracker):
    """
    Extended progress tracker for chunked processing.
    
    Tracks both chunk-level and overall progress.
    """
    
    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        total_chunks: int = 0,
        frames_per_chunk: int = 0,
        desc: str = "Processing chunks"
    ):
        super().__init__(progress_callback, total=total_chunks, desc=desc, unit="chunks")
        self.total_chunks = total_chunks
        self.frames_per_chunk = frames_per_chunk
        self.current_chunk = 0
        self.current_frame_in_chunk = 0
    
    def update_chunk(self, chunk_num: int, desc: Optional[str] = None):
        """Update to specific chunk number"""
        self.current_chunk = chunk_num
        self.current_frame_in_chunk = 0
        
        if desc is None:
            desc = f"Processing chunk {chunk_num + 1}/{self.total_chunks}"
        
        self.set_progress(chunk_num, desc=desc)
    
    def update_frame_in_chunk(self, frame_num: int, total_frames_in_chunk: int):
        """Update progress within current chunk"""
        self.current_frame_in_chunk = frame_num
        
        # Calculate overall progress
        chunks_done = self.current_chunk
        chunk_progress = frame_num / max(1, total_frames_in_chunk)
        overall_progress = (chunks_done + chunk_progress) / max(1, self.total_chunks)
        
        desc = f"Chunk {self.current_chunk + 1}/{self.total_chunks} - Frame {frame_num}/{total_frames_in_chunk}"
        
        if self.progress_callback:
            try:
                # Report as fraction
                self.progress_callback(overall_progress, desc=desc)
            except Exception:
                pass

