"""
Progress Tracking with ETA Calculation

Provides intelligent progress tracking for long-running operations:
- Progress percentage calculation
- ETA (Estimated Time of Arrival) based on current speed
- Smooth progress updates
- Support for multi-step operations
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List
from statistics import mean


@dataclass
class ProgressTracker:
    """
    Track progress and calculate ETA for operations.
    
    Features:
    - Percentage-based progress tracking
    - ETA calculation based on moving average of processing speed
    - Support for multi-step operations (chunks, batch, etc.)
    - Smooth progress updates
    """
    
    total_steps: int = 100
    current_step: int = 0
    start_time: Optional[float] = None
    step_times: List[float] = field(default_factory=list)
    max_history: int = 10  # Keep last N step times for moving average
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def start(self):
        """Start or restart tracking"""
        self.start_time = time.time()
        self.current_step = 0
        self.step_times.clear()
    
    def update(self, current: int, total: Optional[int] = None):
        """
        Update progress.
        
        Args:
            current: Current step number
            total: Total steps (optional, updates total if provided)
        """
        if total is not None:
            self.total_steps = max(1, total)
        
        self.current_step = min(current, self.total_steps)
        
        # Track step completion time
        now = time.time()
        if len(self.step_times) > 0:
            step_duration = now - self.step_times[-1]
        else:
            step_duration = now - (self.start_time or now)
        
        self.step_times.append(now)
        
        # Keep only recent history
        if len(self.step_times) > self.max_history:
            self.step_times = self.step_times[-self.max_history:]
    
    def increment(self, amount: int = 1):
        """Increment current step by amount"""
        self.update(self.current_step + amount)
    
    def get_percentage(self) -> float:
        """Get current progress as percentage (0-100)"""
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_eta_seconds(self) -> Optional[float]:
        """
        Calculate ETA (Estimated Time of Arrival) in seconds.
        
        Returns:
            Seconds until completion, or None if cannot calculate
        """
        if self.current_step <= 0 or self.total_steps <= 0:
            return None
        
        if self.current_step >= self.total_steps:
            return 0.0
        
        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            return None
        
        # Calculate average time per step
        if len(self.step_times) >= 2:
            # Use moving average of recent step times
            recent_durations = []
            for i in range(1, len(self.step_times)):
                duration = self.step_times[i] - self.step_times[i-1]
                recent_durations.append(duration)
            
            if recent_durations:
                avg_time_per_step = mean(recent_durations)
            else:
                avg_time_per_step = elapsed / self.current_step
        else:
            # Fallback to simple average
            avg_time_per_step = elapsed / self.current_step
        
        remaining_steps = self.total_steps - self.current_step
        return avg_time_per_step * remaining_steps
    
    def format_time(self, seconds: float) -> str:
        """
        Format seconds into human-readable time.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted string like "2h 15m 30s" or "45s"
        """
        if seconds <= 0:
            return "0s"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    def get_eta_formatted(self) -> str:
        """
        Get formatted ETA string.
        
        Returns:
            String like "ETA: 2h 15m" or "Calculating..." or "Complete"
        """
        if self.current_step >= self.total_steps:
            return "✅ Complete"
        
        eta_seconds = self.get_eta_seconds()
        if eta_seconds is None:
            return "⏳ Calculating ETA..."
        
        if eta_seconds <= 0:
            return "✅ Almost done"
        
        return f"⏱️ ETA: {self.format_time(eta_seconds)}"
    
    def get_progress_message(self) -> str:
        """
        Get comprehensive progress message.
        
        Returns:
            Message like "45% complete (23/50) - ETA: 2h 15m"
        """
        percentage = self.get_percentage()
        eta = self.get_eta_formatted()
        elapsed = self.format_time(self.get_elapsed_time())
        
        return f"📊 {percentage:.1f}% ({self.current_step}/{self.total_steps}) | Elapsed: {elapsed} | {eta}"


def create_chunked_progress_tracker(total_chunks: int) -> ProgressTracker:
    """Create a progress tracker for chunked processing"""
    return ProgressTracker(total_steps=total_chunks)


def create_batch_progress_tracker(total_files: int) -> ProgressTracker:
    """Create a progress tracker for batch processing"""
    return ProgressTracker(total_steps=total_files)


def create_frame_progress_tracker(total_frames: int) -> ProgressTracker:
    """Create a progress tracker for frame-by-frame processing"""
    return ProgressTracker(total_steps=total_frames)


__all__ = [
    'ProgressTracker',
    'create_chunked_progress_tracker',
    'create_batch_progress_tracker',
    'create_frame_progress_tracker',
]

