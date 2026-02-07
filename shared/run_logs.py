"""
Run Logs System with JSON Summaries

Creates detailed JSON summaries for each processing run:
- Input/output paths and metadata
- All processing parameters
- Performance metrics (duration, speed, etc.)
- System information (GPU, memory, etc.)
- Error/warning logs
- Reproducibility information
"""

import json
import time
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class RunMetrics:
    """Performance metrics for a processing run"""
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    frames_processed: int = 0
    fps_average: float = 0.0
    peak_memory_mb: float = 0.0
    peak_vram_mb: float = 0.0
    cpu_percent_avg: float = 0.0


@dataclass
class SystemInfo:
    """System information snapshot"""
    os: str = ""
    python_version: str = ""
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    cpu_count: int = 0
    total_ram_gb: float = 0.0
    total_vram_gb: float = 0.0


@dataclass
class RunSummary:
    """Complete run summary"""
    # Metadata
    run_id: str
    timestamp: str
    model_type: str  # "seedvr2", "gan", "rife"
    model_name: str
    
    # Paths
    input_path: str
    output_path: str
    temp_dir: Optional[str] = None
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: str = "pending"  # pending, running, completed, failed, canceled
    exit_code: int = 0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: Optional[RunMetrics] = None
    system_info: Optional[SystemInfo] = None
    
    # Processing details
    batch_mode: bool = False
    batch_file_count: int = 0
    chunked_processing: bool = False
    chunk_count: int = 0
    
    # Reproducibility
    command: Optional[str] = None
    preset_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            k: asdict(v) if hasattr(v, '__dataclass_fields__') else v
            for k, v in asdict(self).items()
        }


class RunLogger:
    """
    Logger for processing runs with JSON summary generation.
    
    Features:
    - Automatic run ID generation
    - Performance metrics tracking
    - System information capture
    - JSON summary files saved alongside outputs
    - Batch run aggregation
    - Error/warning collection
    """
    
    def __init__(self, enabled: bool = True, output_dir: Optional[str] = None):
        self.enabled = enabled
        self.output_dir = Path(output_dir) if output_dir else None
        self.current_run: Optional[RunSummary] = None
        self._start_cpu: List[float] = []
        
    def start_run(
        self,
        model_type: str,
        model_name: str,
        input_path: str,
        output_path: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> RunSummary:
        """
        Start a new processing run.
        
        Returns:
            RunSummary instance for this run
        """
        if not self.enabled:
            return None
        
        # Generate run ID
        run_id = self._generate_run_id()
        
        # Create run summary
        self.current_run = RunSummary(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            model_type=model_type,
            model_name=model_name,
            input_path=input_path,
            output_path=output_path,
            parameters=parameters,
            status="running",
            metrics=RunMetrics(start_time=time.time()),
            system_info=self._capture_system_info(),
            **kwargs
        )
        
        # Start CPU monitoring
        try:
            self._start_cpu = [psutil.cpu_percent(interval=0.1) for _ in range(5)]
        except Exception:
            pass
        
        return self.current_run
    
    def update_metrics(
        self,
        frames_processed: Optional[int] = None,
        current_memory_mb: Optional[float] = None,
        current_vram_mb: Optional[float] = None
    ):
        """Update performance metrics during run"""
        if not self.enabled or not self.current_run or not self.current_run.metrics:
            return
        
        metrics = self.current_run.metrics
        
        if frames_processed is not None:
            metrics.frames_processed = frames_processed
            
            # Calculate FPS
            elapsed = time.time() - metrics.start_time
            if elapsed > 0:
                metrics.fps_average = frames_processed / elapsed
        
        if current_memory_mb is not None:
            metrics.peak_memory_mb = max(metrics.peak_memory_mb, current_memory_mb)
        
        if current_vram_mb is not None:
            metrics.peak_vram_mb = max(metrics.peak_vram_mb, current_vram_mb)
        
        # Update CPU usage
        try:
            cpu = psutil.cpu_percent(interval=None)
            if cpu > 0:
                if metrics.cpu_percent_avg == 0:
                    metrics.cpu_percent_avg = cpu
                else:
                    # Running average
                    metrics.cpu_percent_avg = (metrics.cpu_percent_avg * 0.9) + (cpu * 0.1)
        except Exception:
            pass
    
    def add_warning(self, warning: str):
        """Add a warning message to current run"""
        if not self.enabled or not self.current_run:
            return
        self.current_run.warnings.append(warning)
    
    def end_run(
        self,
        status: str = "completed",
        exit_code: int = 0,
        error_message: Optional[str] = None
    ) -> Optional[str]:
        """
        End the current run and save summary.
        
        Args:
            status: "completed", "failed", or "canceled"
            exit_code: Process exit code
            error_message: Error message if failed
            
        Returns:
            Path to saved JSON summary, or None if disabled
        """
        if not self.enabled or not self.current_run:
            return None
        
        # Update final metrics
        if self.current_run.metrics:
            self.current_run.metrics.end_time = time.time()
            self.current_run.metrics.duration_seconds = (
                self.current_run.metrics.end_time - self.current_run.metrics.start_time
            )
        
        # Update status
        self.current_run.status = status
        self.current_run.exit_code = exit_code
        if error_message:
            self.current_run.error_message = error_message
        
        # Save JSON summary
        summary_path = self._save_summary()
        
        # Clear current run
        self.current_run = None
        self._start_cpu = []
        
        return summary_path
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
    
    def _capture_system_info(self) -> SystemInfo:
        """Capture current system information"""
        info = SystemInfo()
        
        try:
            info.os = f"{platform.system()} {platform.release()}"
            info.python_version = platform.python_version()
            info.cpu_count = psutil.cpu_count()
            info.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
        
        # CUDA/GPU info
        try:
            import torch
            info.cuda_available = torch.cuda.is_available()
            if info.cuda_available:
                info.cuda_version = torch.version.cuda
                info.gpu_count = torch.cuda.device_count()
                if info.gpu_count > 0:
                    info.gpu_name = torch.cuda.get_device_name(0)
                    info.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
        
        return info
    
    def _save_summary(self) -> Optional[str]:
        """Save run summary as JSON file"""
        if not self.current_run:
            return None
        
        try:
            # Determine output location
            if self.output_dir:
                log_dir = self.output_dir / "run_logs"
            else:
                # Save next to output file
                output_path = Path(self.current_run.output_path)
                log_dir = output_path.parent / "run_logs"
            
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            model_safe = self.current_run.model_name.replace('/', '_').replace('\\', '_')
            filename = f"{self.current_run.run_id}_{model_safe}.json"
            summary_path = log_dir / filename
            
            # Save JSON
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_run.to_dict(), f, indent=2, default=str)
            
            return str(summary_path)
            
        except Exception as e:
            print(f"Warning: Failed to save run summary: {e}")
            return None
    
    def get_run_history(
        self,
        model_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent run history from saved logs.
        
        Args:
            model_type: Filter by model type (optional)
            limit: Maximum number of runs to return
            
        Returns:
            List of run summaries (most recent first)
        """
        if not self.output_dir:
            return []
        
        log_dir = self.output_dir / "run_logs"
        if not log_dir.exists():
            return []
        
        # Get all JSON files
        json_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        summaries = []
        for json_file in json_files[:limit * 2]:  # Read extra in case of filtering
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                # Filter by model type if specified
                if model_type and summary.get('model_type') != model_type:
                    continue
                
                summaries.append(summary)
                
                if len(summaries) >= limit:
                    break
                    
            except Exception:
                continue
        
        return summaries
    
    def generate_markdown_report(
        self,
        summary: Dict[str, Any]
    ) -> str:
        """Generate human-readable markdown report from run summary"""
        lines = []
        
        lines.append(f"# Processing Run Report")
        lines.append(f"")
        lines.append(f"**Run ID:** `{summary.get('run_id')}`  ")
        lines.append(f"**Timestamp:** {summary.get('timestamp')}  ")
        lines.append(f"**Status:** {summary.get('status')}  ")
        lines.append(f"")
        
        lines.append(f"## Model")
        lines.append(f"- **Type:** {summary.get('model_type')}")
        lines.append(f"- **Name:** {summary.get('model_name')}")
        if summary.get('preset_used'):
            lines.append(f"- **Preset:** {summary.get('preset_used')}")
        lines.append(f"")
        
        lines.append(f"## Files")
        lines.append(f"- **Input:** `{summary.get('input_path')}`")
        lines.append(f"- **Output:** `{summary.get('output_path')}`")
        lines.append(f"")
        
        metrics = summary.get('metrics', {})
        if metrics:
            lines.append(f"## Performance")
            lines.append(f"- **Duration:** {metrics.get('duration_seconds', 0):.1f}s")
            if metrics.get('frames_processed', 0) > 0:
                lines.append(f"- **Frames:** {metrics.get('frames_processed')}")
                lines.append(f"- **Average FPS:** {metrics.get('fps_average', 0):.2f}")
            if metrics.get('peak_memory_mb', 0) > 0:
                lines.append(f"- **Peak RAM:** {metrics.get('peak_memory_mb', 0):.0f} MB")
            if metrics.get('peak_vram_mb', 0) > 0:
                lines.append(f"- **Peak VRAM:** {metrics.get('peak_vram_mb', 0):.0f} MB")
            lines.append(f"")
        
        system = summary.get('system_info', {})
        if system:
            lines.append(f"## System")
            lines.append(f"- **OS:** {system.get('os', 'Unknown')}")
            if system.get('gpu_name'):
                lines.append(f"- **GPU:** {system.get('gpu_name')}")
            lines.append(f"")
        
        warnings = summary.get('warnings', [])
        if warnings:
            lines.append(f"## Warnings")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append(f"")
        
        if summary.get('error_message'):
            lines.append(f"## Error")
            lines.append(f"```")
            lines.append(summary.get('error_message'))
            lines.append(f"```")
            lines.append(f"")
        
        return "\n".join(lines)

