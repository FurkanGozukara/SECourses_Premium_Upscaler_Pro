"""
Command Logger - Stores all executed upscale commands for analysis

Logs all commands from:
- SeedVR2 (Video/Image)
- RIFE (FPS Interpolation)
- GAN (Image-Based Upscaling)
- FlashVSR+ (Real-Time Diffusion)
- Face Restoration

Each command is logged with:
- Timestamp
- Tab/Model type
- Full command with arguments
- Settings/parameters
- Exit code and status
- Output path
- Error logs (if any)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class CommandLogger:
    """Centralized command logging for all upscale operations"""
    
    def __init__(self, log_dir: Path):
        """
        Initialize command logger
        
        Args:
            log_dir: Directory to store command logs (executed_commands folder)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each tab/model type
        self.subdirs = {
            "seedvr2": self.log_dir / "seedvr2",
            "rife": self.log_dir / "rife",
            "gan": self.log_dir / "gan",
            "flashvsr": self.log_dir / "flashvsr",
            "face_restoration": self.log_dir / "face_restoration",
            "realesrgan": self.log_dir / "realesrgan",
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def log_command(
        self,
        tab_name: str,
        command: List[str],
        settings: Dict[str, Any],
        returncode: Optional[int] = None,
        output_path: Optional[str] = None,
        error_logs: Optional[List[str]] = None,
        execution_time: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Log an executed command to disk
        
        Args:
            tab_name: Name of the tab/model (seedvr2, rife, gan, etc.)
            command: Full command as list of strings
            settings: Dictionary of all settings/parameters
            returncode: Exit code (0 = success, non-zero = error)
            output_path: Path to output file/folder
            error_logs: List of error messages (if any)
            execution_time: Time taken to execute (seconds)
            additional_info: Any additional metadata
        
        Returns:
            Path to the created log file
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Determine log subdirectory
        log_subdir = self.subdirs.get(tab_name.lower(), self.log_dir)
        
        # Create log filename with timestamp
        log_filename = f"{timestamp_str}_{tab_name}.json"
        log_path = log_subdir / log_filename
        
        # Build log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "tab": tab_name,
            "command": command,
            "command_string": " ".join(f'"{c}"' if " " in c else c for c in command),
            "settings": settings,
            "returncode": returncode,
            "status": self._get_status_string(returncode),
            "output_path": output_path,
            "execution_time_seconds": execution_time,
            "error_logs": error_logs or [],
            "additional_info": additional_info or {}
        }
        
        # Write to JSON file
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            # Also append to master log (all commands in one file)
            self._append_to_master_log(log_entry)
            
            return log_path
        except Exception as e:
            print(f"⚠️ Failed to write command log: {e}")
            return log_path
    
    def _get_status_string(self, returncode: Optional[int]) -> str:
        """Convert return code to human-readable status"""
        if returncode is None:
            return "unknown"
        elif returncode == 0:
            return "success"
        elif returncode == -1:
            return "cancelled"
        else:
            return "failed"
    
    def _append_to_master_log(self, log_entry: Dict[str, Any]):
        """Append entry to master log file (all commands)"""
        master_log = self.log_dir / "all_commands.jsonl"
        
        try:
            with open(master_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Failed to append to master log: {e}")
    
    def log_batch_command(
        self,
        tab_name: str,
        batch_info: Dict[str, Any],
        individual_commands: List[Dict[str, Any]]
    ) -> Path:
        """
        Log a batch processing operation
        
        Args:
            tab_name: Name of the tab/model
            batch_info: Overall batch metadata
            individual_commands: List of individual command logs
        
        Returns:
            Path to the batch log file
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        log_subdir = self.subdirs.get(tab_name.lower(), self.log_dir)
        log_filename = f"{timestamp_str}_{tab_name}_BATCH.json"
        log_path = log_subdir / log_filename
        
        batch_log = {
            "timestamp": timestamp.isoformat(),
            "tab": tab_name,
            "type": "batch",
            "batch_info": batch_info,
            "total_files": len(individual_commands),
            "successful": sum(1 for c in individual_commands if c.get("returncode") == 0),
            "failed": sum(1 for c in individual_commands if c.get("returncode") != 0),
            "commands": individual_commands
        }
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(batch_log, f, indent=2, ensure_ascii=False)
            return log_path
        except Exception as e:
            print(f"⚠️ Failed to write batch log: {e}")
            return log_path
    
    def get_recent_logs(self, tab_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent command logs
        
        Args:
            tab_name: Filter by tab name (None = all tabs)
            limit: Maximum number of logs to return
        
        Returns:
            List of log entries (most recent first)
        """
        logs = []
        
        # Determine which directories to search
        if tab_name:
            search_dirs = [self.subdirs.get(tab_name.lower(), self.log_dir)]
        else:
            search_dirs = list(self.subdirs.values()) + [self.log_dir]
        
        # Collect all log files
        log_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                log_files.extend(search_dir.glob("*.json"))
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Read and return logs
        for log_file in log_files[:limit]:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs.append(json.load(f))
            except Exception:
                continue
        
        return logs
    
    def get_failed_commands(self, tab_name: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent failed commands for error analysis
        
        Args:
            tab_name: Filter by tab name (None = all tabs)
            limit: Maximum number of logs to return
        
        Returns:
            List of failed command logs
        """
        all_logs = self.get_recent_logs(tab_name, limit=limit * 2)  # Get more to filter
        failed = [log for log in all_logs if log.get("returncode") != 0]
        return failed[:limit]
    
    def generate_summary_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a summary report of all logged commands
        
        Args:
            output_path: Optional path to save report (markdown format)
        
        Returns:
            Report text (markdown)
        """
        # Count commands by tab
        tab_counts = {}
        success_counts = {}
        failed_counts = {}
        
        for tab_name, subdir in self.subdirs.items():
            if not subdir.exists():
                continue
            
            log_files = list(subdir.glob("*.json"))
            tab_counts[tab_name] = len(log_files)
            
            success = 0
            failed = 0
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log = json.load(f)
                        if log.get("returncode") == 0:
                            success += 1
                        else:
                            failed += 1
                except Exception:
                    continue
            
            success_counts[tab_name] = success
            failed_counts[tab_name] = failed
        
        # Build report
        report_lines = [
            "# Upscale Command Execution Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nLog Directory: `{self.log_dir}`",
            "\n## Summary by Tab\n",
            "| Tab | Total Commands | Successful | Failed |",
            "|-----|----------------|------------|--------|"
        ]
        
        for tab_name in sorted(tab_counts.keys()):
            total = tab_counts[tab_name]
            success = success_counts.get(tab_name, 0)
            failed = failed_counts.get(tab_name, 0)
            report_lines.append(f"| {tab_name} | {total} | {success} | {failed} |")
        
        report_lines.extend([
            "\n## Recent Failed Commands\n",
            "Last 10 failed commands for error analysis:\n"
        ])
        
        failed_logs = self.get_failed_commands(limit=10)
        for i, log in enumerate(failed_logs, 1):
            report_lines.append(f"\n### {i}. {log.get('tab', 'unknown')} - {log.get('timestamp', 'unknown')}")
            report_lines.append(f"- **Status**: {log.get('status', 'unknown')}")
            report_lines.append(f"- **Return Code**: {log.get('returncode', 'N/A')}")
            report_lines.append(f"- **Command**: `{log.get('command_string', 'N/A')[:100]}...`")
            if log.get('error_logs'):
                report_lines.append(f"- **Error**: {log['error_logs'][-1][:200]}...")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
            except Exception as e:
                print(f"⚠️ Failed to save report: {e}")
        
        return report_text


# Global singleton instance
_command_logger: Optional[CommandLogger] = None


def get_command_logger(log_dir: Optional[Path] = None) -> CommandLogger:
    """
    Get or create the global command logger instance
    
    Args:
        log_dir: Directory for logs (only used on first call)
    
    Returns:
        CommandLogger instance
    """
    global _command_logger
    
    if _command_logger is None:
        if log_dir is None:
            # Default to executed_commands in project root
            from pathlib import Path
            log_dir = Path(__file__).parent.parent.parent / "executed_commands"
        
        _command_logger = CommandLogger(log_dir)
    
    return _command_logger

