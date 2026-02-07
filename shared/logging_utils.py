import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """Persist per-run JSON summaries alongside outputs."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def write_summary(self, output_path: Path, payload: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        target_dir = output_path if output_path.is_dir() else output_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        summary_path = target_dir / "run_summary.json"
        payload = payload.copy()
        payload.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Load existing entries if file exists
        existing_entries = []
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        existing_entries = json.loads(content)
                        # Ensure it's a list
                        if not isinstance(existing_entries, list):
                            existing_entries = [existing_entries]
            except (json.JSONDecodeError, Exception):
                # If file is corrupted or not valid JSON, start fresh
                existing_entries = []
        
        # Append new entry
        existing_entries.append(payload)
        
        # Write back all entries
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(existing_entries, f, indent=2)
        
        return summary_path


