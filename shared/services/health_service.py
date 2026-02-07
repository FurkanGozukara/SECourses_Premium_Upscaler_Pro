from pathlib import Path
from typing import Dict, Any
import gradio as gr

from shared.health import collect_health_report


def build_health_callbacks(global_settings: Dict[str, Any], shared_state: gr.State = None):
    def health_check_action(state=None):
        state = state or {"health_banner": {"text": ""}}
        report = collect_health_report(temp_dir=Path(global_settings["temp_dir"]), output_dir=Path(global_settings["output_dir"]))
        lines = []
        warnings = []
        for key, info in report.items():
            line = f"**{key}**: {info.get('status')} - {info.get('detail')}"
            lines.append(line)
            if info.get("status") not in ("ok", "skipped"):
                warnings.append(line)
        health_text = "\n".join(warnings) if warnings else "All health checks passed."
        state["health_banner"]["text"] = health_text
        return "\n".join(lines), health_text, state

    return {"health_check_action": health_check_action}


