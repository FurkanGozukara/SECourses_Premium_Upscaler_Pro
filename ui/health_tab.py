"""
Health Check tab with structured, readable status output.
"""

import html
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr

from shared.gradio_compat import get_compatibility_report
from shared.health import collect_health_report
from shared.repo_scanner import generate_repo_scan_report


_CHECK_LABELS = {
    "gradio": "Gradio Compatibility",
    "ffmpeg": "FFmpeg",
    "cuda": "CUDA + GPUs",
    "vs_build_tools": "Visual Studio Build Tools",
    "temp_dir": "Temp Directory Access",
    "output_dir": "Output Directory Access",
    "disk_temp": "Temp Disk Space",
    "disk_output": "Output Disk Space",
}


_STATUS_META = {
    "ok": {"label": "OK", "class": "is-ok", "severity": 0},
    "warning": {"label": "Warning", "class": "is-warning", "severity": 1},
    "error": {"label": "Error", "class": "is-error", "severity": 2},
    "missing": {"label": "Missing", "class": "is-error", "severity": 2},
    "skipped": {"label": "Skipped", "class": "is-skipped", "severity": 0},
}


def _get_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent.parent
    except Exception:
        return Path.cwd()


def _status_meta(status: Any) -> Dict[str, Any]:
    normalized = str(status or "unknown").strip().lower()
    return _STATUS_META.get(
        normalized,
        {"label": normalized.title() or "Unknown", "class": "is-warning", "severity": 1},
    )


def _normalize_detail(detail: Any) -> str:
    raw = str(detail or "No detail provided.").replace("**", "").strip()
    lines = [line.rstrip() for line in raw.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    normalized = "\n".join(lines).strip() or "No detail provided."
    return html.escape(normalized)


def _render_health_report(report: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    counts = {"ok": 0, "warning": 0, "error": 0, "missing": 0, "skipped": 0}
    worst_severity = 0
    issue_labels = []
    cards = []

    for key, info in report.items():
        info = info or {}
        status = str(info.get("status", "unknown")).strip().lower()
        meta = _status_meta(status)
        counts[status] = counts.get(status, 0) + 1
        worst_severity = max(worst_severity, int(meta["severity"]))

        label = _CHECK_LABELS.get(key, key.replace("_", " ").title())
        detail_html = _normalize_detail(info.get("detail"))
        cards.append(
            f"""
            <article class="health-report-card {meta['class']}">
              <div class="health-report-card-head">
                <h4>{html.escape(label)}</h4>
                <span class="health-report-badge {meta['class']}">{html.escape(meta['label'])}</span>
              </div>
              <pre class="health-report-detail">{detail_html}</pre>
            </article>
            """
        )

        if status not in ("ok", "skipped"):
            issue_labels.append(label)

    overall_class = "is-ok"
    overall_title = "All checks passed"
    if worst_severity >= 2:
        overall_class = "is-error"
        overall_title = "Action required"
    elif worst_severity == 1:
        overall_class = "is-warning"
        overall_title = "Review warnings"

    total = len(report)
    ok_count = counts.get("ok", 0)
    skipped_count = counts.get("skipped", 0)
    warning_count = counts.get("warning", 0)
    error_count = counts.get("error", 0) + counts.get("missing", 0)

    summary_html = (
        f"""
        <section class="health-report-summary {overall_class}">
          <div class="health-report-summary-title">{overall_title}</div>
          <div class="health-report-metrics">
            <span><strong>{total}</strong> checks</span>
            <span><strong>{ok_count}</strong> OK</span>
            <span><strong>{warning_count}</strong> warnings</span>
            <span><strong>{error_count}</strong> errors</span>
            <span><strong>{skipped_count}</strong> skipped</span>
          </div>
        </section>
        """
    )

    report_html = (
        '<div class="health-report-shell">'
        + summary_html
        + '<section class="health-report-grid">'
        + "".join(cards)
        + "</section>"
        + "</div>"
    )
    if issue_labels:
        banner_text = "Health issues detected: " + ", ".join(issue_labels)
    else:
        banner_text = "All health checks passed."
    return report_html, banner_text


def _render_initial_health_placeholder() -> str:
    return """
    <div class="health-report-shell">
      <section class="health-report-summary is-warning">
        <div class="health-report-summary-title">Health check not run yet</div>
        <div class="health-report-metrics">
          <span>Click <strong>Run Health Check</strong> to generate a full report.</span>
        </div>
      </section>
    </div>
    """


def health_tab(global_settings: Dict[str, Any], shared_state: gr.State, temp_dir: Path, output_dir: Path):
    """
    Self-contained Health Check tab.
    """

    def run_health_check(state):
        report = collect_health_report(temp_dir=temp_dir, output_dir=output_dir)
        report_html, health_text = _render_health_report(report)

        next_state = state if isinstance(state, dict) else {}
        health_banner = next_state.setdefault("health_banner", {})
        health_banner["text"] = health_text

        return report_html, health_text, next_state

    def run_gradio_scan():
        try:
            return get_compatibility_report()
        except Exception as exc:
            return f"Gradio scan failed: {exc}"

    def run_repo_scan(base_dir: Path):
        try:
            return generate_repo_scan_report(base_dir)
        except Exception as exc:
            return f"Repository scan failed: {exc}"

    gr.Markdown("### System Health Check")
    gr.Markdown("Run diagnostics for Gradio, FFmpeg, CUDA/GPU, VS Build Tools, and disk paths.")

    with gr.Row():
        health_btn = gr.Button(
            "Run Health Check",
            variant="primary",
            size="lg",
            elem_classes=["action-btn", "action-btn-preview"],
        )

    health_report = gr.HTML(
        value=_render_initial_health_placeholder(),
        elem_classes=["health-report-root"],
    )

    with gr.Accordion("What Each Check Does", open=False):
        gr.Markdown(
            """
            - **Gradio compatibility**: Verifies version and required feature support.
            - **FFmpeg**: Confirms `ffmpeg` is available in PATH for video processing.
            - **CUDA + GPUs**: Detects NVIDIA GPUs, CUDA version, and free VRAM.
            - **VS Build Tools**: Validates MSVC toolchain availability for `torch.compile` on Windows.
            - **Directory access**: Checks write access for temp and output folders.
            - **Disk space**: Checks available free space for temp and output paths.
            """
        )

    with gr.Accordion("Troubleshooting Tips", open=False):
        gr.Markdown(
            """
            **FFmpeg**
            - Install FFmpeg and add it to PATH.
            - Windows: https://ffmpeg.org/download.html

            **CUDA**
            - Install NVIDIA drivers and CUDA runtime.
            - Validate with: `nvidia-smi`
            - Validate PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

            **VS Build Tools (Windows)**
            - Install Visual Studio Build Tools.
            - Include the **Desktop development with C++** workload.

            **Permissions / Disk**
            - Ensure temp/output folders are writable.
            - Ensure enough free disk space for temporary and final outputs.
            """
        )

    health_status = gr.Markdown("", visible=False)

    with gr.Accordion("Gradio Source Scan", open=False):
        gr.Markdown("Scan installed Gradio package for components and feature surface.")
        gradio_scan_btn = gr.Button("Scan Gradio Installation", variant="secondary")
        gradio_scan_report = gr.Markdown("Click to scan Gradio source...", buttons=["copy"])

    with gr.Accordion("Repository Scan (SeedVR2, Real-ESRGAN, OMDB)", open=False):
        gr.Markdown("Scan external repositories for recent commits and changes.")
        repo_scan_btn = gr.Button("Scan Repositories", variant="secondary")
        repo_scan_report = gr.Markdown("Click to scan repositories...", buttons=["copy"])

    health_btn.click(
        fn=run_health_check,
        inputs=shared_state,
        outputs=[health_report, health_status, shared_state],
    )

    gradio_scan_btn.click(
        fn=run_gradio_scan,
        outputs=gradio_scan_report,
    )

    repo_scan_btn.click(
        fn=lambda: run_repo_scan(_get_base_dir()),
        outputs=repo_scan_report,
    )
