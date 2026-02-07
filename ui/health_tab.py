"""
Health Check Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.health import collect_health_report
from shared.gradio_compat import get_compatibility_report
from shared.repo_scanner import generate_repo_scan_report


def health_tab(global_settings: Dict[str, Any], shared_state: gr.State, temp_dir: Path, output_dir: Path):
    """
    Self-contained Health Check tab.
    All logic internal to this function.
    """

    def run_health_check(state, base_dir):
        """Run comprehensive health check"""
        report = collect_health_report(temp_dir=temp_dir, output_dir=output_dir)
        lines = []
        warnings = []

        for key, info in report.items():
            status_icon = {
                "ok": "✅",
                "warning": "⚠️",
                "error": "❌",
                "skipped": "⏭️"
            }.get(info.get("status"), "❓")

            line = f"**{status_icon} {key}**: {info.get('status')} - {info.get('detail')}"
            lines.append(line)

            if info.get("status") not in ("ok", "skipped"):
                warnings.append(line)

        # Update shared state health banner
        health_text = "\n".join(warnings) if warnings else "✅ All health checks passed."
        state["health_banner"]["text"] = health_text

        report_text = "\n".join(lines)
        return report_text, health_text, state
    
    def run_gradio_scan():
        """Run Gradio compatibility check with source scan"""
        try:
            return get_compatibility_report()
        except Exception as e:
            return f"❌ Gradio scan failed: {str(e)}"
    
    def run_repo_scan(base_dir):
        """Scan external repositories for recent changes"""
        try:
            return generate_repo_scan_report(base_dir)
        except Exception as e:
            return f"❌ Repository scan failed: {str(e)}"

    # Layout
    gr.Markdown("### 🏥 System Health Check")
    gr.Markdown("Verify ffmpeg, CUDA, VS Build Tools, and disk/temp/output writability.")

    with gr.Row():
        health_btn = gr.Button(
            "🔍 Run Health Check",
            variant="primary",
            size="lg"
        )

    health_report = gr.Markdown(
        "Click 'Run Health Check' to verify system components.",
        buttons=["copy"]
    )

    # Info sections
    with gr.Accordion("ℹ️ What Each Check Does", open=False):
        gr.Markdown("""
        - **ffmpeg**: Checks if FFmpeg is available in PATH (required for video processing)
        - **CUDA**: Detects NVIDIA GPU availability and driver status
        - **VS Build Tools**: Windows-only check for Visual Studio Build Tools (needed for torch.compile)
        - **Disk Space**: Verifies adequate free space in temp and output directories
        - **Directory Access**: Confirms read/write permissions for temp and output folders
        """)

    with gr.Accordion("🔧 Troubleshooting Tips", open=False):
        gr.Markdown("""
        **FFmpeg Issues:**
        - Ensure FFmpeg is installed and added to your system PATH
        - On Windows: Download from https://ffmpeg.org/download.html
        - On Linux: `sudo apt install ffmpeg` or equivalent

        **CUDA Issues:**
        - Install NVIDIA drivers and CUDA toolkit
        - Verify GPU is detected with `nvidia-smi`
        - Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

        **VS Build Tools Issues (Windows):**
        - Install Visual Studio Build Tools from Microsoft's website
        - Include "Desktop development with C++" workload
        - torch.compile will be disabled if not found

        **Permission Issues:**
        - Ensure the application has read/write access to temp and output directories
        - Check antivirus software isn't blocking file operations
        """)

    # Health status message (shows what changed)
    health_status = gr.Markdown("", visible=False)
    
    # Additional scans
    with gr.Accordion("🔍 Gradio Source Scan", open=False):
        gr.Markdown("Scan installed Gradio package for components and features")
        
        gradio_scan_btn = gr.Button("🔍 Scan Gradio Installation", variant="secondary")
        gradio_scan_report = gr.Markdown("Click button to scan Gradio source...", buttons=["copy"])
    
    with gr.Accordion("📦 Repository Scan (SeedVR2, Real-ESRGAN, OMDB)", open=False):
        gr.Markdown("Scan external repositories for recent commits and features")
        
        repo_scan_btn = gr.Button("🔍 Scan Repositories", variant="secondary")
        repo_scan_report = gr.Markdown("Click button to scan repositories...", buttons=["copy"])
    
    # Wire up the health check
    # The banner updates automatically via shared_state.change() in main app
    
    # Get base_dir from parent context
    from pathlib import Path as PathLib
    import sys
    
    # Derive base_dir from module path
    def get_base_dir():
        try:
            # Navigate up from ui/ to get base directory
            return PathLib(__file__).parent.parent.resolve()
        except:
            return PathLib.cwd()
    
    health_btn.click(
        fn=lambda state: run_health_check(state, get_base_dir()),
        inputs=shared_state,
        outputs=[health_report, health_status, shared_state]
    )
    
    gradio_scan_btn.click(
        fn=run_gradio_scan,
        outputs=gradio_scan_report
    )
    
    repo_scan_btn.click(
        fn=lambda: run_repo_scan(get_base_dir()),
        outputs=repo_scan_report
    )

    # Auto-run health check on tab load would require tab-level load event
    # Since we're inside a with gr.Tab() context, we can't attach .load()
    # Health check runs on button click instead
