import os
import sys
import hashlib
import json
import string
import argparse
from pathlib import Path
from typing import Any, Dict

# Hugging Face download transport:
# - hf_transfer can improve download speed but can also cause issues on some Windows setups.
# - Default to disabled unless the launcher/user explicitly enables it.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Fix Unicode encoding on Windows console to support emojis and special characters
if sys.platform == 'win32':
    # Force UTF-8 encoding for console output
    import io
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass  # Silently ignore if we can't change encoding

import gradio as gr

from shared.models import scan_gan_models
from shared.health import collect_health_report
from shared.logging_utils import RunLogger
from shared.path_utils import get_default_output_dir, get_default_temp_dir
from shared.preset_manager import PresetManager
from shared.universal_preset import dict_to_values, get_all_defaults
from shared.models.rife_meta import get_rife_default_model
from shared.runner import Runner
from shared.gradio_compat import check_gradio_version, check_required_features
from ui.seedvr2_tab import seedvr2_tab
from ui.resolution_tab import resolution_tab
from ui.output_tab import output_tab
from ui.face_tab import face_tab
from ui.rife_tab import rife_tab
from ui.gan_tab import gan_tab
from ui.flashvsr_tab import flashvsr_tab
from ui.health_tab import health_tab
from ui.queue_tab import queue_tab

BASE_DIR = Path(__file__).parent.resolve()
PRESET_DIR = BASE_DIR / "presets"
APP_TITLE = "SECourses Ultimate Video and Image Upscaler Pro V1.3 – https://www.patreon.com/posts/150202809"


# --------------------------------------------------------------------- #
# Global setup - Honor launcher BAT file environment variables
# --------------------------------------------------------------------- #
preset_manager = PresetManager(PRESET_DIR)

# FIXED: Read ALL launcher BAT settings (TEMP/TMP + model cache paths)
# This ensures user-configured paths from Windows_Run_SECourses_Upscaler_Pro.bat are respected
launcher_temp = os.environ.get("TEMP") or os.environ.get("TMP")
launcher_output = None  # BAT doesn't set OUTPUT_DIR, but we check for future compatibility

# FIXED: Also read model cache paths set by launcher (MODELS_DIR, HF_HOME, etc.)
# These are used by HuggingFace/Transformers for model downloads and caching
launcher_models_dir = os.environ.get("MODELS_DIR")
launcher_hf_home = os.environ.get("HF_HOME")
launcher_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
launcher_hf_datasets = os.environ.get("HF_DATASETS_CACHE")

# Validate and propagate model cache paths if set by launcher
# This ensures models download to the correct location
if launcher_models_dir and Path(launcher_models_dir).exists():
    # User set MODELS_DIR in launcher - ensure HF libraries use it
    if not launcher_hf_home:
        os.environ["HF_HOME"] = launcher_models_dir
    if not launcher_transformers_cache:
        os.environ["TRANSFORMERS_CACHE"] = launcher_models_dir
    if not launcher_hf_datasets:
        os.environ["HF_DATASETS_CACHE"] = launcher_models_dir

# If BAT file set a custom temp that's NOT the system temp, use it
# This detects if user modified the BAT file's TEMP/TMP settings
system_temp = os.environ.get("SystemRoot", "C:\\Windows") + "\\Temp" if os.name == "nt" else "/tmp"
if launcher_temp and launcher_temp.lower() != system_temp.lower():
    default_temp = launcher_temp
else:
    default_temp = str(BASE_DIR / "temp")

GLOBAL_DEFAULTS = {
    "output_dir": launcher_output or str(BASE_DIR / "outputs"),
    "temp_dir": default_temp,
    "telemetry": True,
    "face_global": False,
    "face_strength": 0.5,
    "queue_enabled": True,
    "mode": "subprocess",
    "pinned_reference_path": None,  # Global pinned reference for iterative comparison
    # FIXED: Store model cache paths - editable in UI, persisted across restarts
    "models_dir": launcher_models_dir or str(BASE_DIR / "models"),
    "hf_home": launcher_hf_home or os.environ.get("HF_HOME") or str(BASE_DIR / "models"),
    "transformers_cache": launcher_transformers_cache or os.environ.get("TRANSFORMERS_CACHE") or str(BASE_DIR / "models"),
    # Store originals for change detection (helps warn user about restart requirement)
    "_original_models_dir": launcher_models_dir,
    "_original_hf_home": launcher_hf_home,
    "_original_transformers_cache": launcher_transformers_cache,
}
global_settings = preset_manager.load_global_settings(GLOBAL_DEFAULTS)

# FIXED: Apply saved model cache paths to environment for current session
# If user previously saved custom paths, honor them immediately (though full effect requires restart)
if global_settings.get("models_dir"):
    os.environ["MODELS_DIR"] = global_settings["models_dir"]
if global_settings.get("hf_home"):
    os.environ["HF_HOME"] = global_settings["hf_home"]
if global_settings.get("transformers_cache"):
    os.environ["TRANSFORMERS_CACHE"] = global_settings["transformers_cache"]

temp_dir = get_default_temp_dir(BASE_DIR, global_settings)
output_dir = get_default_output_dir(BASE_DIR, global_settings)
runner = Runner(
    BASE_DIR,
    temp_dir=temp_dir,
    output_dir=output_dir,
    telemetry_enabled=global_settings.get("telemetry", True),
)
# Restore execution mode from saved settings (default to subprocess)
saved_mode = global_settings.get("mode", "subprocess")
try:
    runner.set_mode(saved_mode)
except Exception:
    runner.set_mode("subprocess")
    global_settings["mode"] = "subprocess"
run_logger = RunLogger(enabled=global_settings.get("telemetry", True))



# --------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------- #
def _parse_launch_cli_args(argv=None):
    """
    Parse app-level launch flags and ignore unknown args so launcher
    wrappers/scripts remain backward compatible.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--server", dest="server_name", type=str, default=None, help="Bind Gradio server host/IP")
    parser.add_argument("--port", dest="server_port", type=int, default=None, help="Bind Gradio server port")
    args, _unknown = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    return args


def _scan_disk_roots() -> list[str]:
    """
    Discover mounted disk roots so Gradio can serve/cache files outside
    the project directory (for example C:\\Users\\... paths).
    """
    roots: set[str] = set()
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive_root = Path(f"{letter}:/")
            try:
                if drive_root.exists():
                    roots.add(str(drive_root.resolve()))
            except OSError:
                continue
    else:
        roots.add(str(Path("/").resolve()))
    return sorted(roots)


def _build_launch_allowed_paths(output_dir: str | Path, temp_dir: str | Path) -> list[str]:
    allowed_paths = {
        str(Path(BASE_DIR).resolve()),
        str(Path(output_dir).resolve()),
        str(Path(temp_dir).resolve()),
    }
    allowed_paths.update(_scan_disk_roots())
    return sorted(allowed_paths)


def main(argv=None):
    launch_cli_args = _parse_launch_cli_args(argv)
    share_enabled = launch_cli_args.share

    # Initialize health check data
    try:
        initial_report = collect_health_report(temp_dir=temp_dir, output_dir=output_dir)
        warnings = []
        
        # Check Gradio compatibility FIRST (critical for UI)
        gradio_compatible, gradio_msg, gradio_features = check_gradio_version()
        if not gradio_compatible:
            warnings.append(f"⚠️ GRADIO: {gradio_msg}")
        
        required_features, features_msg = check_required_features()
        if not required_features:
            warnings.append(f"⚠️ GRADIO FEATURES: {features_msg}")
        
        for key, info in initial_report.items():
            # We handle ffmpeg messaging separately so we only show an error when it's missing.
            # Avoids always-on informational notices.
            if key == "ffmpeg":
                continue
            if info.get("status") not in ("ok", "skipped"):
                warnings.append(f"{key}: {info.get('detail')}")

        # Show ffmpeg error ONLY if ffmpeg is missing from PATH
        try:
            from shared.error_handling import check_ffmpeg_available

            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                warnings.append(ffmpeg_msg or "❌ ffmpeg not found in PATH. Please install ffmpeg and add it to your system PATH.")
        except Exception:
            # If the check itself fails, don't block startup; health tab/services will surface details later.
            pass

        vs_info = initial_report.get("vs_build_tools")
        if vs_info and vs_info.get("status") not in ("ok", "skipped"):
            # vs_build_tools already contains a detailed diagnostic (included above). Add a short,
            # accurate summary line without misleading "not detected" wording when VS is present but failing.
            detail = (vs_info.get("detail") or "").lower()
            if "not detected" in detail or "not found" in detail:
                warnings.append("VS Build Tools not detected; torch.compile will be disabled on Windows until installed.")
            else:
                warnings.append("VS Build Tools found but could not be validated; torch.compile may be unreliable. See vs_build_tools details above.")
        health_text = "\n".join(warnings) if warnings else "All health checks passed."
    except Exception:
        health_text = "Health check failed to initialize. Run Health Check tab for details."

    # Ultra-modern theme with maximum readability (Gradio 6.2.0)
    # Using Soft theme + Google Fonts for best readability
    modern_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),  # Most readable UI font
        font_mono=gr.themes.GoogleFont("JetBrains Mono")  # Best code font
    ).set(
        # Enhanced readability settings
        body_text_size="16px",
        body_text_weight="400",
        button_large_text_size="18px",
        button_large_text_weight="600",
        button_large_padding="16px 28px",
        button_border_width="2px",
        button_primary_shadow="0 2px 8px rgba(0,0,0,0.1)",
        button_primary_shadow_hover="0 4px 12px rgba(0,0,0,0.15)",
        input_border_width="2px",
        input_shadow="0 1px 3px rgba(0,0,0,0.05)",
        block_label_text_size="16px",
        block_label_text_weight="600",
        block_title_text_size="18px",
        block_title_text_weight="700",
    )

    # --------------------------------------------------------------------- #
    # Global VRAM OOM banner styling (big + flashing)
    # --------------------------------------------------------------------- #
    CUSTOM_CSS = """
    .vram-oom-banner {
      position: relative;
      border: 2px solid #ff1744;
      background: linear-gradient(90deg, rgba(255,23,68,0.16), rgba(255,193,7,0.14));
      padding: 16px 18px;
      border-radius: 14px;
      margin: 10px 0 18px 0;
      overflow: hidden;
      animation: vramPulse 1.15s ease-in-out infinite;
    }
    .vram-oom-banner::before {
      content: "";
      position: absolute;
      top: -30%;
      left: -60%;
      width: 180%;
      height: 160%;
      background: linear-gradient(120deg, rgba(255,255,255,0.0), rgba(255,255,255,0.22), rgba(255,255,255,0.0));
      transform: rotate(10deg);
      animation: vramShimmer 2.2s linear infinite;
      pointer-events: none;
      opacity: 0.55;
    }
    .vram-oom-title {
      font-size: 28px;
      font-weight: 900;
      letter-spacing: 0.4px;
      color: #ff1744;
      text-transform: uppercase;
      margin-bottom: 4px;
      text-shadow: 0 0 10px rgba(255,23,68,0.35);
    }
    .vram-oom-subtitle {
      font-size: 15px;
      opacity: 0.95;
      margin-bottom: 10px;
    }
    .vram-oom-model, .vram-oom-settings {
      font-size: 14px;
      opacity: 0.9;
      margin-bottom: 6px;
    }
    .vram-oom-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 10px;
      margin-bottom: 10px;
    }
    @media (max-width: 900px) {
      .vram-oom-grid { grid-template-columns: 1fr; }
    }
    .vram-oom-card {
      border: 1px solid rgba(255,23,68,0.25);
      background: rgba(15, 23, 42, 0.06);
      border-radius: 12px;
      padding: 12px 12px;
    }
    .vram-oom-card-title {
      font-weight: 800;
      margin-bottom: 6px;
    }
    .vram-oom-list { margin: 0.4rem 0 0 1.2rem; }
    .vram-oom-snippet-wrap {
      margin-top: 12px;
      border-top: 1px dashed rgba(255,23,68,0.35);
      padding-top: 10px;
    }
    .vram-oom-snippet-title { font-weight: 800; margin-bottom: 6px; }
    .vram-oom-snippet {
      background: rgba(0,0,0,0.55);
      color: #e2e8f0;
      padding: 10px;
      border-radius: 10px;
      overflow-x: auto;
      white-space: pre-wrap;
      font-size: 12.5px;
      line-height: 1.35;
      border: 1px solid rgba(255,255,255,0.10);
    }
    .vram-oom-details {
      margin-top: 6px;
      border: 1px solid rgba(255,23,68,0.22);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.06);
    }
    .vram-oom-summary { font-weight: 800; cursor: pointer; }
    @keyframes vramPulse {
      0%   { box-shadow: 0 0 0 rgba(255,23,68,0.0), 0 0 0 rgba(255,193,7,0.0); }
      50%  { box-shadow: 0 0 18px rgba(255,23,68,0.48), 0 0 34px rgba(255,193,7,0.22); }
      100% { box-shadow: 0 0 0 rgba(255,23,68,0.0), 0 0 0 rgba(255,193,7,0.0); }
    }
    @keyframes vramShimmer {
      0%   { transform: translateX(-20%) rotate(10deg); }
      100% { transform: translateX(35%) rotate(10deg); }
    }

    /* Small inline processing indicator (used in input sizing panels) */
    .processing-banner {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      border: 1px solid rgba(59,130,246,0.28);
      background: linear-gradient(90deg, rgba(59,130,246,0.10), rgba(16,185,129,0.06));
      padding: 10px 12px;
      border-radius: 12px;
      margin: 6px 0;
    }
    .processing-spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(59,130,246,0.25);
      border-top-color: rgba(59,130,246,0.90);
      border-radius: 9999px;
      animation: secSpin 0.9s linear infinite;
      flex: 0 0 auto;
      margin-top: 2px;
    }
    .processing-col { display: flex; flex-direction: column; }
    .processing-text { font-weight: 850; animation: secPulse 1.2s ease-in-out infinite; }
    .processing-sub { font-size: 12px; opacity: 0.85; margin-top: 2px; line-height: 1.35; }
    @keyframes secSpin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    @keyframes secPulse {
      0% { opacity: 0.55; }
      50% { opacity: 1.0; }
      100% { opacity: 0.55; }
    }

    /* SeedVR2 sizing and chunk analysis card */
    .resolution-info .resolution-stats-shell {
      border: 1px solid rgba(99, 102, 241, 0.26);
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.32), rgba(30, 64, 175, 0.10));
      border-radius: 14px;
      padding: 12px;
      margin: 4px 0;
    }
    .resolution-info .resolution-stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .resolution-info .resolution-stats-col {
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 0;
    }
    .resolution-info .resolution-stat-card {
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(15, 23, 42, 0.36);
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .resolution-info .resolution-stat-card-title {
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      opacity: 0.9;
      margin-bottom: 8px;
      color: #c7d2fe;
    }
    .resolution-info .resolution-stat-row {
      display: grid;
      grid-template-columns: minmax(130px, 42%) minmax(0, 1fr);
      gap: 10px;
      align-items: baseline;
      border-top: 1px dashed rgba(148, 163, 184, 0.18);
      padding: 7px 0;
    }
    .resolution-info .resolution-stat-row:first-of-type {
      border-top: none;
      padding-top: 2px;
    }
    .resolution-info .resolution-stat-key {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.2px;
      opacity: 0.75;
    }
    .resolution-info .resolution-stat-val {
      font-size: 14px;
      font-weight: 650;
      word-break: break-word;
      line-height: 1.35;
    }
    .resolution-info .resolution-stat-val.is-up {
      color: #4ade80;
      font-weight: 800;
    }
    .resolution-info .resolution-stat-val.is-down {
      color: #fb7185;
      font-weight: 800;
    }
    .resolution-info .resolution-stat-val.is-neutral {
      color: #f8fafc;
    }
    .resolution-info .resolution-notes {
      margin-top: 10px;
      border-top: 1px solid rgba(148, 163, 184, 0.22);
      padding-top: 9px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .resolution-info .resolution-note-item {
      background: rgba(59, 130, 246, 0.14);
      border: 1px solid rgba(59, 130, 246, 0.28);
      border-radius: 9999px;
      padding: 4px 10px;
      font-size: 12px;
      line-height: 1.3;
      max-width: 100%;
      word-break: break-word;
    }
    @media (max-width: 1100px) {
      .resolution-info .resolution-stats-grid {
        grid-template-columns: 1fr;
      }
      .resolution-info .resolution-stat-row {
        grid-template-columns: minmax(110px, 40%) minmax(0, 1fr);
      }
    }

    /* Distinct action button styles across all processing tabs */
    .gradio-container {
      --action-text: #f8fafc;
      --action-border: rgba(255, 255, 255, 0.28);
      --action-focus: rgba(56, 189, 248, 0.55);
    }
    .action-btn button,
    button.action-btn {
      position: relative;
      overflow: hidden;
      border-radius: 14px !important;
      border: 1px solid var(--action-border) !important;
      color: var(--action-text) !important;
      font-weight: 800 !important;
      letter-spacing: 0.2px;
      transition: transform 0.16s ease, box-shadow 0.22s ease, filter 0.22s ease, border-color 0.22s ease;
    }
    .action-btn button:focus-visible,
    button.action-btn:focus-visible {
      outline: none !important;
      box-shadow: 0 0 0 3px var(--action-focus), 0 10px 24px rgba(2, 6, 23, 0.32) !important;
    }
    .action-btn button:hover,
    button.action-btn:hover {
      transform: translateY(-1px);
      filter: saturate(1.08);
    }
    .action-btn button:active,
    button.action-btn:active {
      transform: translateY(0px) scale(0.995);
    }
    .action-btn button:disabled,
    button.action-btn:disabled {
      filter: saturate(0.45) brightness(0.82);
      opacity: 0.68;
      transform: none;
      box-shadow: none !important;
    }

    /* Hero button: Upscale / Process */
    .action-btn-upscale button,
    button.action-btn-upscale {
      background: linear-gradient(135deg, #4338ca 0%, #0f766e 52%, #15803d 100%) !important;
      border-color: rgba(167, 243, 208, 0.9) !important;
      box-shadow: 0 12px 28px rgba(21, 128, 61, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
    }
    .action-btn-upscale button::after,
    button.action-btn-upscale::after {
      content: "";
      position: absolute;
      inset: -40% auto -40% -28%;
      width: 30%;
      transform: rotate(18deg);
      background: linear-gradient(90deg, rgba(255, 255, 255, 0.0), rgba(255, 255, 255, 0.35), rgba(255, 255, 255, 0.0));
      animation: actionSweep 2.25s linear infinite;
      pointer-events: none;
    }
    .action-btn-upscale button:hover,
    button.action-btn-upscale:hover {
      box-shadow: 0 16px 34px rgba(21, 128, 61, 0.46), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
      border-color: rgba(187, 247, 208, 1) !important;
    }

    /* Preview */
    .action-btn-preview button,
    button.action-btn-preview {
      background: linear-gradient(135deg, #1e40af 0%, #0369a1 56%, #0f766e 100%) !important;
      box-shadow: 0 10px 24px rgba(14, 116, 144, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    .action-btn-preview button:hover,
    button.action-btn-preview:hover {
      box-shadow: 0 12px 28px rgba(14, 116, 144, 0.44), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
    }

    /* Cancel */
    .action-btn-cancel button,
    button.action-btn-cancel {
      background: linear-gradient(135deg, #b91c1c 0%, #dc2626 60%, #fb7185 100%) !important;
      border-color: rgba(254, 202, 202, 0.9) !important;
      box-shadow: 0 10px 24px rgba(220, 38, 38, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    }
    .action-btn-cancel button:hover,
    button.action-btn-cancel:hover {
      box-shadow: 0 12px 30px rgba(220, 38, 38, 0.46), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
    }

    /* Open Outputs */
    .action-btn-open button,
    button.action-btn-open {
      background: linear-gradient(135deg, #047857 0%, #0f766e 58%, #14b8a6 100%) !important;
      box-shadow: 0 10px 24px rgba(15, 118, 110, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    }
    .action-btn-open button:hover,
    button.action-btn-open:hover {
      box-shadow: 0 12px 28px rgba(15, 118, 110, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
    }

    /* Clear/Delete Temp */
    .action-btn-clear button,
    button.action-btn-clear {
      background: linear-gradient(135deg, #9a3412 0%, #c2410c 55%, #f97316 100%) !important;
      border-color: rgba(254, 215, 170, 0.95) !important;
      box-shadow: 0 10px 24px rgba(194, 65, 12, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    }
    .action-btn-clear button:hover,
    button.action-btn-clear:hover {
      box-shadow: 0 12px 28px rgba(194, 65, 12, 0.44), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
    }

    @keyframes actionSweep {
      0% {
        left: -32%;
      }
      100% {
        left: 115%;
      }
    }

    @media (max-width: 768px) {
      .action-btn button,
      button.action-btn {
        border-radius: 12px !important;
      }
      .action-btn-upscale button,
      button.action-btn-upscale {
        box-shadow: 0 10px 24px rgba(21, 128, 61, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
      }
    }
    """
    CUSTOM_HEAD = """
    <script>
    (() => {
      if (window.__secoursesImageSliderFsShimInstalled) return;
      window.__secoursesImageSliderFsShimInstalled = true;

      const getFsElement = () =>
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.msFullscreenElement ||
        null;

      const requestFs = (el) => {
        if (!el) return;
        const fn = el.requestFullscreen || el.webkitRequestFullscreen || el.msRequestFullscreen;
        if (typeof fn === "function") fn.call(el);
      };

      const exitFs = () => {
        const fn = document.exitFullscreen || document.webkitExitFullscreen || document.msExitFullscreen;
        if (typeof fn === "function") fn.call(document);
      };

      document.addEventListener(
        "click",
        (evt) => {
          const btn = evt.target && evt.target.closest ? evt.target.closest("button") : null;
          if (!btn) return;

          const wrapper = btn.closest(".native-image-comparison-slider");
          if (!wrapper) return;

          const label = (
            btn.getAttribute("aria-label") ||
            btn.getAttribute("title") ||
            btn.innerText ||
            ""
          ).toLowerCase();

          if (!label.includes("fullscreen")) return;

          evt.preventDefault();
          evt.stopImmediatePropagation();

          if (getFsElement()) {
            exitFs();
            return;
          }

          const target =
            wrapper.querySelector(".image-container") ||
            wrapper.querySelector(".slider-wrap") ||
            wrapper;

          requestFs(target);
        },
        true
      );
    })();
    </script>
    """

    # =========================================================================
    # UNIVERSAL PRESET SYSTEM - Load last used preset on startup
    # =========================================================================
    def load_startup_universal_preset():
        """
        Load the last-used UNIVERSAL preset on startup.
        
        Universal presets contain ALL settings from ALL tabs in a single file.
        If no universal preset exists, falls back to defaults.
        
        Returns:
            Dict with structure matching update_shared_state_from_preset expectations
        """
        from shared.universal_preset import (
            get_all_defaults,
            merge_preset_with_defaults,
            TAB_CONFIGS,
        )
        from shared.models import get_seedvr2_model_names, scan_gan_models, get_flashvsr_model_names, get_rife_model_names
        
        # Get models list for defaults
        seedvr2_models = get_seedvr2_model_names()
        gan_models = scan_gan_models(BASE_DIR)
        flashvsr_models = get_flashvsr_model_names()
        rife_models = get_rife_model_names(BASE_DIR)
        
        all_models = sorted(list({
            *seedvr2_models,
            *gan_models,
            *flashvsr_models,
            *rife_models,
        }))
        if not all_models:
            all_models = ["default"]
        
        # Try to load last used universal preset
        last_preset_name = preset_manager.get_last_used_universal_preset()
        loaded_preset = None
        
        if last_preset_name:
            loaded_preset = preset_manager.load_universal_preset(last_preset_name)
            if loaded_preset:
                print(f"✅ Loaded universal preset '{last_preset_name}' on startup")
            else:
                print(f"⚠️ Last used preset '{last_preset_name}' not found, using defaults")
        
        if loaded_preset:
            # Merge with defaults to fill any missing keys
            merged_preset = merge_preset_with_defaults(loaded_preset, BASE_DIR, all_models)
            return merged_preset, last_preset_name, all_models
        else:
            # Use defaults
            defaults = get_all_defaults(BASE_DIR, all_models)
            return defaults, None, all_models
    
    # Load universal preset on startup
    startup_preset, startup_preset_name, all_models = load_startup_universal_preset()
    sync_defaults = get_all_defaults(BASE_DIR, all_models)
    
    with gr.Blocks(title=APP_TITLE, theme=modern_theme, css=CUSTOM_CSS, head=CUSTOM_HEAD) as demo:
        # =========================================================================
        # SHARED STATE - Populated from UNIVERSAL PRESET on startup
        # =========================================================================
        # Extract tab settings from universal preset
        startup_res_settings = startup_preset.get("resolution", {})
        startup_output_settings = startup_preset.get("output", {})
        startup_output_settings = dict(startup_output_settings) if isinstance(startup_output_settings, dict) else {}
        startup_output_settings["global_rife_model"] = (
            str(startup_output_settings.get("global_rife_model", "") or "").strip() or get_rife_default_model()
        )
        startup_output_settings["global_rife_process_chunks"] = bool(
            startup_output_settings.get("global_rife_process_chunks", True)
        )
        startup_preset["output"] = startup_output_settings
        startup_auto_chunk = bool((startup_res_settings or {}).get("auto_chunk", True))
        startup_res_settings = dict(startup_res_settings) if isinstance(startup_res_settings, dict) else {}
        startup_res_settings.setdefault("auto_chunk", startup_auto_chunk)
        startup_res_settings.setdefault("auto_detect_scenes", True)
        startup_res_settings.setdefault("frame_accurate_split", True)
        if startup_auto_chunk:
            startup_res_settings["chunk_overlap"] = 0.0
        startup_preset["resolution"] = startup_res_settings
        startup_chunk_overlap_sec = 0.0 if startup_auto_chunk else float(startup_res_settings.get("chunk_overlap", 0.0) or 0.0)
        
        shared_state = gr.State({
            "health_banner": {"text": health_text},
            "alerts": {"oom": {"visible": False, "html": "", "ts": None}},
            "seed_controls": {
                # UNIVERSAL PRESET: Current preset name
                "current_preset_name": startup_preset_name,
                "preset_dirty": False,
                
                # UNIVERSAL PRESET: Full tab settings (used by all tabs)
                "seedvr2_settings": startup_preset.get("seedvr2", {}),
                "gan_settings": startup_preset.get("gan", {}),
                "rife_settings": startup_preset.get("rife", {}),
                "flashvsr_settings": startup_preset.get("flashvsr", {}),
                "face_settings": startup_preset.get("face", {}),
                "resolution_settings": startup_preset.get("resolution", {}),
                "output_settings": startup_preset.get("output", {}),
                
                # Individual cached values (for backward compatibility with other code)
                "upscale_factor_val": startup_res_settings.get("upscale_factor", 4.0),
                "max_resolution_val": startup_res_settings.get("max_target_resolution", 0),
                "current_model": None,
                "last_input_path": "",
                "last_output_dir": "",
                "last_output_path": None,
                
                # Output tab cached values
                "png_padding_val": startup_output_settings.get("png_padding", 6),
                "png_keep_basename_val": startup_output_settings.get("png_keep_basename", True),
                "overwrite_existing_batch_val": bool(startup_output_settings.get("overwrite_existing_batch", False)),
                "skip_first_frames_val": startup_output_settings.get("skip_first_frames", 0),
                "load_cap_val": startup_output_settings.get("load_cap", 0),
                "fps_override_val": startup_output_settings.get("fps_override", 0),
                "frame_interpolation_val": bool(startup_output_settings.get("frame_interpolation", False)),
                "global_rife_enabled_val": bool(startup_output_settings.get("frame_interpolation", False)),
                "global_rife_multiplier_val": startup_output_settings.get("global_rife_multiplier", "x2"),
                "global_rife_model_val": startup_output_settings.get("global_rife_model", get_rife_default_model()),
                "global_rife_precision_val": startup_output_settings.get("global_rife_precision", "fp32"),
                "global_rife_cuda_device_val": startup_output_settings.get("global_rife_cuda_device", ""),
                "global_rife_process_chunks_val": bool(startup_output_settings.get("global_rife_process_chunks", True)),
                "output_format_val": startup_output_settings.get("output_format", "auto"),
                "png_sequence_enabled_val": bool(startup_output_settings.get("png_sequence_enabled", False)),
                "comparison_mode_val": startup_output_settings.get("comparison_mode", "slider"),
                "pin_reference_val": startup_output_settings.get("pin_reference", False),
                "fullscreen_val": startup_output_settings.get("fullscreen_enabled", True),
                "save_metadata_val": startup_output_settings.get("save_metadata", True),
                "telemetry_enabled_val": startup_output_settings.get("telemetry_enabled", True),
                "audio_codec_val": startup_output_settings.get("audio_codec", "copy"),
                "audio_bitrate_val": startup_output_settings.get("audio_bitrate", ""),
                "generate_comparison_video_val": startup_output_settings.get("generate_comparison_video", True),
                "comparison_video_layout_val": startup_output_settings.get("comparison_video_layout", "auto"),
                "face_strength_val": global_settings.get("face_strength", 0.5),
                "queue_enabled_val": bool(global_settings.get("queue_enabled", True)),
                
                # Resolution tab cached values
                "auto_chunk": startup_auto_chunk,
                "auto_detect_scenes": bool(startup_res_settings.get("auto_detect_scenes", True)),
                "frame_accurate_split": bool(startup_res_settings.get("frame_accurate_split", True)),
                "chunk_size_sec": startup_res_settings.get("chunk_size", 0),
                "chunk_overlap_sec": startup_chunk_overlap_sec,
                "ratio_downscale": startup_res_settings.get("ratio_downscale_then_upscale", True),
                "enable_max_target": startup_res_settings.get("enable_max_target", True),
                "auto_resolution": startup_res_settings.get("auto_resolution", True),
                "per_chunk_cleanup": startup_res_settings.get("per_chunk_cleanup", False),
                "scene_threshold": startup_res_settings.get("scene_threshold", 27.0),
                "min_scene_len": startup_res_settings.get("min_scene_len", 1.0),
                
                # Pinned reference (persisted globally)
                "pinned_reference_path": global_settings.get("pinned_reference_path"),
                
                # Available models list (for preset defaults)
                "available_models": all_models,
            },
            "operation_status": "ready"
        })

        # Health banner at the top
        health_banner = gr.Markdown(f'<div class="health-banner">{health_text}</div>')

        # VRAM OOM banner (shown only on VRAM OOM)
        # NOTE: We update this via a Timer tick because gr.State change events can be
        # inconsistent across Gradio versions/environments.
        oom_banner = gr.HTML(value="", visible=False)
        oom_dismiss_btn = gr.Button("Dismiss VRAM Alert", variant="secondary", size="sm", visible=False)
        oom_timer = gr.Timer(value=2.0, active=True)
        health_sync_signature = gr.State(value="")
        oom_sync_signature = gr.State(value="")
        gr.Markdown(f"# {APP_TITLE}")

        # Global settings tab (rendered LAST for a cleaner workflow)
        def render_global_settings_tab():
            with gr.Tab("Global Settings", render_children=True):
                gr.Markdown("### 📁 Output & Temp Directories")
                with gr.Row():
                    output_dir_box = gr.Textbox(
                        label="Default Outputs Folder",
                        value=global_settings["output_dir"],
                        info="Where processed files will be saved"
                    )
                    temp_dir_box = gr.Textbox(
                        label="Temp Folder",
                        value=global_settings["temp_dir"],
                        info="Temporary files during processing"
                    )
                
                gr.Markdown("### 🎭 Face Restoration")
                with gr.Row():
                    telemetry_toggle = gr.Checkbox(
                        label="Save run metadata (local telemetry)",
                        value=global_settings.get("telemetry", True),
                        info="Save processing metadata for troubleshooting"
                    )
                    face_global_toggle = gr.Checkbox(
                        label="Apply Face Restoration globally",
                        value=global_settings.get("face_global", False),
                        info="Enable face restoration for all upscaling operations (SeedVR2, GAN, RIFE, FlashVSR+)"
                    )
                    queue_enabled_toggle = gr.Checkbox(
                        label="Enable Queue",
                        value=bool(global_settings.get("queue_enabled", True)),
                        info="When enabled, extra generate/upscale clicks are queued. When disabled, new clicks are ignored while processing is active."
                    )
                with gr.Row():
                    face_strength_slider = gr.Slider(
                        label="Global Face Restoration Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=global_settings.get("face_strength", 0.5),
                        info="Strength of face restoration when globally enabled (0.0 = subtle, 1.0 = maximum)"
                    )
                
                # FIXED: Make model cache paths EDITABLE and persistable (not read-only)
                gr.Markdown("### 📦 Model Cache Paths")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        **Configure where AI models are downloaded and cached.**
                        
                        These paths control where models (SeedVR2, FlashVSR+, Real-ESRGAN, etc.) are stored.
                        If left empty, defaults from launcher BAT file or system defaults will be used.
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        ⚠️ **IMPORTANT**: Changing these paths will NOT move existing models. You must:
                        1. Update paths here (saved immediately)
                        2. Restart the application
                        3. Models will re-download to new location (or manually copy from old location)
                        """)
                
                with gr.Row():
                    models_dir_box = gr.Textbox(
                        label="Models Directory (MODELS_DIR)",
                        value=global_settings.get("models_dir", ""),
                        placeholder=str(BASE_DIR / "models"),
                        info="Base directory for all model weights. Leave empty to use launcher default or './models'."
                    )
                
                with gr.Row():
                    hf_home_box = gr.Textbox(
                        label="HuggingFace Home (HF_HOME)",
                        value=global_settings.get("hf_home", ""),
                        placeholder=str(BASE_DIR / "models"),
                        info="HuggingFace cache directory. Usually same as Models Directory. Leave empty to use MODELS_DIR."
                    )
                    transformers_cache_box = gr.Textbox(
                        label="Transformers Cache (TRANSFORMERS_CACHE)",
                        value=global_settings.get("transformers_cache", ""),
                        placeholder=str(BASE_DIR / "models"),
                        info="Transformers library cache. Usually same as HF_HOME. Leave empty to use MODELS_DIR."
                    )
                
                gr.Markdown("""
                💡 **Current launcher settings** (from `Windows_Run_SECourses_Upscaler_Pro.bat`):
                - Models Dir: `{}`
                - HF Home: `{}`
                - Transformers: `{}`
                
                You can override these in the UI above, or edit the launcher BAT file for permanent changes.
                """.format(
                    launcher_models_dir or "Not set (using defaults)",
                    launcher_hf_home or "Not set (using MODELS_DIR)",
                    launcher_transformers_cache or "Not set (using MODELS_DIR)"
                ))
                
                # Processing mode selection controls (placed before Execution Mode explanation)
                mode_radio = gr.Radio(
                    choices=["subprocess", "in_app"],
                    value=saved_mode,
                    label="Processing Mode",
                    info="Applies immediately.",
                    interactive=True
                )
                global_status = gr.Markdown("")
                gr.Markdown(
                    "**Restart-required settings:** `MODELS_DIR`, `HF_HOME`, `TRANSFORMERS_CACHE`.\n"
                    "They are saved immediately, but full effect across all libraries is guaranteed after app restart."
                )

                gr.Markdown("### Global Config")
                global_config_choices = preset_manager.list_presets("global_config", "app")
                last_global_config = preset_manager.get_last_used_name("global_config", "app")
                global_config_value = (
                    last_global_config
                    if last_global_config in global_config_choices
                    else (global_config_choices[-1] if global_config_choices else None)
                )
                with gr.Row():
                    global_config_dropdown = gr.Dropdown(
                        label="Saved Global Configs",
                        choices=global_config_choices,
                        value=global_config_value,
                        allow_custom_value=False,
                    )
                    global_config_name = gr.Textbox(
                        label="Save As",
                        placeholder="e.g. workstation_a",
                    )
                with gr.Row():
                    global_config_save_btn = gr.Button("Save Config", variant="secondary")
                    global_config_load_btn = gr.Button("Load Config", variant="secondary")
                    global_config_delete_btn = gr.Button("Delete Config", variant="stop")
                global_config_status = gr.Markdown("")
                
                # Execution mode controls
                gr.Markdown("### ⚙️ Execution Mode")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ## 🟢 Subprocess Mode
                        **(Default & STRONGLY RECOMMENDED)**
                        
                        **What it does:**
                        - Completely isolated subprocess per run
                        - Models load fresh, process, exit with guaranteed cleanup
                        - Works perfectly for ALL models
                        
                        **Benefits:**
                        - ✅ **100% VRAM/RAM cleanup** after each run
                        - ✅ **Full cancellation support** - kill anytime
                        - ✅ **Automatic vcvars wrapper** (Windows torch.compile)
                        - ✅ **Process isolation** - prevents memory leaks
                        - ✅ **Proven stability** - production-ready
                        - ✅ **Cross-platform** - Windows and Linux
                        
                        **Performance:**
                        - ~5-10s loading overhead per run
                        - Negligible for long videos (<1% total time)
                        - Amortized across batch processing
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ## 🔴 In-App Mode
                        **(EXPERIMENTAL - DO NOT USE)**
                        
                        **⚠️ CRITICAL: NON-FUNCTIONAL PLACEHOLDER**
                        
                        **Status**: Partially implemented with **ZERO BENEFITS** and **CRITICAL LIMITATIONS**
                        
                        **Why It Doesn't Work:**
                        
                        **1. ❌ NO MODEL PERSISTENCE**
                        - Models reload EVERY RUN (identical to subprocess)
                        - Expected: Models stay in VRAM for speed
                        - Result: **ZERO SPEED BENEFIT**
                        
                        **2. ❌ NO CANCELLATION**
                        - Cannot stop once started
                        - Must wait or force-quit entire app
                        
                        **3. ❌ NO VCVARS WRAPPER**
                        - torch.compile fails on Windows
                        - Must activate vcvars before launch
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ## 🔴 More Issues
                        
                        **4. ⚠️ MEMORY LEAKS**
                        - VRAM may not fully clear
                        - Usage creeps up, eventual OOM crashes
                        
                        **5. ℹ️ MODE SWITCH**
                        - Mode changes apply immediately from the selector above
                        - No restart needed for mode-only changes
                        
                        **Required for Functional Mode:**
                        - Refactor CLIs for persistent loading
                        - Implement VRAM caching
                        - Add threading cancellation
                        - Pre-activate vcvars on startup
                        - CUDA memory profiling
                        
                        **Effort**: 40-60 hours + testing  
                        **Risk**: High (CUDA complexity)  
                        **Benefit**: Marginal (5-10% for short videos)
                        
                        ---
                        
                        **Recommendation:**
                        
                        ✅ **USE SUBPROCESS MODE**
                        - Production-ready and battle-tested
                        - Reliable cleanup and cancellation
                        
                        🚫 **AVOID IN-APP MODE**
                        - Non-functional (models reload anyway)
                        - Dangerous (memory leaks, failures)
                        """)
                
                gr.Markdown("💡 **Note**: In-app mode exists ONLY as a code framework for potential future optimization. Consider it **disabled** for all practical purposes.")

                # Wire up global settings events (auto-apply)
                def apply_global_settings_live(od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache, state):
                    from shared.services.global_service import apply_global_settings_live
                    return apply_global_settings_live(
                        od, td, tel, face, face_str, queue_enabled, mode_choice,
                        models_dir, hf_home, trans_cache,
                        runner, preset_manager, global_settings, run_logger, state,
                    )

                def apply_global_settings_live_no_mode(od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache, state):
                    status_upd, _mode_upd, next_state = apply_global_settings_live(
                        od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache, state
                    )
                    return status_upd, next_state

                global_live_inputs = [
                    output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, face_strength_slider,
                    queue_enabled_toggle, mode_radio, models_dir_box, hf_home_box, transformers_cache_box, shared_state,
                ]

                for comp in [
                    output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, face_strength_slider,
                    queue_enabled_toggle, models_dir_box, hf_home_box, transformers_cache_box,
                ]:
                    comp.change(
                        fn=apply_global_settings_live_no_mode,
                        inputs=global_live_inputs,
                        outputs=[global_status, shared_state],
                        queue=False,
                        show_progress="hidden",
                        trigger_mode="always_last",
                    )

                mode_radio.change(
                    fn=apply_global_settings_live,
                    inputs=global_live_inputs,
                    outputs=[global_status, mode_radio, shared_state],
                    queue=False,
                    show_progress="hidden",
                    trigger_mode="always_last",
                )

                def build_global_config_payload(od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache):
                    return {
                        "output_dir": od,
                        "temp_dir": td,
                        "telemetry": bool(tel),
                        "face_global": bool(face),
                        "face_strength": float(face_str),
                        "queue_enabled": bool(queue_enabled),
                        "mode": str(mode_choice or "subprocess"),
                        "models_dir": models_dir,
                        "hf_home": hf_home,
                        "transformers_cache": trans_cache,
                    }

                def save_global_config(cfg_name, od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache, state):
                    name = str(cfg_name or "").strip()
                    if not name:
                        return gr.update(), gr.update(value="Enter a config name first."), gr.update(), gr.update(), state

                    payload = build_global_config_payload(
                        od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache
                    )
                    preset_manager.save_preset_safe("global_config", "app", name, payload)
                    dropdown_upd = gr.update(
                        choices=preset_manager.list_presets("global_config", "app"),
                        value=name,
                    )
                    status_upd, mode_upd, next_state = apply_global_settings_live(
                        od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache, state
                    )
                    return (
                        dropdown_upd,
                        gr.update(value=f"Saved global config '{name}'."),
                        status_upd,
                        mode_upd,
                        next_state,
                    )

                def load_global_config(cfg_name, od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache, state):
                    name = str(cfg_name or "").strip()
                    if not name:
                        return (
                            od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache,
                            gr.update(value="Select a config first."), gr.update(), state,
                        )

                    loaded = preset_manager.load_preset_safe("global_config", "app", name)
                    if not loaded:
                        return (
                            od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache,
                            gr.update(value=f"Config '{name}' not found."), gr.update(), state,
                        )
                    preset_manager.set_last_used("global_config", "app", name)

                    merged = build_global_config_payload(
                        od, td, tel, face, face_str, queue_enabled, mode_choice, models_dir, hf_home, trans_cache
                    )
                    merged.update(loaded)

                    od2 = merged.get("output_dir", od)
                    td2 = merged.get("temp_dir", td)
                    tel2 = bool(merged.get("telemetry", tel))
                    face2 = bool(merged.get("face_global", face))
                    face_str2 = float(merged.get("face_strength", face_str))
                    queue_enabled2 = bool(merged.get("queue_enabled", queue_enabled))
                    mode2 = str(merged.get("mode", mode_choice) or "subprocess")
                    models2 = merged.get("models_dir", models_dir)
                    hf2 = merged.get("hf_home", hf_home)
                    trans2 = merged.get("transformers_cache", trans_cache)

                    status_upd, mode_upd, next_state = apply_global_settings_live(
                        od2, td2, tel2, face2, face_str2, queue_enabled2, mode2, models2, hf2, trans2, state
                    )
                    mode_val = mode2
                    if isinstance(mode_upd, dict) and ("value" in mode_upd):
                        mode_val = mode_upd.get("value")

                    return (
                        od2, td2, tel2, face2, face_str2, queue_enabled2, mode_val, models2, hf2, trans2,
                        gr.update(value=f"Loaded global config '{name}'"), status_upd, next_state,
                    )

                def delete_global_config(cfg_name):
                    name = str(cfg_name or "").strip()
                    if not name:
                        return gr.update(), gr.update(value="Select a config first.")
                    ok = preset_manager.delete_preset("global_config", "app", name)
                    choices = preset_manager.list_presets("global_config", "app")
                    value = choices[-1] if choices else None
                    msg = f"Deleted global config '{name}'." if ok else f"Config '{name}' not found."
                    return gr.update(choices=choices, value=value), gr.update(value=msg)

                global_config_save_btn.click(
                    fn=save_global_config,
                    inputs=[global_config_name] + global_live_inputs,
                    outputs=[global_config_dropdown, global_config_status, global_status, mode_radio, shared_state],
                    queue=False,
                    show_progress="hidden",
                )

                global_config_load_btn.click(
                    fn=load_global_config,
                    inputs=[global_config_dropdown] + global_live_inputs,
                    outputs=[
                        output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, face_strength_slider,
                        queue_enabled_toggle, mode_radio, models_dir_box, hf_home_box, transformers_cache_box,
                        global_config_status, global_status, shared_state,
                    ],
                    queue=False,
                    show_progress="hidden",
                )

                global_config_delete_btn.click(
                    fn=delete_global_config,
                    inputs=[global_config_dropdown],
                    outputs=[global_config_dropdown, global_config_status],
                    queue=False,
                    show_progress="hidden",
                )

        # ------------------------------------------------------------------ #
        # Universal preset sync:
        # - The load button updates ALL tabs in shared_state.
        # - Each tab refreshes its UI values when the user selects the tab.
        # ------------------------------------------------------------------ #
        def _sync_signature(payload: Dict[str, Any]) -> str:
            try:
                blob = json.dumps(
                    payload,
                    sort_keys=True,
                    ensure_ascii=True,
                    default=str,
                    separators=(",", ":"),
                )
            except Exception:
                blob = str(payload)
            return hashlib.sha1(blob.encode("utf-8")).hexdigest()

        def _make_tab_sync(tab_name: str):
            tab_defaults = sync_defaults.get(tab_name, {})

            def _sync(state: Dict[str, Any], previous_signature: str = ""):
                seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
                tab_settings = seed_controls.get(f"{tab_name}_settings", {}) if isinstance(seed_controls, dict) else {}
                tab_settings = tab_settings if isinstance(tab_settings, dict) else {}

                # Enforce guardrails that are shared-state level invariants.
                if tab_name == "resolution":
                    if bool(tab_settings.get("auto_chunk", True)):
                        tab_settings = dict(tab_settings)
                        tab_settings["chunk_overlap"] = 0.0

                current = seed_controls.get("current_preset_name") if isinstance(seed_controls, dict) else None
                presets = preset_manager.list_universal_presets()
                signature = _sync_signature(
                    {
                        "tab": tab_name,
                        "current_preset_name": current,
                        "tab_settings": tab_settings,
                        "presets": presets,
                    }
                )
                if signature == str(previous_signature or ""):
                    return gr.skip()

                values = dict_to_values(tab_name, tab_settings, tab_defaults)
                # Dropdown-safe fallbacks for historical presets/state corruption.
                if tab_name == "rife" and len(values) > 5 and not str(values[5] or "").strip():
                    values[5] = get_rife_default_model()
                if tab_name == "output" and len(values) > 17 and not str(values[17] or "").strip():
                    values[17] = get_rife_default_model()

                selected = current if current in presets else (presets[-1] if presets else None)
                dropdown_upd = gr.update(choices=presets, value=selected)
                status_text = f"✅ Synced from universal preset '{selected}'" if selected else "ℹ️ Synced from shared state"
                return (*values, dropdown_upd, gr.update(value=status_text), signature)

            return _sync

        tab_sync_seedvr2 = gr.State(value="")
        tab_sync_resolution = gr.State(value="")
        tab_sync_output = gr.State(value="")
        tab_sync_face = gr.State(value="")
        tab_sync_rife = gr.State(value="")
        tab_sync_gan = gr.State(value="")
        tab_sync_flashvsr = gr.State(value="")
        seedvr2_auto_res_sync = gr.State(value="")

        # Self-contained tabs following SECourses pattern
        with gr.Tab("🎬 SeedVR2 (Video/Image)", render_children=True) as tab_seedvr2:
            seedvr2_ui = seedvr2_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )
        seedvr2_tab_select_evt = tab_seedvr2.select(
            fn=_make_tab_sync("seedvr2"),
            inputs=[shared_state, tab_sync_seedvr2],
            outputs=seedvr2_ui["inputs_list"] + [seedvr2_ui["preset_dropdown"], seedvr2_ui["preset_status"], tab_sync_seedvr2],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )
        # Ensure sizing panel (including Output FPS + Global RIFE forecast) is refreshed
        # when navigating back to SeedVR2 after changing Output-tab settings.
        def _refresh_seedvr2_auto_res_if_needed(state: Dict[str, Any], previous_signature: str = ""):
            seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
            refresh_signature = _sync_signature(
                {
                    "last_input_path": seed_controls.get("last_input_path"),
                    "upscale_factor_val": seed_controls.get("upscale_factor_val"),
                    "max_resolution_val": seed_controls.get("max_resolution_val"),
                    "resolution_settings": seed_controls.get("resolution_settings", {}),
                    "output_settings": seed_controls.get("output_settings", {}),
                }
            )
            if refresh_signature == str(previous_signature or ""):
                return gr.skip(), previous_signature
            return seedvr2_ui["refresh_auto_res"](state), refresh_signature

        seedvr2_tab_select_evt.then(
            fn=_refresh_seedvr2_auto_res_if_needed,
            inputs=[shared_state, seedvr2_auto_res_sync],
            outputs=[seedvr2_ui["auto_res_msg"], seedvr2_auto_res_sync],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("📐 Resolution & Scene Split", render_children=True) as tab_resolution:
            resolution_ui = resolution_tab(
                preset_manager=preset_manager,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )
        tab_resolution.select(
            fn=_make_tab_sync("resolution"),
            inputs=[shared_state, tab_sync_resolution],
            outputs=resolution_ui["inputs_list"] + [resolution_ui["preset_dropdown"], resolution_ui["preset_status"], tab_sync_resolution],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("🎭 Output & Comparison", render_children=True) as tab_output:
            output_ui = output_tab(
                preset_manager=preset_manager,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                global_settings=global_settings
            )
        tab_output.select(
            fn=_make_tab_sync("output"),
            inputs=[shared_state, tab_sync_output],
            outputs=output_ui["inputs_list"] + [output_ui["preset_dropdown"], output_ui["preset_status"], tab_sync_output],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("👤 Face Restoration", render_children=True) as tab_face:
            face_ui = face_tab(
                preset_manager=preset_manager,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )
        tab_face.select(
            fn=_make_tab_sync("face"),
            inputs=[shared_state, tab_sync_face],
            outputs=face_ui["inputs_list"] + [face_ui["preset_dropdown"], face_ui["preset_status"], tab_sync_face],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("⏱️ RIFE / FPS / Edit Videos", render_children=True) as tab_rife:
            rife_ui = rife_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )
        tab_rife.select(
            fn=_make_tab_sync("rife"),
            inputs=[shared_state, tab_sync_rife],
            outputs=rife_ui["inputs_list"] + [rife_ui["preset_dropdown"], rife_ui["preset_status"], tab_sync_rife],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("🖼️ Image-Based (GAN)", render_children=True) as tab_gan:
            gan_ui = gan_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )
        tab_gan.select(
            fn=_make_tab_sync("gan"),
            inputs=[shared_state, tab_sync_gan],
            outputs=gan_ui["inputs_list"] + [gan_ui["preset_dropdown"], gan_ui["preset_status"], tab_sync_gan],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("⚡ FlashVSR+ (Real-Time Diffusion)", render_children=True) as tab_flashvsr:
            flashvsr_ui = flashvsr_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )
        tab_flashvsr.select(
            fn=_make_tab_sync("flashvsr"),
            inputs=[shared_state, tab_sync_flashvsr],
            outputs=flashvsr_ui["inputs_list"] + [flashvsr_ui["preset_dropdown"], flashvsr_ui["preset_status"], tab_sync_flashvsr],
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )

        with gr.Tab("Queue (0)", render_children=True) as tab_queue:
            queue_tab(tab_queue)

        with gr.Tab("🏥 Health Check", render_children=True):
            health_tab(
                global_settings=global_settings,
                shared_state=shared_state,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        # Global Settings should be the last tab (far-right)
        render_global_settings_tab()

        # Update health banner on load and changes
        def update_health_banner(state, previous_signature: str = ""):
            """Update health banner with current state"""
            health_text = state.get("health_banner", {}).get("text", "System ready")
            signature = _sync_signature({"health_text": health_text})
            if signature == str(previous_signature or ""):
                return gr.skip()
            return gr.update(value=f'<div class="health-banner">{health_text}</div>'), signature

        def update_oom_banner(state, previous_signature: str = ""):
            """Update global VRAM OOM banner."""
            info = (state or {}).get("alerts", {}).get("oom", {}) if isinstance(state, dict) else {}
            html = info.get("html", "") if isinstance(info, dict) else ""
            visible = bool(isinstance(info, dict) and info.get("visible") and html)
            signature = _sync_signature({"visible": visible, "html": html if visible else ""})
            if signature == str(previous_signature or ""):
                return gr.skip()
            return gr.update(value=html or "", visible=visible), gr.update(visible=visible), signature

        def dismiss_oom(state):
            """Clear VRAM OOM banner (user dismiss)."""
            try:
                from shared.oom_alert import clear_vram_oom_alert
                clear_vram_oom_alert(state)
            except Exception:
                pass
            return state

        # Update on load
        demo.load(
            fn=update_health_banner,
            inputs=[shared_state, health_sync_signature],
            outputs=[health_banner, health_sync_signature],
        )
        demo.load(
            fn=update_oom_banner,
            inputs=[shared_state, oom_sync_signature],
            outputs=[oom_banner, oom_dismiss_btn, oom_sync_signature],
        )
        
        # Update when shared state changes (for dynamic updates from tabs)
        shared_state.change(
            fn=update_health_banner,
            inputs=[shared_state, health_sync_signature],
            outputs=[health_banner, health_sync_signature],
        )
        shared_state.change(
            fn=update_oom_banner,
            inputs=[shared_state, oom_sync_signature],
            outputs=[oom_banner, oom_dismiss_btn, oom_sync_signature],
        )

        # Polling fallback (most reliable): refresh OOM banner visibility periodically
        oom_timer.tick(
            fn=update_oom_banner,
            inputs=[shared_state, oom_sync_signature],
            outputs=[oom_banner, oom_dismiss_btn, oom_sync_signature],
        )

        # Allow user to dismiss the banner without restarting
        oom_dismiss_btn.click(fn=dismiss_oom, inputs=shared_state, outputs=shared_state)

    # Enable Gradio queue so built-in toast notifications (gr.Info/gr.Warning/gr.Error) can work
    # and to improve streaming/progress consistency.
    demo.queue()
    launch_allowed_paths = _build_launch_allowed_paths(output_dir=output_dir, temp_dir=temp_dir)
    launch_kwargs = {
        "inbrowser": True,
        "allowed_paths": launch_allowed_paths,
        "share": share_enabled,
    }
    if launch_cli_args.server_name:
        launch_kwargs["server_name"] = launch_cli_args.server_name
    if launch_cli_args.server_port is not None:
        launch_kwargs["server_port"] = launch_cli_args.server_port
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
