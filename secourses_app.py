import os
import sys
import hashlib
import json
import string
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict

# Hugging Face download transport:
# - hf_transfer can improve download speed but can also cause issues on some Windows setups.
# - Default to disabled unless the launcher/user explicitly enables it.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Upstream dependency warning (not emitted by this repo's direct code):
# suppress only this known PyTorch deprecation message to keep startup logs clean.
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.",
    category=UserWarning,
)

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
from shared.universal_preset import dict_to_values, get_all_defaults, GLOBAL_ORDER
from shared.models.rife_meta import get_rife_default_model
from shared.services.output_service import OUTPUT_ORDER
from shared.runner import Runner
from shared.gradio_compat import check_gradio_version, check_required_features
from shared.gpu_utils import (
    build_global_gpu_dropdown_choices,
    describe_gpu_selection,
    get_gpu_info,
    resolve_global_gpu_device,
)
from shared.video_codec_options import (
    get_pixel_format_choices,
    DEFAULT_AV1_FILM_GRAIN,
    DEFAULT_AV1_FILM_GRAIN_DENOISE,
)
from ui.seedvr2_tab import seedvr2_tab
from ui.resolution_tab import resolution_tab
from ui.output_tab import output_tab
from ui.face_tab import face_tab
from ui.rife_tab import rife_tab
from ui.gan_tab import gan_tab
from ui.flashvsr_tab import flashvsr_tab
from ui.health_tab import health_tab
from ui.queue_tab import queue_tab
from ui.universal_preset_section import universal_preset_section, wire_universal_preset_events

BASE_DIR = Path(__file__).parent.resolve()
PRESET_DIR = BASE_DIR / "presets"
APP_TITLE = "SECourses Ultimate Video and Image Upscaler Pro V2.9 – https://www.patreon.com/posts/150202809"


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

default_global_gpu_device = resolve_global_gpu_device(None)

GLOBAL_DEFAULTS = {
    "output_dir": launcher_output or str(BASE_DIR / "outputs"),
    "temp_dir": default_temp,
    "theme_mode": "dark",
    "telemetry": True,
    "face_global": False,
    "face_strength": 0.5,
    "queue_enabled": True,
    "global_gpu_device": default_global_gpu_device,
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
# Runtime global settings start from defaults.
# Persistent global values are sourced only from user universal presets.
global_settings = GLOBAL_DEFAULTS.copy()
global_settings["global_gpu_device"] = resolve_global_gpu_device(global_settings.get("global_gpu_device"))
os.environ["SECOURSES_GLOBAL_GPU_DEVICE"] = global_settings["global_gpu_device"]

# Apply current model cache paths to environment for current session.
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
    /* Keep runtime progress/info area height stable so action buttons do not jump. */
    .runtime-status-box {
      --runtime-status-height: 34px;
      height: var(--runtime-status-height);
      max-height: var(--runtime-status-height);
      overflow: hidden;
    }
    .runtime-status-box .prose,
    .runtime-status-box .md {
      height: 100%;
      margin: 0 !important;
      display: flex;
      align-items: center;
      overflow: hidden;
    }
    .runtime-status-box .prose p,
    .runtime-status-box .md p {
      margin: 0 !important;
      line-height: 1.25;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .runtime-progress-box {
      --runtime-progress-height: 74px;
      height: var(--runtime-progress-height);
      max-height: var(--runtime-progress-height);
      overflow: hidden;
    }
    .runtime-progress-box .prose,
    .runtime-progress-box .md {
      height: 100%;
      max-height: 100%;
      margin: 0 !important;
      display: block;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-gutter: stable;
    }
    .runtime-progress-box .prose p,
    .runtime-progress-box .md p {
      margin: 0 !important;
      line-height: 1.35;
    }
    .runtime-progress-box .processing-banner {
      margin: 0;
      height: 100%;
      max-height: 100%;
      box-sizing: border-box;
      overflow: hidden;
    }
    .runtime-progress-box .processing-col {
      min-width: 0;
      width: 100%;
    }
    .runtime-progress-box .processing-text,
    .runtime-progress-box .processing-sub {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    @media (max-width: 900px) {
      .runtime-status-box {
        --runtime-status-height: 32px;
      }
      .runtime-progress-box {
        --runtime-progress-height: 68px;
      }
    }
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

    /* Optimize parameters */
    .action-btn-optimize button,
    button.action-btn-optimize {
      background: linear-gradient(132deg, #0b1222 0%, #0f766e 43%, #0ea5e9 100%) !important;
      border-color: rgba(153, 246, 228, 1) !important;
      box-shadow: 0 13px 30px rgba(8, 145, 178, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
      letter-spacing: 0.25px;
      font-weight: 800;
    }
    .action-btn-optimize button::after,
    button.action-btn-optimize::after {
      content: "";
      position: absolute;
      inset: -36% auto -36% -26%;
      width: 28%;
      transform: rotate(16deg);
      background: linear-gradient(90deg, rgba(255, 255, 255, 0.0), rgba(255, 255, 255, 0.34), rgba(255, 255, 255, 0.0));
      animation: actionSweep 2.0s linear infinite;
      pointer-events: none;
    }
    .action-btn-optimize button:hover,
    button.action-btn-optimize:hover {
      box-shadow: 0 17px 34px rgba(8, 145, 178, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.27) !important;
      border-color: rgba(204, 251, 241, 1) !important;
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

    /* Resolution tab quick source buttons */
    .quick-actions-row {
      gap: 10px;
      flex-wrap: wrap;
    }
    .quick-actions-row .action-btn button,
    .quick-actions-row button.action-btn {
      min-height: 46px;
      font-size: 0.95rem;
    }
    .action-btn-source-seed button,
    button.action-btn-source-seed {
      background: linear-gradient(135deg, #1d4ed8 0%, #0369a1 52%, #0f766e 100%) !important;
      border-color: rgba(186, 230, 253, 0.95) !important;
      box-shadow: 0 11px 26px rgba(3, 105, 161, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    .action-btn-source-seed button:hover,
    button.action-btn-source-seed:hover {
      box-shadow: 0 14px 30px rgba(3, 105, 161, 0.46), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
    }
    .action-btn-source-gan button,
    button.action-btn-source-gan {
      background: linear-gradient(135deg, #b45309 0%, #ea580c 54%, #dc2626 100%) !important;
      border-color: rgba(254, 215, 170, 0.95) !important;
      box-shadow: 0 11px 26px rgba(194, 65, 12, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    }
    .action-btn-source-gan button:hover,
    button.action-btn-source-gan:hover {
      box-shadow: 0 14px 30px rgba(194, 65, 12, 0.44), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
    }
    .action-btn-source-flash button,
    button.action-btn-source-flash {
      background: linear-gradient(135deg, #15803d 0%, #0f766e 52%, #0891b2 100%) !important;
      border-color: rgba(167, 243, 208, 0.95) !important;
      box-shadow: 0 11px 26px rgba(5, 150, 105, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    .action-btn-source-flash button:hover,
    button.action-btn-source-flash:hover {
      box-shadow: 0 14px 30px rgba(5, 150, 105, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
    }

    /* Musubi-style small file/folder icon button */
    #open_folder_small {
      min-width: 2.25em !important;
      max-width: 2.9em;
      flex-grow: 0;
      padding: 0.4em 0.35em !important;
      font-size: 1.35em !important;
      line-height: 1;
      border-radius: 12px !important;
    }

    @keyframes actionSweep {
      0% {
        left: -32%;
      }
      100% {
        left: 115%;
      }
    }

    /* Health status banner + Health tab report cards */
    .top-status-row {
      gap: 10px;
      align-items: center;
      margin-bottom: 4px;
    }
    .global-gpu-inline {
      gap: 8px;
      align-items: center;
      flex-wrap: nowrap;
      margin-top: 2px;
    }
    .global-gpu-inline .gradio-dropdown {
      min-width: 0;
      width: 100%;
      flex: 1 1 auto;
    }
    .global-gpu-inline .gradio-dropdown .wrap {
      min-height: 38px;
    }
    @media (max-width: 960px) {
      .global-gpu-inline {
        flex-wrap: wrap;
      }
      .global-gpu-inline .gradio-dropdown {
        min-width: 100%;
      }
    }
    .health-banner {
      border: 1px solid rgba(14, 116, 144, 0.32);
      background: linear-gradient(120deg, rgba(2, 132, 199, 0.10), rgba(16, 185, 129, 0.10));
      border-radius: 12px;
      padding: 10px 12px;
      white-space: pre-wrap;
      line-height: 1.45;
      font-weight: 600;
    }
    .health-report-shell {
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 6px;
    }
    .health-report-summary {
      border: 1px solid rgba(148, 163, 184, 0.24);
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.42), rgba(30, 41, 59, 0.28));
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .health-report-summary.is-ok {
      border-color: rgba(34, 197, 94, 0.34);
      background: linear-gradient(135deg, rgba(22, 101, 52, 0.28), rgba(5, 46, 22, 0.25));
    }
    .health-report-summary.is-warning {
      border-color: rgba(245, 158, 11, 0.38);
      background: linear-gradient(135deg, rgba(146, 64, 14, 0.30), rgba(120, 53, 15, 0.22));
    }
    .health-report-summary.is-error {
      border-color: rgba(248, 113, 113, 0.42);
      background: linear-gradient(135deg, rgba(127, 29, 29, 0.34), rgba(69, 10, 10, 0.25));
    }
    .health-report-summary-title {
      font-size: 16px;
      font-weight: 850;
      letter-spacing: 0.2px;
      margin-bottom: 6px;
    }
    .health-report-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .health-report-metrics span {
      border: 1px solid rgba(148, 163, 184, 0.30);
      border-radius: 9999px;
      padding: 3px 10px;
      font-size: 12px;
      background: rgba(15, 23, 42, 0.34);
    }
    .health-report-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .health-report-card {
      border: 1px solid rgba(148, 163, 184, 0.24);
      background: linear-gradient(160deg, rgba(15, 23, 42, 0.40), rgba(30, 41, 59, 0.26));
      border-radius: 14px;
      padding: 12px;
      min-width: 0;
    }
    .health-report-card.is-ok {
      border-color: rgba(74, 222, 128, 0.34);
    }
    .health-report-card.is-warning {
      border-color: rgba(251, 191, 36, 0.42);
    }
    .health-report-card.is-error {
      border-color: rgba(248, 113, 113, 0.48);
    }
    .health-report-card.is-skipped {
      border-color: rgba(148, 163, 184, 0.34);
      opacity: 0.93;
    }
    .health-report-card-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .health-report-card-head h4 {
      margin: 0;
      font-size: 14px;
      font-weight: 800;
      min-width: 0;
    }
    .health-report-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 9999px;
      padding: 3px 9px;
      font-size: 11px;
      font-weight: 850;
      border: 1px solid rgba(148, 163, 184, 0.3);
      background: rgba(15, 23, 42, 0.42);
      text-transform: uppercase;
      letter-spacing: 0.35px;
      white-space: nowrap;
    }
    .health-report-badge.is-ok {
      color: #86efac;
      border-color: rgba(74, 222, 128, 0.55);
    }
    .health-report-badge.is-warning {
      color: #fcd34d;
      border-color: rgba(251, 191, 36, 0.58);
    }
    .health-report-badge.is-error {
      color: #fca5a5;
      border-color: rgba(248, 113, 113, 0.62);
    }
    .health-report-badge.is-skipped {
      color: #cbd5e1;
      border-color: rgba(148, 163, 184, 0.46);
    }
    .health-report-detail {
      margin: 0;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.24);
      background: rgba(2, 6, 23, 0.48);
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.42;
      font-size: 12.5px;
      color: #e2e8f0;
      max-height: 230px;
      overflow: auto;
    }
    /* Main navigation tabs (Gradio Tabs.svelte): always visible in two rows */
    #secourses-main-tabs {
      --main-tab-col-count: 5;
      --main-tab-gap: 6px;
      --main-tab-min-height: 46px;
      --main-tab-row-slack: 16px;
    }
    #secourses-main-tabs .tab-wrapper {
      position: relative;
      display: block;
      height: auto;
      min-height: 0;
      align-items: stretch;
      justify-content: flex-start;
      padding: 10px 10px 12px;
      border: 1px solid rgba(59, 130, 246, 0.26);
      border-radius: 14px;
      background: linear-gradient(130deg, rgba(15, 23, 42, 0.42), rgba(14, 116, 144, 0.20) 48%, rgba(5, 150, 105, 0.20));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 8px 18px rgba(2, 6, 23, 0.20);
    }
    #secourses-main-tabs .tab-container {
      height: auto;
      overflow: visible;
      display: flex;
      flex-wrap: wrap;
      align-items: stretch;
      gap: var(--main-tab-gap);
    }
    #secourses-main-tabs .tab-container::after {
      bottom: -1px;
      border-radius: 9999px;
      background: linear-gradient(90deg, rgba(59, 130, 246, 0.68), rgba(16, 185, 129, 0.68));
      opacity: 0.7;
    }
    #secourses-main-tabs .tab-container.visually-hidden {
      position: absolute !important;
      left: 0 !important;
      top: 0 !important;
      width: 100% !important;
      height: auto !important;
      margin: 0 !important;
      padding: 0 !important;
      clip: auto !important;
      white-space: normal !important;
      overflow: visible !important;
      visibility: hidden !important;
      pointer-events: none !important;
    }
    #secourses-main-tabs .tab-container button,
    #secourses-main-tabs .overflow-menu .overflow-dropdown button {
      flex: 0 0 calc((100% - (var(--main-tab-gap) * (var(--main-tab-col-count) - 1)) - var(--main-tab-row-slack)) / var(--main-tab-col-count));
      max-width: calc((100% - (var(--main-tab-gap) * (var(--main-tab-col-count) - 1)) - var(--main-tab-row-slack)) / var(--main-tab-col-count));
      min-width: 0;
      min-height: var(--main-tab-min-height);
      height: auto;
      margin: 0;
      padding: 6px 8px;
      line-height: 1.24;
      white-space: normal;
      text-align: center;
      justify-content: center;
      border: 1px solid rgba(148, 163, 184, 0.30);
      border-radius: 12px;
      background: linear-gradient(160deg, rgba(15, 23, 42, 0.56), rgba(30, 41, 59, 0.44));
      box-shadow: 0 4px 11px rgba(2, 6, 23, 0.14);
      font-weight: 740;
      font-size: 13.5px;
    }
    #secourses-main-tabs .tab-container button:hover:not(:disabled):not(.selected),
    #secourses-main-tabs .overflow-menu .overflow-dropdown button:hover:not(:disabled):not(.selected) {
      transform: translateY(-1px);
      border-color: rgba(125, 211, 252, 0.66);
      background: linear-gradient(160deg, rgba(14, 116, 144, 0.44), rgba(15, 23, 42, 0.58));
    }
    #secourses-main-tabs .tab-container button.selected,
    #secourses-main-tabs .overflow-menu .overflow-dropdown button.selected {
      color: #e0f2fe;
      border-color: rgba(45, 212, 191, 0.92);
      background: linear-gradient(145deg, rgba(30, 64, 175, 0.72), rgba(13, 148, 136, 0.76));
      box-shadow: 0 8px 18px rgba(15, 118, 110, 0.32);
    }
    #secourses-main-tabs .tab-container button.selected::after,
    #secourses-main-tabs .overflow-menu .overflow-dropdown button.selected::after {
      left: 14%;
      width: 72%;
      bottom: 3px;
      height: 3px;
      border-radius: 999px;
      background-color: rgba(240, 253, 250, 0.96);
    }
    #secourses-main-tabs .overflow-menu.hide {
      display: none !important;
    }
    #secourses-main-tabs .overflow-menu {
      display: block !important;
      width: 100%;
      margin-top: var(--main-tab-gap);
    }
    #secourses-main-tabs .overflow-menu > button {
      display: none !important;
    }
    #secourses-main-tabs .overflow-menu .overflow-dropdown,
    #secourses-main-tabs .overflow-menu .overflow-dropdown.hide {
      position: static !important;
      display: flex !important;
      flex-wrap: wrap !important;
      align-items: stretch;
      gap: var(--main-tab-gap);
      width: 100%;
      margin: 0 !important;
      padding: 0 !important;
      border: 0 !important;
      background: transparent !important;
      box-shadow: none !important;
    }

    @media (max-width: 768px) {
      #secourses-main-tabs {
        --main-tab-gap: 5px;
        --main-tab-min-height: 42px;
        --main-tab-row-slack: 12px;
      }
      #secourses-main-tabs .tab-wrapper {
        padding: 8px 8px 10px;
      }
      #secourses-main-tabs .tab-container button,
      #secourses-main-tabs .overflow-menu .overflow-dropdown button {
        padding: 5px 6px;
        font-size: 12px;
      }
      .action-btn button,
      button.action-btn {
        border-radius: 12px !important;
      }
      .action-btn-upscale button,
      button.action-btn-upscale {
        box-shadow: 0 10px 24px rgba(21, 128, 61, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
      }
      .health-report-grid {
        grid-template-columns: 1fr;
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
        )
        from shared.models import (
            get_seedvr2_model_names,
            get_seedvr2_models,
            scan_gan_models,
            get_flashvsr_model_names,
            get_rife_model_names,
        )
        
        # Get models list for defaults
        seedvr2_models = get_seedvr2_model_names()
        # Ensure separate GGUF Q8_0 variants are available in shared model choices.
        # These are intentionally downloaded via the dedicated GGUF/FP8 downloader.
        gguf_q8_variants = [
            "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_7b-Q8_0.gguf",
            "seedvr2_ema_7b_sharp-Q8_0.gguf",
        ]
        try:
            discovered_seedvr2 = {m.name for m in get_seedvr2_models()}
        except Exception:
            discovered_seedvr2 = set(seedvr2_models)
        for model_name in gguf_q8_variants:
            if model_name in discovered_seedvr2 and model_name not in seedvr2_models:
                seedvr2_models.append(model_name)
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

        def _runtime_global_defaults() -> Dict[str, Any]:
            return {
                "output_dir": str(global_settings.get("output_dir", BASE_DIR / "outputs")),
                "temp_dir": str(global_settings.get("temp_dir", BASE_DIR / "temp")),
                "theme_mode": str(global_settings.get("theme_mode", "dark") or "dark"),
                "telemetry": bool(global_settings.get("telemetry", True)),
                "face_global": bool(global_settings.get("face_global", False)),
                "face_strength": float(global_settings.get("face_strength", 0.5)),
                "queue_enabled": bool(global_settings.get("queue_enabled", True)),
                "global_gpu_device": resolve_global_gpu_device(global_settings.get("global_gpu_device")),
                "mode": str(global_settings.get("mode", "subprocess") or "subprocess"),
                "models_dir": str(global_settings.get("models_dir", BASE_DIR / "models")),
                "hf_home": str(global_settings.get("hf_home", BASE_DIR / "models")),
                "transformers_cache": str(global_settings.get("transformers_cache", BASE_DIR / "models")),
                "pinned_reference_path": global_settings.get("pinned_reference_path"),
            }

        if loaded_preset:
            # Merge with defaults to fill any missing keys
            merged_preset = merge_preset_with_defaults(loaded_preset, BASE_DIR, all_models)
            merged_global = _runtime_global_defaults()
            loaded_global = merged_preset.get("global", {})
            if isinstance(loaded_global, dict):
                for key, value in loaded_global.items():
                    if value is not None:
                        merged_global[key] = value
            merged_preset["global"] = merged_global
            return merged_preset, last_preset_name, all_models
        else:
            # Use defaults
            defaults = get_all_defaults(BASE_DIR, all_models)
            defaults["global"] = _runtime_global_defaults()
            return defaults, None, all_models
    
    # Load universal preset on startup
    startup_preset, startup_preset_name, all_models = load_startup_universal_preset()
    sync_defaults = get_all_defaults(BASE_DIR, all_models)

    def _merge_global_values(raw: Dict[str, Any] | None) -> Dict[str, Any]:
        merged = {key: global_settings.get(key) for key in GLOBAL_ORDER}
        for key in GLOBAL_ORDER:
            if merged.get(key) is None:
                merged[key] = sync_defaults.get("global", {}).get(key)
        if isinstance(raw, dict):
            for key in GLOBAL_ORDER:
                if key in raw and raw.get(key) is not None:
                    merged[key] = raw.get(key)
        merged["output_dir"] = str(merged.get("output_dir", "") or "")
        merged["temp_dir"] = str(merged.get("temp_dir", "") or "")
        theme_raw = str(merged.get("theme_mode", "dark") or "dark").strip().lower()
        merged["theme_mode"] = theme_raw if theme_raw in {"dark", "light"} else "dark"
        merged["telemetry"] = bool(merged.get("telemetry", True))
        merged["face_global"] = bool(merged.get("face_global", False))
        try:
            merged["face_strength"] = max(0.0, min(1.0, float(merged.get("face_strength", 0.5) or 0.5)))
        except Exception:
            merged["face_strength"] = 0.5
        merged["queue_enabled"] = bool(merged.get("queue_enabled", True))
        merged["global_gpu_device"] = resolve_global_gpu_device(merged.get("global_gpu_device"))
        mode_raw = str(merged.get("mode", "subprocess") or "subprocess").strip().lower()
        merged["mode"] = mode_raw if mode_raw in {"subprocess", "in_app"} else "subprocess"
        merged["models_dir"] = str(merged.get("models_dir", "") or "")
        merged["hf_home"] = str(merged.get("hf_home", "") or "")
        merged["transformers_cache"] = str(merged.get("transformers_cache", "") or "")
        merged["pinned_reference_path"] = str(merged.get("pinned_reference_path", "") or "")
        return merged

    def apply_global_settings_live(
        od,
        td,
        theme_mode,
        tel,
        face_str,
        queue_enabled,
        global_gpu_device,
        mode_choice,
        models_dir,
        hf_home,
        trans_cache,
        state,
    ):
        from shared.services.global_service import apply_global_settings_live as _apply_global_settings_live

        return _apply_global_settings_live(
            output_dir_val=od,
            temp_dir_val=td,
            theme_mode_val=theme_mode,
            telemetry_enabled=tel,
            face_strength=face_str,
            queue_enabled=queue_enabled,
            global_gpu_device_val=global_gpu_device,
            mode_choice=mode_choice,
            models_dir_val=models_dir,
            hf_home_val=hf_home,
            transformers_cache_val=trans_cache,
            runner=runner,
            preset_manager=preset_manager,
            global_settings=global_settings,
            run_logger=run_logger,
            state=state,
        )

    startup_global_settings = _merge_global_values(startup_preset.get("global", {}))
    startup_preset["global"] = startup_global_settings
    _startup_state = {
        "seed_controls": {
            "global_settings": dict(startup_global_settings),
            "pinned_reference_path": startup_global_settings.get("pinned_reference_path"),
        }
    }
    _startup_status, _startup_mode, _startup_state = apply_global_settings_live(
        startup_global_settings.get("output_dir", global_settings.get("output_dir")),
        startup_global_settings.get("temp_dir", global_settings.get("temp_dir")),
        startup_global_settings.get("theme_mode", global_settings.get("theme_mode", "dark")),
        bool(startup_global_settings.get("telemetry", global_settings.get("telemetry", True))),
        float(startup_global_settings.get("face_strength", global_settings.get("face_strength", 0.5))),
        bool(startup_global_settings.get("queue_enabled", global_settings.get("queue_enabled", True))),
        startup_global_settings.get("global_gpu_device", global_settings.get("global_gpu_device", "cpu")),
        str(startup_global_settings.get("mode", global_settings.get("mode", "subprocess")) or "subprocess"),
        startup_global_settings.get("models_dir", global_settings.get("models_dir")),
        startup_global_settings.get("hf_home", global_settings.get("hf_home")),
        startup_global_settings.get("transformers_cache", global_settings.get("transformers_cache")),
        _startup_state,
    )
    startup_global_settings = _merge_global_values(_startup_state.get("seed_controls", {}).get("global_settings", {}))
    startup_preset["global"] = startup_global_settings
    active_temp_dir = Path(global_settings.get("temp_dir", temp_dir))
    active_output_dir = Path(global_settings.get("output_dir", output_dir))
    startup_theme_mode = str(startup_global_settings.get("theme_mode", global_settings.get("theme_mode", "dark")) or "dark")
    if startup_theme_mode not in {"dark", "light"}:
        startup_theme_mode = "dark"

    theme_bootstrap_head = f"""
    <script>
    (() => {{
      try {{
        const preferredTheme = {json.dumps(startup_theme_mode)};
        if (!preferredTheme || (preferredTheme !== "dark" && preferredTheme !== "light")) return;
        const url = new URL(window.location.href);
        if (url.searchParams.get("__theme") !== preferredTheme) {{
          url.searchParams.set("__theme", preferredTheme);
          window.history.replaceState(null, "", url.toString());
        }}
        const applyTheme = () => {{
          if (preferredTheme === "dark") document.body.classList.add("dark");
          else document.body.classList.remove("dark");
        }};
        if (document.readyState === "loading") {{
          document.addEventListener("DOMContentLoaded", applyTheme, {{ once: true }});
        }} else {{
          applyTheme();
        }}
      }} catch (_) {{}}
    }})();
    </script>
    """
    
    with gr.Blocks(title=APP_TITLE) as demo:
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
                "global_settings": startup_global_settings,
                "seedvr2_settings": startup_preset.get("seedvr2", {}),
                "gan_settings": startup_preset.get("gan", {}),
                "rife_settings": startup_preset.get("rife", {}),
                "flashvsr_settings": startup_preset.get("flashvsr", {}),
                "face_settings": startup_preset.get("face", {}),
                "resolution_settings": startup_preset.get("resolution", {}),
                "output_settings": startup_preset.get("output", {}),
                
                # Individual cached values (for backward compatibility with other code)
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
                "image_output_format_val": startup_output_settings.get("image_output_format", "png"),
                "image_output_quality_val": startup_output_settings.get("image_output_quality", 95),
                "seedvr2_video_backend_val": startup_output_settings.get("seedvr2_video_backend", "opencv"),
                "seedvr2_use_10bit_val": bool(startup_output_settings.get("seedvr2_use_10bit", False)),
                "video_codec_val": str(startup_output_settings.get("video_codec", "h264") or "h264"),
                "video_quality_val": int(startup_output_settings.get("video_quality", 18) or 18),
                "video_preset_val": str(startup_output_settings.get("video_preset", "medium") or "medium"),
                "h265_tune_val": str(startup_output_settings.get("h265_tune", "none") or "none"),
                "av1_film_grain_val": int(
                    startup_output_settings.get("av1_film_grain", DEFAULT_AV1_FILM_GRAIN) or DEFAULT_AV1_FILM_GRAIN
                ),
                "av1_film_grain_denoise_val": bool(
                    startup_output_settings.get("av1_film_grain_denoise", DEFAULT_AV1_FILM_GRAIN_DENOISE)
                ),
                "two_pass_encoding_val": bool(startup_output_settings.get("two_pass_encoding", False)),
                "pixel_format_val": str(startup_output_settings.get("pixel_format", "yuv420p") or "yuv420p"),
                "frame_interpolation_val": bool(startup_output_settings.get("frame_interpolation", False)),
                "global_rife_enabled_val": bool(startup_output_settings.get("frame_interpolation", False)),
                "global_rife_multiplier_val": startup_output_settings.get("global_rife_multiplier", "x2"),
                "global_rife_model_val": startup_output_settings.get("global_rife_model", get_rife_default_model()),
                "global_rife_precision_val": startup_output_settings.get("global_rife_precision", "fp32"),
                "global_gpu_device_val": resolve_global_gpu_device(startup_global_settings.get("global_gpu_device")),
                "global_rife_cuda_device_val": (
                    "" if resolve_global_gpu_device(startup_global_settings.get("global_gpu_device")) == "cpu"
                    else resolve_global_gpu_device(startup_global_settings.get("global_gpu_device"))
                ),
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
                "face_strength_val": float(startup_global_settings.get("face_strength", global_settings.get("face_strength", 0.5))),
                "queue_enabled_val": bool(startup_global_settings.get("queue_enabled", global_settings.get("queue_enabled", True))),
                "theme_mode_val": str(startup_global_settings.get("theme_mode", global_settings.get("theme_mode", "dark")) or "dark"),
                
                # Resolution tab cached values
                "auto_chunk": startup_auto_chunk,
                "auto_detect_scenes": bool(startup_res_settings.get("auto_detect_scenes", True)),
                "frame_accurate_split": bool(startup_res_settings.get("frame_accurate_split", True)),
                "chunk_size_sec": startup_res_settings.get("chunk_size", 0),
                "chunk_overlap_sec": startup_chunk_overlap_sec,
                "per_chunk_cleanup": startup_res_settings.get("per_chunk_cleanup", False),
                "scene_threshold": startup_res_settings.get("scene_threshold", 27.0),
                "min_scene_len": startup_res_settings.get("min_scene_len", 1.0),
                
                # Pinned reference (persisted globally)
                "pinned_reference_path": startup_global_settings.get("pinned_reference_path", global_settings.get("pinned_reference_path")),
                
                # Available models list (for preset defaults)
                "available_models": all_models,
            },
            "operation_status": "ready"
        })

        initial_gpu_list = get_gpu_info()
        initial_gpu_choices = build_global_gpu_dropdown_choices(initial_gpu_list)
        initial_global_gpu_value = resolve_global_gpu_device(
            startup_global_settings.get("global_gpu_device"),
            initial_gpu_list,
        )
        if all(str(value) != str(initial_global_gpu_value) for _label, value in initial_gpu_choices):
            initial_gpu_choices.append((describe_gpu_selection(initial_global_gpu_value), initial_global_gpu_value))

        # Top status row: left = global GPU selector, right = health banner
        with gr.Row(equal_height=True, elem_classes="top-status-row"):
            with gr.Column(scale=1):
                with gr.Row(elem_classes="global-gpu-inline"):
                    global_gpu_dropdown = gr.Dropdown(
                        choices=initial_gpu_choices,
                        value=initial_global_gpu_value,
                        show_label=False,
                        container=False,
                    )
            with gr.Column(scale=1):
                health_banner = gr.Markdown(f'<div class="health-banner">{health_text}</div>')

        # VRAM OOM banner (shown only on VRAM OOM)
        # NOTE: We update this via a Timer tick because gr.State change events can be
        # inconsistent across Gradio versions/environments.
        oom_banner = gr.HTML(value="", visible=False)
        oom_dismiss_btn = gr.Button("Dismiss VRAM Alert", variant="secondary", size="sm", visible=False)
        oom_timer = gr.Timer(value=2.0, active=True)
        health_sync_signature = gr.State(value="")
        oom_sync_signature = gr.State(value="")
        global_sync_signature = gr.State(value="")
        gpu_selector_sync_signature = gr.State(value="")
        gr.Markdown(f"# {APP_TITLE}")

        # Global settings tab (rendered LAST for a cleaner workflow)
        def render_global_settings_tab():
            with gr.Tab("⚙️ Global Settings", render_children=True) as tab_global:
                gr.Markdown("### Global Settings")
                gr.Markdown("Configure absolute output/temp directories and choose light or dark theme.")
                with gr.Row():
                    output_dir_box = gr.Textbox(
                        label="Default Outputs Folder",
                        value=global_settings["output_dir"],
                        placeholder=str(BASE_DIR / "outputs"),
                        info="Absolute path required. Supports /, //, and \\ separators."
                    )
                    temp_dir_box = gr.Textbox(
                        label="Default Temp Folder",
                        value=global_settings["temp_dir"],
                        placeholder=str(BASE_DIR / "temp"),
                        info="Absolute path required. Supports /, //, and \\ separators."
                    )

                theme_mode_default = str(global_settings.get("theme_mode", "dark") or "dark").strip().lower()
                if theme_mode_default not in {"dark", "light"}:
                    theme_mode_default = "dark"

                theme_mode_radio = gr.Radio(
                    choices=["dark", "light"],
                    value=theme_mode_default,
                    label="Theme",
                    info="Select the Gradio UI theme mode."
                )
                theme_mode_radio.change(
                    fn=None,
                    inputs=[theme_mode_radio],
                    outputs=None,
                    queue=False,
                    show_progress="hidden",
                    js="""
                    (mode) => {
                      try {
                        const chosen = (mode === "dark" || mode === "light") ? mode : "dark";
                        const url = new URL(window.location.href);
                        url.searchParams.set("__theme", chosen);
                        window.history.replaceState(null, "", url.toString());
                        if (chosen === "dark") document.body.classList.add("dark");
                        else document.body.classList.remove("dark");
                      } catch (_) {}
                    }
                    """,
                )
                global_status = gr.Markdown("")

                # Hidden compatibility controls for existing global preset schema.
                telemetry_toggle = gr.Checkbox(
                    label="Save run metadata (local telemetry)",
                    value=global_settings.get("telemetry", True),
                    visible=False,
                )
                face_global_hidden = gr.Checkbox(
                    label="Global Face Restore (synced from Face tab)",
                    value=bool(global_settings.get("face_global", False)),
                    interactive=False,
                    visible=False,
                )
                face_strength_slider = gr.Slider(
                    label="Global Face Restoration Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=global_settings.get("face_strength", 0.5),
                    visible=False,
                )
                queue_enabled_toggle = gr.Checkbox(
                    label="Enable Queue",
                    value=bool(global_settings.get("queue_enabled", True)),
                    visible=False,
                )
                mode_radio = gr.Radio(
                    choices=["subprocess", "in_app"],
                    value=str(global_settings.get("mode", "subprocess") or "subprocess"),
                    label="Processing Mode",
                    visible=False,
                    interactive=True
                )
                models_dir_box = gr.Textbox(
                    label="Models Directory (MODELS_DIR)",
                    value=global_settings.get("models_dir", ""),
                    visible=False,
                )
                hf_home_box = gr.Textbox(
                    label="HuggingFace Home (HF_HOME)",
                    value=global_settings.get("hf_home", ""),
                    visible=False,
                )
                transformers_cache_box = gr.Textbox(
                    label="Transformers Cache (TRANSFORMERS_CACHE)",
                    value=global_settings.get("transformers_cache", ""),
                    visible=False,
                )
                pinned_reference_path_hidden = gr.Textbox(
                    label="Pinned Reference Path",
                    value=str(global_settings.get("pinned_reference_path", "") or ""),
                    interactive=False,
                    visible=False,
                )

                (
                    preset_dropdown,
                    preset_name_input,
                    save_preset_btn,
                    load_preset_btn,
                    preset_status,
                    reset_defaults_btn,
                    delete_preset_btn,
                    preset_callbacks,
                ) = universal_preset_section(
                    preset_manager=preset_manager,
                    shared_state=shared_state,
                    tab_name="global",
                    inputs_list=[],
                    base_dir=BASE_DIR,
                    models_list=all_models,
                    open_accordion=True,
                )

                # Must match GLOBAL_ORDER in shared/universal_preset.py
                global_preset_inputs = [
                    output_dir_box,
                    temp_dir_box,
                    theme_mode_radio,
                    telemetry_toggle,
                    face_global_hidden,
                    face_strength_slider,
                    queue_enabled_toggle,
                    global_gpu_dropdown,
                    mode_radio,
                    models_dir_box,
                    hf_home_box,
                    transformers_cache_box,
                    pinned_reference_path_hidden,
                ]
                wire_universal_preset_events(
                    preset_dropdown=preset_dropdown,
                    preset_name_input=preset_name_input,
                    save_btn=save_preset_btn,
                    load_btn=load_preset_btn,
                    preset_status=preset_status,
                    reset_btn=reset_defaults_btn,
                    delete_btn=delete_preset_btn,
                    callbacks=preset_callbacks,
                    inputs_list=global_preset_inputs,
                    shared_state=shared_state,
                    tab_name="global",
                )

                return {
                    "tab": tab_global,
                    "inputs_list": global_preset_inputs,
                    "preset_dropdown": preset_dropdown,
                    "preset_status": preset_status,
                    "global_status": global_status,
                    "mode_radio": mode_radio,
                }


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
                if tab_name == "output":
                    try:
                        codec_idx = OUTPUT_ORDER.index("video_codec")
                        pix_fmt_idx = OUTPUT_ORDER.index("pixel_format")
                        codec_val = str(values[codec_idx] or "h264").strip().lower()
                        pix_fmt_choices = get_pixel_format_choices(codec_val)
                        pix_fmt_fallback = pix_fmt_choices[0] if pix_fmt_choices else "yuv420p"
                        pix_fmt_val = str(values[pix_fmt_idx] or pix_fmt_fallback).strip().lower()
                        if pix_fmt_val not in pix_fmt_choices:
                            pix_fmt_val = pix_fmt_fallback
                        values[pix_fmt_idx] = gr.update(choices=pix_fmt_choices, value=pix_fmt_val)

                        global_rife_model_idx = OUTPUT_ORDER.index("global_rife_model")
                        if len(values) > global_rife_model_idx and not str(values[global_rife_model_idx] or "").strip():
                            values[global_rife_model_idx] = get_rife_default_model()
                    except Exception:
                        pass

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
        tab_sync_global = gr.State(value="")
        seedvr2_auto_res_sync = gr.State(value="")

        # Self-contained tabs following SECourses pattern
        with gr.Tabs(elem_id="secourses-main-tabs", elem_classes=["secourses-main-tabs"]):
            with gr.Tab("🎬 SeedVR2", render_children=True) as tab_seedvr2:
                seedvr2_ui = seedvr2_tab(
                    preset_manager=preset_manager,
                    runner=runner,
                    run_logger=run_logger,
                    global_settings=global_settings,
                    shared_state=shared_state,
                    base_dir=BASE_DIR,
                    temp_dir=active_temp_dir,
                    output_dir=active_output_dir
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

            with gr.Tab("⚡ FlashVSR+", render_children=True) as tab_flashvsr:
                flashvsr_ui = flashvsr_tab(
                    preset_manager=preset_manager,
                    runner=runner,
                    run_logger=run_logger,
                    global_settings=global_settings,
                    shared_state=shared_state,
                    base_dir=BASE_DIR,
                    temp_dir=active_temp_dir,
                    output_dir=active_output_dir
                )
            tab_flashvsr.select(
                fn=_make_tab_sync("flashvsr"),
                inputs=[shared_state, tab_sync_flashvsr],
                outputs=flashvsr_ui["inputs_list"] + [flashvsr_ui["preset_dropdown"], flashvsr_ui["preset_status"], tab_sync_flashvsr],
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

            with gr.Tab("🖼️ Image-Based (GAN)", render_children=True) as tab_gan:
                gan_ui = gan_tab(
                    preset_manager=preset_manager,
                    runner=runner,
                    run_logger=run_logger,
                    global_settings=global_settings,
                    shared_state=shared_state,
                    base_dir=BASE_DIR,
                    temp_dir=active_temp_dir,
                    output_dir=active_output_dir
                )
            tab_gan.select(
                fn=_make_tab_sync("gan"),
                inputs=[shared_state, tab_sync_gan],
                outputs=gan_ui["inputs_list"] + [gan_ui["preset_dropdown"], gan_ui["preset_status"], tab_sync_gan],
                queue=False,
                show_progress="hidden",
                trigger_mode="always_last",
            )

            with gr.Tab("🔍 Output & Comparison", render_children=True) as tab_output:
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

            with gr.Tab("⏱️ RIFE / FPS / Edit Videos", render_children=True) as tab_rife:
                rife_ui = rife_tab(
                    preset_manager=preset_manager,
                    runner=runner,
                    run_logger=run_logger,
                    global_settings=global_settings,
                    shared_state=shared_state,
                    base_dir=BASE_DIR,
                    temp_dir=active_temp_dir,
                    output_dir=active_output_dir
                )
            tab_rife.select(
                fn=_make_tab_sync("rife"),
                inputs=[shared_state, tab_sync_rife],
                outputs=rife_ui["inputs_list"] + [rife_ui["preset_dropdown"], rife_ui["preset_status"], tab_sync_rife],
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

            with gr.Tab("⏳ Queue (0)", render_children=True) as tab_queue:
                queue_tab(tab_queue)

            with gr.Tab("🏥 Health Check", render_children=True):
                health_tab(
                    global_settings=global_settings,
                    shared_state=shared_state,
                    temp_dir=active_temp_dir,
                    output_dir=active_output_dir
                )

            # Global Settings should be the last tab (far-right)
            global_ui = render_global_settings_tab()
            global_ui["tab"].select(
                fn=_make_tab_sync("global"),
                inputs=[shared_state, tab_sync_global],
                outputs=global_ui["inputs_list"] + [global_ui["preset_dropdown"], global_ui["preset_status"], tab_sync_global],
                queue=False,
                show_progress="hidden",
                trigger_mode="always_last",
            )

        # Update health banner on load and changes
        def update_health_banner(state, previous_signature: str = ""):
            """Update health banner with current state"""
            health_text = state.get("health_banner", {}).get("text", "System ready")
            signature = _sync_signature({"health_text": health_text})
            if signature == str(previous_signature or ""):
                return gr.skip()
            return gr.update(value=f'<div class="health-banner">{health_text}</div>'), signature

        def update_global_gpu_dropdown(state, previous_signature: str = ""):
            """Keep top global GPU selector synced with shared global settings."""
            seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
            state_global = seed_controls.get("global_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(state_global, dict):
                state_global = {}

            gpu_list = get_gpu_info()
            selected = resolve_global_gpu_device(
                state_global.get("global_gpu_device", seed_controls.get("global_gpu_device_val")),
                gpu_list,
            )
            choices = build_global_gpu_dropdown_choices(gpu_list)
            if all(str(value) != str(selected) for _label, value in choices):
                choices.append((describe_gpu_selection(selected, gpu_list), selected))

            signature = _sync_signature({"selected": selected, "choices": choices})
            if signature == str(previous_signature or ""):
                return gr.skip()
            return gr.update(choices=choices, value=selected), signature

        def update_oom_banner(state, previous_signature: str = ""):
            """Update global VRAM OOM banner."""
            info = (state or {}).get("alerts", {}).get("oom", {}) if isinstance(state, dict) else {}
            html = info.get("html", "") if isinstance(info, dict) else ""
            visible = bool(isinstance(info, dict) and info.get("visible") and html)
            signature = _sync_signature({"visible": visible, "html": html if visible else ""})
            if signature == str(previous_signature or ""):
                return gr.skip()
            return gr.update(value=html or "", visible=visible), gr.update(visible=visible), signature

        def apply_global_settings_from_state(state, previous_signature: str = ""):
            """
            Keep runtime global settings (runner/env/shared_state) in sync with
            universal preset loads from any tab.
            """
            seed_controls = (state or {}).get("seed_controls", {}) if isinstance(state, dict) else {}
            global_cfg = seed_controls.get("global_settings", {}) if isinstance(seed_controls, dict) else {}
            if not isinstance(global_cfg, dict):
                global_cfg = {}

            merged_global = _merge_global_values(global_cfg)
            signature = _sync_signature({"global_settings": merged_global})
            if signature == str(previous_signature or ""):
                return gr.skip()

            status_upd, mode_upd, next_state = apply_global_settings_live(
                merged_global.get("output_dir", global_settings.get("output_dir")),
                merged_global.get("temp_dir", global_settings.get("temp_dir")),
                merged_global.get("theme_mode", global_settings.get("theme_mode", "dark")),
                bool(merged_global.get("telemetry", global_settings.get("telemetry", True))),
                float(merged_global.get("face_strength", global_settings.get("face_strength", 0.5))),
                bool(merged_global.get("queue_enabled", global_settings.get("queue_enabled", True))),
                merged_global.get("global_gpu_device", global_settings.get("global_gpu_device", "cpu")),
                str(merged_global.get("mode", global_settings.get("mode", "subprocess")) or "subprocess"),
                merged_global.get("models_dir", global_settings.get("models_dir")),
                merged_global.get("hf_home", global_settings.get("hf_home")),
                merged_global.get("transformers_cache", global_settings.get("transformers_cache")),
                state,
            )
            return status_upd, mode_upd, next_state, signature

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
            fn=update_global_gpu_dropdown,
            inputs=[shared_state, gpu_selector_sync_signature],
            outputs=[global_gpu_dropdown, gpu_selector_sync_signature],
        )
        demo.load(
            fn=update_oom_banner,
            inputs=[shared_state, oom_sync_signature],
            outputs=[oom_banner, oom_dismiss_btn, oom_sync_signature],
        )
        demo.load(
            fn=apply_global_settings_from_state,
            inputs=[shared_state, global_sync_signature],
            outputs=[global_ui["global_status"], global_ui["mode_radio"], shared_state, global_sync_signature],
        )
        
        # Update when shared state changes (for dynamic updates from tabs)
        shared_state.change(
            fn=update_health_banner,
            inputs=[shared_state, health_sync_signature],
            outputs=[health_banner, health_sync_signature],
        )
        shared_state.change(
            fn=update_global_gpu_dropdown,
            inputs=[shared_state, gpu_selector_sync_signature],
            outputs=[global_gpu_dropdown, gpu_selector_sync_signature],
        )
        shared_state.change(
            fn=update_oom_banner,
            inputs=[shared_state, oom_sync_signature],
            outputs=[oom_banner, oom_dismiss_btn, oom_sync_signature],
        )
        shared_state.change(
            fn=apply_global_settings_from_state,
            inputs=[shared_state, global_sync_signature],
            outputs=[global_ui["global_status"], global_ui["mode_radio"], shared_state, global_sync_signature],
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
    launch_allowed_paths = _build_launch_allowed_paths(output_dir=active_output_dir, temp_dir=active_temp_dir)
    launch_kwargs = {
        "inbrowser": True,
        "allowed_paths": launch_allowed_paths,
        "share": share_enabled,
        "theme": modern_theme,
        "css": CUSTOM_CSS,
        "head": CUSTOM_HEAD + theme_bootstrap_head,
    }
    if launch_cli_args.server_name:
        launch_kwargs["server_name"] = launch_cli_args.server_name
    if launch_cli_args.server_port is not None:
        launch_kwargs["server_port"] = launch_cli_args.server_port
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
