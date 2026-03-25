import io
import os
import platform
import re
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .path_utils import (
    emit_metadata,
    ffmpeg_set_fps,
    get_media_fps,
    normalize_path,
    resolve_output_location,
    rife_output_path,
    write_png_metadata,
    detect_input_type,
)
from .model_manager import get_model_manager, ModelType
from .command_logger import get_command_logger
from .video_codec_options import build_ffmpeg_video_encode_args


class RunResult:
    def __init__(self, returncode: int, output_path: Optional[str], log: str):
        self.returncode = returncode
        self.output_path = output_path
        self.log = log


def _normalize_rife_sequence_format(raw: Any) -> str:
    fmt = str(raw or "png").strip().lower()
    return "jpg" if fmt in {"jpg", "jpeg"} else "png"


def _clear_rife_sequence_staging(staging_dir: Path) -> None:
    if not staging_dir.exists():
        staging_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in staging_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass


def _sync_rife_sequence_staging(staging_dir: Path, output_dir: Path) -> int:
    if not staging_dir.exists():
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in staging_dir.iterdir():
        if not src.is_file():
            continue
        dest = output_dir / src.name
        try:
            src_stat = src.stat()
            should_copy = (
                (not dest.exists())
                or src_stat.st_size != dest.stat().st_size
                or src_stat.st_mtime_ns > dest.stat().st_mtime_ns
            )
        except Exception:
            should_copy = not dest.exists()
        if not should_copy:
            continue
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception:
            pass
    return copied


def _finalize_rife_sequence_output(
    output_dir: Path,
    settings: Dict[str, Any],
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Optional[str]]:
    if not output_dir.exists() or not output_dir.is_dir():
        return output_dir, None

    sequence_format = _normalize_rife_sequence_format(settings.get("sequence_format", "png"))
    if sequence_format != "jpg":
        return output_dir, None

    try:
        quality = int(float(settings.get("sequence_quality", 95) or 95))
    except Exception:
        quality = 95
    quality = max(1, min(100, quality))

    frame_paths = sorted(
        p for p in output_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )
    jpg_paths = sorted(
        p for p in output_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )

    metadata_path = output_dir / ".png_settings.json"
    def _write_sequence_meta() -> None:
        try:
            import json

            meta: Dict[str, Any] = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        meta = loaded
            meta.setdefault("padding", int(settings.get("png_padding", 6) or 6))
            meta.setdefault("keep_basename", bool(settings.get("png_keep_basename", False)))
            meta.setdefault("base_name", output_dir.name)
            meta["format"] = "jpg"
            meta["quality"] = quality
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    if not frame_paths:
        if jpg_paths:
            _write_sequence_meta()
            return output_dir, None
        return output_dir, None

    try:
        from PIL import Image

        temp_outputs: List[Tuple[Path, Path, Path]] = []
        for src in frame_paths:
            tmp_dst = src.with_suffix(".jpg.tmp")
            final_dst = src.with_suffix(".jpg")
            with Image.open(src) as img:
                if "A" in img.getbands():
                    alpha = img.getchannel("A")
                    base = Image.new("RGB", img.size, (0, 0, 0))
                    base.paste(img.convert("RGBA"), mask=alpha)
                    save_img = base
                else:
                    save_img = img.convert("RGB")
                save_img.save(tmp_dst, format="JPEG", quality=quality)
            temp_outputs.append((src, tmp_dst, final_dst))

        for src, tmp_dst, final_dst in temp_outputs:
            final_dst.unlink(missing_ok=True)
            tmp_dst.replace(final_dst)
            src.unlink(missing_ok=True)

        _write_sequence_meta()

        message = f"Converted PNG sequence to JPG ({len(frame_paths)} frames, quality {quality})."
        if on_progress:
            on_progress(message + "\n")
        return output_dir, message
    except Exception as exc:
        for tmp_file in output_dir.glob("*.jpg.tmp"):
            tmp_file.unlink(missing_ok=True)
        warning = f"WARNING: JPG sequence conversion failed; keeping PNG frames. {exc}"
        if on_progress:
            on_progress(warning + "\n")
        return output_dir, warning


class Runner:
    """
    Wrapper for invoking model CLIs with cancellation support.

    Defaults to subprocess mode. An "in-app" mode executes inline (no subprocess, not cancelable mid-run).
    """

    def __init__(self, base_dir: Path, temp_dir: Path, output_dir: Path, telemetry_enabled: bool = True):
        self.base_dir = Path(base_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self._active_process: Optional[subprocess.Popen] = None
        # Guard against overlapping SeedVR2 subprocess runs. One active SeedVR2
        # run at a time keeps cancellation/process tracking deterministic.
        self._seedvr2_inflight = False
        self._active_mode = "subprocess"
        self._log_lines: List[str] = []
        self._canceled = False
        self._telemetry_enabled = telemetry_enabled
        self._last_model_id: Optional[str] = None
        self._model_manager = get_model_manager()
        
        # Initialize command logger
        executed_commands_dir = self.base_dir.parent / "executed_commands"
        self._command_logger = get_command_logger(executed_commands_dir)

    # ------------------------------------------------------------------ #
    # Mode management
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str):
        """
        Set execution mode: 'subprocess' or 'in_app'.
        
        MODEL-SPECIFIC BEHAVIOR:
        - SeedVR2: Always uses subprocess for CLI execution (even in 'in_app' mode).
                   CLI architecture prevents persistent model caching.
        - GAN: Can benefit from in-app mode (models can persist between runs).
        - RIFE: Can benefit from in-app mode (models can persist between runs).
        - FlashVSR+: Similar to SeedVR2, CLI-based (limited in-app benefit).
        
        IMPORTANT: In-app mode has NO cancellation support and requires manual
        vcvars activation on Windows for torch.compile. Subprocess mode is
        STRONGLY RECOMMENDED for all use cases.
        """
        if mode not in ("subprocess", "in_app"):
            raise ValueError("Invalid mode")
        self._active_mode = mode

    def get_mode(self) -> str:
        return self._active_mode

    def set_telemetry(self, enabled: bool):
        """Toggle metadata emission (run summaries) at runtime."""
        self._telemetry_enabled = bool(enabled)

    def reset_cancel_state(self):
        """Clear stale cancellation state before a new processing run."""
        with self._lock:
            self._canceled = False

    # ------------------------------------------------------------------ #
    # Cancellation
    # ------------------------------------------------------------------ #
    def cancel(self) -> bool:
        """
        Cancel the active subprocess.
        
        Handles platform-specific termination:
        - Windows: Uses CTRL_BREAK_EVENT then terminate/kill
        - Unix: Uses SIGTERM then SIGKILL
        
        Returns True if cancellation was attempted, False if no active process.
        """
        with self._lock:
            proc = self._active_process
            if not proc:
                # A SeedVR2 run may be in pre-launch setup (vcvars/env prep) before
                # Popen is assigned. Mark cancel requested so launch can be aborted.
                if self._seedvr2_inflight:
                    self._canceled = True
                    return True
                # No active process to cancel. Ensure stale cancel state does not
                # poison the next queued/chunked run.
                self._canceled = False
                return False
            self._canceled = True
        
        try:
            if platform.system() == "Windows":
                # Windows-specific graceful shutdown
                try:
                    # First try CTRL_BREAK_EVENT (only works if CREATE_NEW_PROCESS_GROUP was used)
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    
                    # Wait briefly for graceful shutdown
                    try:
                        proc.wait(timeout=2.0)
                        return True  # Process exited gracefully
                    except subprocess.TimeoutExpired:
                        pass
                except (OSError, AttributeError):
                    # CTRL_BREAK might not work, continue to terminate
                    pass
                
                # Try terminate
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                        return True
                    except subprocess.TimeoutExpired:
                        pass
                except OSError:
                    pass
                
                # Force kill as last resort
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass

                # EXTRA SAFETY: kill the whole process tree (SeedVR2 may spawn helpers).
                # Prevents orphaned child processes from holding VRAM/handles after cancel.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:
                    pass
                    
            else:
                # Unix/Linux: SIGTERM then SIGKILL
                try:
                    proc.terminate()  # SIGTERM
                    try:
                        proc.wait(timeout=2.0)
                        return True
                    except subprocess.TimeoutExpired:
                        pass
                except OSError:
                    pass
                
                # Force kill
                try:
                    proc.kill()  # SIGKILL
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            # Log error but don't crash
            print(f"Error during cancellation: {e}")
            return False
        finally:
            # Always clear the active process reference
            with self._lock:
                self._active_process = None
            
            # Clear CUDA cache after cancellation to free VRAM
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                # Note: If the *main* app process has initialized CUDA, Windows may still
                # show a small persistent VRAM reservation (CUDA context overhead).
                # That is expected and only fully disappears when the app exits.
                print("SUCCESS: CUDA cache cleared after cancellation")
            except Exception:
                # Silently ignore if CUDA not available
                pass
                
            # Clean up any zombie processes on Unix
            if platform.system() != "Windows":
                try:
                    import os
                    os.waitpid(-1, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass

    def is_canceled(self) -> bool:
        with self._lock:
            return self._canceled

    # ------------------------------------------------------------------ #
    # SeedVR2 runner
    # ------------------------------------------------------------------ #
    def run_seedvr2(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
        preview_only: bool = False,
    ) -> RunResult:
        """
        Run SeedVR2 CLI with given settings.

        settings should include all CLI-relevant keys aligned with inference_cli.py.
        """
        cli_path = self.base_dir / "SeedVR2" / "inference_cli.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"SeedVR2 CLI not found at {cli_path}")

        # Work with a per-run copy so fallback/guardrail mutations do not leak
        # across queued items or future runs.
        settings = settings.copy()

        input_path = normalize_path(settings.get("input_path"))
        if not input_path:
            raise ValueError("Input path is required.")

        output_format = settings.get("output_format") or "auto"
        format_for_cli = None if output_format == "auto" else output_format

        batch_mode = bool(settings.get("batch_mode"))
        # FIXED: Always honor global output_dir, even when Output Override is empty
        # The SeedVR2 CLI's generate_output_path() REQUIRES --output to use custom directory,
        # otherwise it defaults to writing next to input. We must always pass --output.
        effective_output_override = settings.get("output_override") or None
        if preview_only:
            # Force single-frame preview: load_cap=1, image path remains same
            settings = settings.copy()
            settings["load_cap"] = 1
            # The CLI supports batch_size=1 (4n+1 rule) and recommends it for 1-frame runs.
            # Our UI batch size slider starts at 5, so force a sane preview value here.
            try:
                settings["batch_size"] = 1
                settings["uniform_batch_size"] = False
            except Exception:
                pass

        # Respect CLI auto-detect: if format is None/auto, choose based on input type
        if preview_only:
            effective_output_format = "png"
        elif format_for_cli is None:
            itype = detect_input_type(input_path)
            effective_output_format = "png" if itype in ("image", "directory") else "mp4"
        else:
            effective_output_format = format_for_cli

        # FIXED: Predict output path AND ensure CLI receives --output for global output_dir
        # Even when user doesn't set Output Override, we need to pass global output_dir to CLI
        predicted_output: Optional[Path]
        cli_output_arg: Optional[str] = None  # What we pass to CLI via --output
        
        if effective_output_override:
            # User explicitly set Output Override - use it
            override_path = Path(normalize_path(effective_output_override))
            if override_path.suffix:
                # Explicit file path
                predicted_output = override_path
                cli_output_arg = str(override_path)
            else:
                # Directory override
                cli_output_arg = str(override_path)
                predicted_output = resolve_output_location(
                    input_path=input_path,
                    output_format=effective_output_format,
                    global_output_dir=cli_output_arg,
                    batch_mode=batch_mode,
                    png_padding=settings.get("png_padding"),
                    png_keep_basename=settings.get("png_keep_basename", False),
                    original_filename=settings.get("_original_filename"),  # Preserve user's filename
                )
        else:
            # FIXED: No override, but still pass global output_dir to CLI
            # This ensures files go to user's configured output folder, not next to input
            cli_output_arg = str(self.output_dir)  # CRITICAL: Pass to CLI even when no override
            predicted_output = resolve_output_location(
                input_path=input_path,
                output_format=effective_output_format,
                global_output_dir=cli_output_arg,
                batch_mode=batch_mode,
                png_padding=settings.get("png_padding"),
                png_keep_basename=settings.get("png_keep_basename", False),
                original_filename=settings.get("_original_filename"),  # Preserve user's filename
            )

        # IMPORTANT: If we predicted a FILE output (e.g., .mp4 or single-image .png),
        # pass that exact path to the CLI so the real output matches our collision-safe
        # prediction. Otherwise the CLI may overwrite the base name and the UI won't
        # find the expected *_0001 file.
        if predicted_output and predicted_output.suffix:
            cli_output_arg = str(predicted_output)

        # Keep app-side model selection compatible with CLI-side argparse choices.
        # This handles custom GGUF drops and stale presets before process launch.
        try:
            self._prepare_seedvr2_model_selection(settings, on_progress=on_progress)
        except Exception as e:
            warn_msg = f"[SeedVR2] Model selection preflight warning: {e}\n"
            print(warn_msg, end="", flush=True)
            if on_progress:
                on_progress(warn_msg)

        # FIXED: Pass cli_output_arg to command builder (not effective_output_override)
        cmd = self._build_seedvr2_cmd(cli_path, settings, format_for_cli, preview_only, output_override=cli_output_arg)

        # Route based on execution mode
        if self._active_mode == "in_app":
            return self._run_seedvr2_in_app(cli_path, cmd, predicted_output, settings, on_progress)
        else:
            result = self._run_seedvr2_subprocess(cli_path, cmd, predicted_output, settings, on_progress)

            # -----------------------------------------------------------------
            # CLI model-choice fallback (invalid --dit_model)
            # -----------------------------------------------------------------
            invalid_model, allowed_models = self._extract_seedvr2_invalid_model_error(result.log or "")
            if result.returncode == 2 and allowed_models:
                requested_model = str(settings.get("dit_model") or invalid_model or "").strip()
                fallback_model = self._pick_seedvr2_fallback_model(requested_model, allowed_models)
                if fallback_model and str(fallback_model) != requested_model:
                    retry_msg = (
                        "\n[SeedVR2] WARNING: Selected model is not accepted by this CLI build.\n"
                        f"[SeedVR2] RETRY: Auto-retrying with compatible model: {fallback_model}\n"
                    )
                    print(retry_msg, flush=True)
                    if on_progress:
                        on_progress(retry_msg)

                    retry_settings = settings.copy()
                    retry_settings["dit_model"] = fallback_model
                    fallback_path = self._resolve_seedvr2_model_file(fallback_model, retry_settings)
                    if fallback_path:
                        retry_settings["model_dir"] = str(fallback_path.parent)

                    retry_cmd = self._build_seedvr2_cmd(
                        cli_path,
                        retry_settings,
                        format_for_cli,
                        preview_only,
                        output_override=cli_output_arg,
                    )
                    retry_result = self._run_seedvr2_subprocess(
                        cli_path, retry_cmd, predicted_output, retry_settings, on_progress
                    )

                    combined_log = "\n".join(
                        [
                            "=== SeedVR2 attempt 1 (invalid --dit_model) ===",
                            result.log or "",
                            "",
                            f"=== SeedVR2 attempt 2 (auto-fallback: {fallback_model}) ===",
                            retry_result.log or "",
                        ]
                    )
                    return RunResult(retry_result.returncode, retry_result.output_path, combined_log)

            # -----------------------------------------------------------------
            # Windows vcvars/Build Tools failure auto-fallback
            # -----------------------------------------------------------------
            # If torch.compile is requested, we wrap the CLI with vcvarsall.bat.
            # If vcvars activation fails, `cmd /c` will exit quickly and Python never starts.
            # We surface a clear marker ([VCVARS_ERROR]) from the wrapper and then retry once
            # with torch.compile disabled so the run can still proceed.
            compile_requested = bool(settings.get("compile_dit") or settings.get("compile_vae"))
            vcvars_error = (
                platform.system() == "Windows"
                and compile_requested
                and result.returncode != 0
                and "[VCVARS_ERROR]" in (result.log or "")
            )
            if vcvars_error:
                retry_msg = (
                    "\n[SeedVR2] WARNING: VS Build Tools activation failed, so torch.compile cannot run.\n"
                    "[SeedVR2] RETRY: Auto-retrying with torch.compile disabled.\n"
                    "        To re-enable compile: install/repair Visual Studio Build Tools and the\n"
                    "        'Desktop development with C++' workload (MSVC toolset).\n"
                )
                print(retry_msg, flush=True)
                if on_progress:
                    on_progress(retry_msg)

                retry_settings = settings.copy()
                retry_settings["compile_dit"] = False
                retry_settings["compile_vae"] = False

                retry_cmd = self._build_seedvr2_cmd(
                    cli_path,
                    retry_settings,
                    format_for_cli,
                    preview_only,
                    output_override=cli_output_arg,
                )
                retry_result = self._run_seedvr2_subprocess(
                    cli_path, retry_cmd, predicted_output, retry_settings, on_progress
                )

                combined_log = "\n".join(
                    [
                        "=== SeedVR2 attempt 1 (compile requested, vcvars activation failed) ===",
                        result.log or "",
                        "",
                        "=== SeedVR2 attempt 2 (auto-fallback: compile disabled) ===",
                        retry_result.log or "",
                    ]
                )
                return RunResult(retry_result.returncode, retry_result.output_path, combined_log)

            # -----------------------------------------------------------------
            # Windows hard-crash auto-retry
            # -----------------------------------------------------------------
            # Some CUDA extensions (flash-attn / sage-attn / triton) can hard-crash
            # on Windows with 0xC0000005 (3221225477) depending on GPU + build.
            # We can't catch a Python exception because the process dies, so we
            # provide a single safe retry using PyTorch SDPA.
            windows_access_violation = (platform.system() == "Windows" and result.returncode == 3221225477)
            current_attn = str(settings.get("attention_mode") or "").strip().lower()
            if windows_access_violation and current_attn and current_attn != "sdpa":
                retry_msg = (
                    "\n[SeedVR2] WARNING: Detected Windows native crash (0xC0000005 / access violation).\n"
                    "[SeedVR2] RETRY: Auto-retrying with safer settings: attention_mode=sdpa"
                )
                if preview_only or int(settings.get("load_cap") or 0) == 1:
                    retry_msg += ", batch_size=1"
                retry_msg += "\n"

                print(retry_msg, flush=True)
                if on_progress:
                    on_progress(retry_msg)

                retry_settings = settings.copy()
                retry_settings["attention_mode"] = "sdpa"
                # Ensure compile stays off for the retry (compile-related crashes can also happen)
                retry_settings["compile_dit"] = False
                retry_settings["compile_vae"] = False

                # Safe batch size for tiny runs (incl. preview-only)
                if preview_only or int(retry_settings.get("load_cap") or 0) == 1:
                    retry_settings["batch_size"] = 1
                    retry_settings["uniform_batch_size"] = False

                retry_cmd = self._build_seedvr2_cmd(
                    cli_path,
                    retry_settings,
                    format_for_cli,
                    preview_only,
                    output_override=cli_output_arg,
                )
                retry_result = self._run_seedvr2_subprocess(
                    cli_path, retry_cmd, predicted_output, retry_settings, on_progress
                )

                combined_log = "\n".join(
                    [
                        "=== SeedVR2 attempt 1 (original settings) ===",
                        result.log or "",
                        "",
                        "=== SeedVR2 attempt 2 (auto-retry: attention_mode=sdpa) ===",
                        retry_result.log or "",
                    ]
                )
                return RunResult(retry_result.returncode, retry_result.output_path, combined_log)

            return result

    def _run_seedvr2_subprocess(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 CLI as a subprocess with proper error handling and logging.
        
        ENHANCED: Now prints all subprocess output to console (CMD) for user visibility,
        in addition to sending it to the on_progress callback for UI updates.
        """
        # Guard against overlapping SeedVR2 runs (e.g., preview + upscale clicked together).
        with self._lock:
            if self._seedvr2_inflight:
                busy_msg = (
                    "Another SeedVR2 run is already active. "
                    "Wait for it to finish or cancel it before starting a new run."
                )
                if on_progress:
                    on_progress(f"{busy_msg}\n")
                return RunResult(1, None, busy_msg)
            self._seedvr2_inflight = True
            self._canceled = False

        # Helper function to log to both console AND callback
        def log_output(message: str, force_console: bool = True):
            """Log message to console (always) and callback (if provided)."""
            if force_console:
                print(message, end='', flush=True)  # Print to CMD for user visibility
            if on_progress:
                on_progress(message)
        
        # Visual separator for user visibility in CMD
        print("\n[SeedVR2] Starting upscaling process...", flush=True)
        
        # Show key settings for user visibility
        input_path = settings.get("input_path", "Unknown")
        model_name = settings.get("dit_model", "Unknown")
        resolution = settings.get("resolution", "Unknown")
        print(f"[SeedVR2] Input: {input_path}", flush=True)
        print(f"[SeedVR2] Model: {model_name}", flush=True)
        print(f"[SeedVR2] Resolution: {resolution}p", flush=True)
        
        # Prepare environment (optionally inject MSVC toolchain env for torch.compile on Windows)
        env = os.environ.copy()
        # PyTorch renamed allocator env var; migrate legacy key to keep settings effective
        # and avoid warning spam in subprocess logs.
        legacy_alloc_conf = env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if legacy_alloc_conf and not env.get("PYTORCH_ALLOC_CONF"):
            env["PYTORCH_ALLOC_CONF"] = legacy_alloc_conf
            log_output("[SeedVR2] Migrated PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF\n")

        compile_requested = bool(settings.get("compile_dit") or settings.get("compile_vae"))
        if platform.system() == "Windows" and compile_requested:
            vcvars_path = self._find_vcvars()
            if not vcvars_path:
                warning_msg = (
                    "WARNING: VS Build Tools not found; disabling torch.compile for compatibility.\n"
                    "Install 'Desktop development with C++' workload from Visual Studio Installer for torch.compile support.\n"
                )
                self._log_lines.append(warning_msg.strip())
                log_output(warning_msg)
                settings["compile_dit"] = False
                settings["compile_vae"] = False
                cmd = self._strip_torch_compile_flags(cmd)
            else:
                vs_env, vs_detail = self._capture_vcvars_env(vcvars_path, arch="x64")
                if not vs_env:
                    vs_env, vs_detail = self._capture_vcvars_env(vcvars_path, arch="amd64")

                if not vs_env:
                    warning_msg = (
                        "WARNING: VS Build Tools detected but could not be activated; running without torch.compile.\n"
                        f"Path: {vcvars_path}\n"
                        f"Reason: {vs_detail}\n"
                    )
                    self._log_lines.append(warning_msg.strip())
                    log_output(warning_msg)
                    settings["compile_dit"] = False
                    settings["compile_vae"] = False
                    cmd = self._strip_torch_compile_flags(cmd)
                else:
                    env.update(vs_env)
                    log_output(f"INFO: Using VS Build Tools for torch.compile: {vcvars_path}\n   {vs_detail}\n")

        # Constrain visible GPU(s) before CLI startup to prevent stray contexts on GPU 0.
        cmd, gpu_isolation_note = self._enforce_seedvr2_gpu_visibility(cmd, settings, env)
        if gpu_isolation_note:
            log_output(gpu_isolation_note + "\n")

        # Log the (final) command being executed for debugging
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        log_output(f"[SeedVR2] Executing command:\n{cmd_str}\n")

        log_output("[SeedVR2] Starting subprocess (CLI will handle model loading)...\n")

        # Ensure our temp dir is used (even if vcvarsall overwrote TEMP/TMP in captured env)
        env["TEMP"] = str(self.temp_dir)
        env["TMP"] = str(self.temp_dir)
        env.setdefault("PYTHONWARNINGS", "ignore")
        # Windows consoles often default to legacy code pages (cp1252) which can crash
        # SeedVR2 when it prints emojis (UnicodeEncodeError). Force UTF-8 for the CLI.
        if platform.system() == "Windows":
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

        # Runtime ffmpeg diagnostics (printed to CMD + UI log) to trace codec issues.
        try:
            ffmpeg_bin = shutil.which("ffmpeg", path=env.get("PATH"))
            ffprobe_bin = shutil.which("ffprobe", path=env.get("PATH"))
            if ffmpeg_bin:
                log_output(f"[SeedVR2] ffmpeg resolved to: {ffmpeg_bin}\n")
            else:
                log_output("[SeedVR2] ffmpeg not found in subprocess PATH.\n")
            if ffprobe_bin:
                log_output(f"[SeedVR2] ffprobe resolved to: {ffprobe_bin}\n")
            else:
                log_output("[SeedVR2] ffprobe not found in subprocess PATH.\n")

            if ffmpeg_bin:
                enc_probe = subprocess.run(
                    [ffmpeg_bin, "-hide_banner", "-encoders"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    timeout=12,
                )
                enc_text = f"{enc_probe.stdout or ''}\n{enc_probe.stderr or ''}".lower()
                has_x264 = "libx264" in enc_text
                has_x265 = "libx265" in enc_text
                log_output(
                    f"[SeedVR2] ffmpeg encoder support: libx264={has_x264}, libx265={has_x265}\n"
                )
        except Exception as ff_diag_exc:
            log_output(f"[SeedVR2] ffmpeg probe warning: {ff_diag_exc}\n")

        # Ensure base dirs exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid  # type: ignore[arg-type]

        proc: Optional[subprocess.Popen] = None
        log_lines: List[str] = []
        returncode = -1
        start_time = time.time()  # Track execution time

        try:
            with self._lock:
                if self._canceled:
                    msg = "[SeedVR2] Run canceled before subprocess launch.\n"
                    log_lines.append(msg.strip())
                    log_output(msg)
                    return RunResult(1, None, "\n".join(log_lines))

            log_output("[SeedVR2] Launching CLI process...\n")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # line-buffered text stream for lower UI log latency
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=self.base_dir,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )

            with self._lock:
                self._active_process = proc
                self._canceled = False

            log_output("[SeedVR2] Process started, monitoring output...\n")

            assert proc.stdout is not None

            # Read output line by line with timeout handling
            while True:
                # Check if process is still running and not canceled
                with self._lock:
                    if self._active_process is None:
                        log_output("[SeedVR2] Process canceled by user\n")
                        break

                try:
                    # Use timeout to avoid blocking indefinitely
                    if platform.system() == "Windows":
                        # Windows doesn't have select for file handles, use simpler approach
                        line = proc.stdout.readline()
                        if not line:
                            break
                    else:
                        # Unix systems can use select
                        import select
                        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                        if ready:
                            line = proc.stdout.readline()
                            if not line:
                                break
                        else:
                            continue

                    line = line.rstrip()
                    if line:  # Only add non-empty lines
                        log_lines.append(line)
                        # Print to console for user visibility AND send to callback
                        print(line, flush=True)  # Always print to CMD
                        if on_progress:
                            on_progress(line + "\n")

                except Exception as e:
                    log_output(f"[SeedVR2] Error reading subprocess output: {e}\n")
                    break

            # Wait for process to complete
            returncode = proc.wait()

            # Log completion status with clear visual indicator
            if returncode == 0:
                print("[SeedVR2] Process completed successfully", flush=True)
                log_output(f"[SeedVR2] Process completed successfully (code: {returncode})\n")
            else:
                print(f"[SeedVR2] Process exited with error code: {returncode}", flush=True)
                log_output(f"[SeedVR2] Process exited with error code: {returncode}\n")
                
                # Show last few log lines as error context
                if log_lines:
                    print("\n[SeedVR2] Last 15 lines of output:", flush=True)
                    for line in log_lines[-15:]:
                        print(f"  {line}", flush=True)
                    log_output("[SeedVR2] Last 15 lines of output:\n")
                    for line in log_lines[-15:]:
                        log_output(f"  {line}\n")
                else:
                    print("\n[SeedVR2] No output captured from subprocess!", flush=True)
                    print("Possible causes:", flush=True)
                    print("  1. Python/CUDA initialization failed", flush=True)
                    print("  2. Missing dependencies or models", flush=True)
                    print("  3. Invalid input file format", flush=True)
                    print("  4. Insufficient VRAM or memory", flush=True)
                    log_output("[SeedVR2] No output captured from subprocess - check CMD for details\n")
            
            # Clear CUDA cache after subprocess completes
            # This ensures VRAM is freed even if the subprocess didn't clean up properly
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                if returncode == 0:
                    log_output("[SeedVR2] CUDA cache cleared\n")
            except Exception:
                # Silently ignore if CUDA not available or clear fails
                pass

        except FileNotFoundError as e:
            error_msg = f"CLI script not found: {e}"
            log_lines.append(error_msg)
            print(f"[SeedVR2] ERROR: FILE NOT FOUND", flush=True)
            print(f"[SeedVR2] CLI Path: {cli_path}", flush=True)
            print(f"[SeedVR2] Error: {e}", flush=True)
            log_output(f"[SeedVR2] {error_msg}\n")
            returncode = 1
        except Exception as e:
            error_msg = f"Failed to execute subprocess: {e}"
            log_lines.append(error_msg)
            print(f"[SeedVR2] ERROR: SUBPROCESS EXECUTION FAILED", flush=True)
            print(f"[SeedVR2] Error Type: {type(e).__name__}", flush=True)
            print(f"[SeedVR2] Error Message: {e}", flush=True)
            log_output(f"[SeedVR2] {error_msg}\n")
            # Print full traceback to CMD for debugging
            import traceback
            traceback_str = traceback.format_exc()
            print("\n[SeedVR2] FULL TRACEBACK:", flush=True)
            print(traceback_str, flush=True)
            log_lines.append(traceback_str)
            returncode = 1
        finally:
            with self._lock:
                self._active_process = None
                self._seedvr2_inflight = False
            
            # Also clear CUDA cache on error/cancellation
            try:
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
            except Exception:
                pass

        # Determine output path
        output_path = None
        if predicted_output and Path(predicted_output).exists():
            output_path = str(predicted_output)
            log_output(f"[SeedVR2] Output file created: {output_path}\n")
        elif predicted_output:
            log_output(f"[SeedVR2] WARNING: Expected output not found: {predicted_output}\n")

        # Handle cancellation case
        if self._canceled and predicted_output and Path(predicted_output).exists():
            log_lines.append("Run canceled; partial output preserved.")
            output_path = str(predicted_output)

        # Emit metadata for ALL runs (success, failure, cancellation) if enabled
        # This provides crucial telemetry for troubleshooting failed/cancelled runs
        # Check both global telemetry AND per-run metadata settings
        should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
        if should_emit_metadata:
            try:
                # Determine metadata target: output path if exists, otherwise output_dir
                metadata_target = Path(output_path) if output_path else self.output_dir
                
                # Build comprehensive metadata including failure/cancellation info
                metadata_payload = {
                    "returncode": returncode,
                    "output": output_path,
                    "args": settings,
                    "command": cmd_str,
                    "status": "success" if returncode == 0 else ("cancelled" if self._canceled else "failed"),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                # Add failure-specific context
                if returncode != 0:
                    metadata_payload["error_logs"] = log_lines[-50:]  # Last 50 log lines for debugging
                    if self._canceled:
                        metadata_payload["cancellation_reason"] = "User cancelled processing"
                
                emit_metadata(metadata_target, metadata_payload)
            except Exception as e:
                if on_progress:
                    on_progress(f"Warning: Failed to emit metadata: {e}\n")

        self._last_model_id = settings.get("dit_model", self._last_model_id)

        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log command to executed_commands folder
        try:
            self._command_logger.log_command(
                tab_name="seedvr2",
                command=cmd,
                settings=settings,
                returncode=returncode,
                output_path=output_path,
                error_logs=log_lines[-50:] if returncode != 0 else None,  # Last 50 lines on error
                execution_time=execution_time,
                additional_info={
                    "mode": "subprocess",
                    "cancelled": self._canceled,
                    "predicted_output": str(predicted_output) if predicted_output else None
                }
            )
            log_output(f"[SeedVR2] INFO: Command logged to executed_commands folder\n")
        except Exception as e:
            log_output(f"[SeedVR2] WARNING: Failed to log command: {e}\n")

        # Combine all log lines
        full_log = "\n".join(log_lines)

        return RunResult(returncode, output_path, full_log)

    def _run_seedvr2_in_app(self, cli_path: Path, cmd: List[str], predicted_output: Optional[Path], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> RunResult:
        """
        Execute SeedVR2 in-app mode (EXPERIMENTAL - NOT RECOMMENDED).

        WARNING: CRITICAL LIMITATION: SeedVR2 CLI ARCHITECTURE PREVENTS MODEL PERSISTENCE
        ============================================================================
        The SeedVR2 CLI is designed to load models, process, then exit. Even when run
        via runpy (in-process), the CLI code does NOT maintain model instances between
        runs. Each invocation reloads models from disk.
        
        RESULT: In-app mode provides **ZERO SPEED BENEFIT** for SeedVR2 compared to subprocess.
        
        CURRENT IMPLEMENTATION STATUS:
        - WARNING: PARTIALLY IMPLEMENTED: Runs CLI via runpy but does NOT implement persistent model caching
        - ERROR: Models reload each run (IDENTICAL to subprocess mode - no performance gain)
        - ERROR: Cannot cancel mid-run (no subprocess to kill)
        - ERROR: VS Build Tools wrapper not applied (torch.compile may fail on Windows)
        - WARNING: Memory leaks possible without subprocess isolation
        - WARNING: ModelManager tracking exists but cannot force CLI to keep models loaded
        
        WHY THIS EXISTS:
        - Framework placeholder for future GAN/RIFE in-app optimization
        - Demonstrates in-app execution pattern for other model types
        - SeedVR2 would need CLI refactoring to support true model persistence
        
        RECOMMENDATION FOR SEEDVR2:
        DO NOT USE IN-APP MODE. It provides no benefits and loses cancellation.
        USE SUBPROCESS MODE. Same speed, full cancellation, better isolation.
        
        FUTURE WORK (requires SeedVR2 CLI changes):
        - Refactor CLI to expose model loading/inference as separate functions
        - Implement persistent model caching in ModelManager
        - Add intelligent model swapping when user changes models
        - Enable proper cancellation via threading interrupts
        """
        if on_progress:
            on_progress("WARNING: IN-APP MODE ACTIVE (NOT RECOMMENDED FOR SEEDVR2)\n")
            on_progress("CRITICAL: SeedVR2 CLI reloads models each run - NO SPEED BENEFIT over subprocess\n")
            on_progress("LIMITATION: Cannot cancel mid-run (no subprocess to kill)\n")
            on_progress("RECOMMENDATION: Use subprocess mode for SeedVR2 (same speed + cancellation)\n")
        
        # Check for compile + Windows - attempt vcvars environment setup
        if platform.system() == "Windows" and (settings.get("compile_dit") or settings.get("compile_vae")):
            from .health import is_vs_build_tools_available
            
            # Check if vcvars environment is already active
            vcvars_active = os.environ.get("VSCMD_ARG_TGT_ARCH") is not None
            
            if not vcvars_active:
                if on_progress:
                    on_progress("WARNING: torch.compile requested but vcvars environment not active.\n")
                
                # Try to find and source vcvars
                vcvars_path = self._find_vcvars()
                
                if vcvars_path and vcvars_path.exists():
                    if on_progress:
                        on_progress(f"INFO: Attempting to activate VS Build Tools: {vcvars_path}\n")
                    
                    # In-app mode limitation: We cannot directly modify the current process environment
                    # after Python has started. The vcvars.bat sets up C++ compiler paths, but these
                    # need to be active BEFORE Python imports torch.
                    if on_progress:
                        on_progress("WARNING: IN-APP LIMITATION: Cannot activate vcvars after Python started.\n")
                        on_progress("WORKAROUND: Activate vcvars BEFORE starting this app, or use subprocess mode.\n")
                        on_progress("INFO: Auto-disabling torch.compile to prevent cryptic compilation errors.\n")
                    
                    settings["compile_dit"] = False
                    settings["compile_vae"] = False
                else:
                    if on_progress:
                        on_progress("ERROR: VS Build Tools not found. torch.compile disabled.\n")
                        on_progress("INFO: Install 'Desktop development with C++' workload from Visual Studio Installer.\n")
                    
                    settings["compile_dit"] = False
                    settings["compile_vae"] = False
            else:
                if on_progress:
                    on_progress("INFO: VS Build Tools environment active - torch.compile should work.\n")

        log_lines: List[str] = []
        returncode = -1
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env["TEMP"] = str(self.temp_dir)
            env["TMP"] = str(self.temp_dir)
            env.setdefault("PYTHONWARNINGS", "ignore")
            
            # Ensure directories exist
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            if on_progress:
                on_progress("Loading model and initializing pipeline...\n")
            
            # Track model loading via ModelManager
            model_manager = self._model_manager
            dit_model = settings.get("dit_model", "")
            model_id = model_manager._generate_model_id(ModelType.SEEDVR2, dit_model, **settings)
            
            # Update current model tracking
            old_model_id = model_manager.current_model_id
            if old_model_id and old_model_id != model_id:
                if on_progress:
                    on_progress(f"Model changed ({old_model_id} -> {model_id}), clearing cache...\n")
                try:
                    from .gpu_utils import clear_cuda_cache
                    clear_cuda_cache()
                except Exception:
                    pass
            
            model_manager.current_model_id = model_id
            
            # Run CLI directly via runpy (stays in same process, models persist)
            # Build sys.argv from cmd (skip python executable)
            import sys
            old_argv = sys.argv.copy()
            
            try:
                # cmd format: [python_path, cli_path, ...args]
                sys.argv = [str(cli_path)] + cmd[2:]  # Skip python path and cli path
                
                if on_progress:
                    on_progress(f"Executing: {' '.join(sys.argv)}\n")
                
                # Capture output
                log_buffer = io.StringIO()
                with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                    try:
                        # Run the CLI script directly in this process
                        runpy.run_path(str(cli_path), run_name="__main__")
                        returncode = 0
                    except SystemExit as e:
                        returncode = e.code if isinstance(e.code, int) else (1 if e.code else 0)
                    except Exception as e:
                        log_lines.append(f"ERROR: In-app execution error: {str(e)}")
                        returncode = 1
                
                log_lines.append(log_buffer.getvalue())
                
            finally:
                # Restore sys.argv
                sys.argv = old_argv
            
            if on_progress:
                on_progress(f"Execution completed with code {returncode}\n")
            
            # Check for output
            output_path = None
            if predicted_output and predicted_output.exists():
                output_path = str(predicted_output)
                if on_progress:
                    on_progress(f"Output created: {output_path}\n")
            
            # Emit metadata for ALL runs (success, failure, cancellation) if enabled
            should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
            if should_emit_metadata:
                try:
                    # Determine metadata target
                    metadata_target = Path(output_path) if output_path else self.output_dir
                    
                    metadata_payload = {
                        "returncode": returncode,
                        "output": output_path,
                        "args": settings,
                        "mode": "in_app",
                        "command": " ".join(cmd),
                        "status": "success" if returncode == 0 else "failed",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    
                    # Add failure context
                    if returncode != 0:
                        metadata_payload["error_logs"] = log_lines[-50:]  # Last 50 lines for debugging
                    
                    emit_metadata(metadata_target, metadata_payload)
                except Exception as e:
                    if on_progress:
                        on_progress(f"Warning: Failed to emit metadata: {e}\n")
            
            return RunResult(returncode, output_path, "\n".join(log_lines))
            
        except Exception as e:
            error_msg = f"In-app execution failed: {str(e)}"
            log_lines.append(f"ERROR: {error_msg}")
            if on_progress:
                on_progress(f"ERROR: {error_msg}\\n")
            return RunResult(1, None, "\n".join(log_lines))

    def ensure_seedvr2_model_loaded(self, settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """
        Ensure the required SeedVR2 model is loaded, loading it if necessary.
        
        Uses ModelManager for intelligent caching and delayed loading.
        In subprocess mode, the CLI will handle actual loading.
        In in-app mode, the ModelManager would cache the loaded model.
        """
        dit_model = settings.get("dit_model", "")
        if not dit_model:
            return False

        # In subprocess mode, CLI handles loading - but we still track state
        if self._active_mode == "subprocess":
            if on_progress:
                on_progress(f"Model '{dit_model}' will be loaded by subprocess CLI\n")
            
            # Update model manager state for tracking (even though CLI does the work)
            model_id = self._model_manager._generate_model_id(
                ModelType.SEEDVR2,
                dit_model,
                **settings
            )
            self._model_manager.current_model_id = model_id
            
            return True
        
        # In in-app mode, use ModelManager for actual caching
        # (This would require implementing in-app model loading, which is future work)
        # For now, always defer to CLI
        if on_progress:
            on_progress(f"Model '{dit_model}' loading delegated to CLI\n")
        
        return True

    # ------------------------------------------------------------------ #
    # Command builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_compile_dynamic_setting(value: Any, default: str = "none") -> str:
        """Normalize compile_dynamic to one of: 'none', 'false', 'true'."""
        default_text = str(default or "none").strip().lower()
        if default_text in {"", "auto", "default"}:
            default_text = "none"
        if default_text not in {"none", "false", "true"}:
            default_text = "none"

        if value is None:
            return default_text
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return "true" if bool(value) else "false"

        text = str(value).strip().lower()
        if text in {"", "none", "auto", "default"}:
            return "none"
        if text in {"true", "1", "yes", "on"}:
            return "true"
        if text in {"false", "0", "no", "off"}:
            return "false"
        return default_text

    @staticmethod
    def _strip_torch_compile_flags(command: List[str]) -> List[str]:
        """
        Remove torch.compile-related CLI args from a command list.

        IMPORTANT:
        - Some flags are boolean (no value) and others take a value.
        - We must NOT skip the next token for boolean flags like --compile_dit,
          otherwise we can accidentally delete unrelated args.
        """
        flags_no_value = {
            "--compile_dit",
            "--compile_vae",
            "--compile_fullgraph",
        }
        flags_with_value = {
            "--compile_backend",
            "--compile_mode",
            "--compile_dynamo_cache_size_limit",
            "--compile_dynamo_recompile_limit",
        }

        out: List[str] = []
        i = 0
        while i < len(command):
            token = command[i]
            if token in flags_no_value:
                i += 1
                continue
            if token == "--compile_dynamic":
                if i + 1 < len(command) and str(command[i + 1]).strip().lower() in {"none", "false", "true"}:
                    i += 2
                else:
                    i += 1
                continue
            if token in flags_with_value:
                i += 2  # Skip flag + its value
                continue
            out.append(token)
            i += 1
        return out

    @staticmethod
    def _normalize_cuda_token(token: Any) -> str:
        text = str(token or "").strip().lower()
        if text.startswith("cuda:"):
            text = text.split(":", 1)[1].strip()
        return text

    @classmethod
    def _split_cuda_spec(cls, value: Any) -> List[str]:
        raw = str(value or "").strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        devices: List[str] = []
        for part in parts:
            normalized = cls._normalize_cuda_token(part)
            if normalized.isdigit() and normalized not in devices:
                devices.append(normalized)
        return devices

    @staticmethod
    def _find_flag_index(command: List[str], flag: str) -> Optional[int]:
        for i, token in enumerate(command):
            if token == flag:
                return i
        return None

    @classmethod
    def _get_flag_value(cls, command: List[str], flag: str) -> Optional[str]:
        idx = cls._find_flag_index(command, flag)
        if idx is None or idx + 1 >= len(command):
            return None
        return str(command[idx + 1])

    @classmethod
    def _set_flag_value(cls, command: List[str], flag: str, value: str) -> List[str]:
        updated = list(command)
        idx = cls._find_flag_index(updated, flag)
        if idx is None:
            return updated
        if idx + 1 < len(updated):
            updated[idx + 1] = str(value)
        else:
            updated.append(str(value))
        return updated

    def _enforce_seedvr2_gpu_visibility(
        self,
        command: List[str],
        settings: Dict[str, Any],
        env: Dict[str, str],
    ) -> Tuple[List[str], Optional[str]]:
        """
        Ensure SeedVR2 subprocess can only see selected GPU(s).

        Why this exists:
        - SeedVR2 pre-parses `--cuda_device` before heavy imports.
        - If CUDA_VISIBLE_DEVICES is not pre-set, it imports torch to validate IDs.
        - That early validation can create a context on physical GPU 0.

        We set CUDA_VISIBLE_DEVICES in the parent launcher and remap CLI device IDs
        to local indices (0..N-1) inside the constrained view.
        """
        updated_cmd = list(command)
        selected_raw = str(settings.get("cuda_device", "") or "").strip()
        global_raw = str(env.get("SECOURSES_GLOBAL_GPU_DEVICE", "") or "").strip().lower()

        if global_raw == "cpu" or selected_raw.lower() in {"cpu", "none"}:
            env["CUDA_VISIBLE_DEVICES"] = ""
            return updated_cmd, "[SeedVR2] GPU isolation: CPU mode (CUDA_VISIBLE_DEVICES cleared)"

        # Prefer explicit SeedVR2 selection, fallback to global runtime selection.
        visible_devices = self._split_cuda_spec(selected_raw or global_raw)
        if not visible_devices:
            return updated_cmd, None

        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        mapping = {src: str(idx) for idx, src in enumerate(visible_devices)}

        cmd_cuda_value = self._get_flag_value(updated_cmd, "--cuda_device")
        requested_devices = self._split_cuda_spec(cmd_cuda_value if cmd_cuda_value is not None else selected_raw)
        if requested_devices:
            remapped_devices = [mapping.get(device_id, "0") for device_id in requested_devices]
            updated_cmd = self._set_flag_value(updated_cmd, "--cuda_device", ",".join(remapped_devices))

        for flag in ("--dit_offload_device", "--vae_offload_device", "--tensor_offload_device"):
            current = self._get_flag_value(updated_cmd, flag)
            if current is None:
                continue
            current_str = str(current).strip()
            current_lower = current_str.lower()
            if current_lower in {"", "cpu", "none"}:
                continue

            had_cuda_prefix = current_lower.startswith("cuda:")
            current_id = self._normalize_cuda_token(current_str)
            if not current_id.isdigit():
                continue

            remapped = mapping.get(current_id, "0")
            remapped_value = f"cuda:{remapped}" if had_cuda_prefix else remapped
            updated_cmd = self._set_flag_value(updated_cmd, flag, remapped_value)

        remap_pairs = ", ".join(f"{src}->{dst}" for src, dst in mapping.items())
        return (
            updated_cmd,
            f"[SeedVR2] GPU isolation: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} (local mapping {remap_pairs})",
        )

    def _capture_vcvars_env(self, vcvars_path: Path, arch: str = "x64", timeout: int = 25) -> Tuple[Optional[Dict[str, str]], str]:
        """
        Run vcvarsall.bat and capture the resulting environment via `set`.

        This is more robust than wrapping the whole Python command in `cmd /c ...`:
        - avoids complex quoting issues
        - keeps subprocess stdout/stderr capture reliable

        Returns:
            (env, detail)
            - env: dict of env vars to apply, or None on failure
            - detail: short human-readable status / error context
        """
        if platform.system() != "Windows":
            return None, "Not Windows"

        cmd = ["cmd", "/d", "/s", "/c", "call", str(vcvars_path), arch, "&&", "set"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except subprocess.TimeoutExpired:
            return None, f"vcvars activation timed out ({timeout}s)"
        except Exception as e:
            return None, f"vcvars activation failed: {e}"

        if result.returncode != 0:
            stdout_tail = "\n".join((result.stdout or "").splitlines()[-25:])
            stderr_tail = "\n".join((result.stderr or "").splitlines()[-25:])
            detail_lines = [f"vcvarsall.bat returned code {result.returncode}"]
            if stdout_tail.strip():
                detail_lines.append("--- stdout (tail) ---\n" + stdout_tail)
            if stderr_tail.strip():
                detail_lines.append("--- stderr (tail) ---\n" + stderr_tail)
            return None, "\n".join(detail_lines)

        env_updates: Dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if not key or key.startswith("="):
                # Skip special drive vars like '=C:' emitted by `set`
                continue
            env_updates[key] = value

        cl = shutil.which("cl.exe", path=env_updates.get("PATH"))
        if not cl:
            return None, "cl.exe not found after vcvarsall activation (missing MSVC workload?)"

        return env_updates, f"cl.exe: {cl}"

    def _find_vcvars(self) -> Optional[Path]:
        """
        Try to locate vcvarsall.bat using multiple detection methods.
        
        Uses the most robust approach:
        1. Check VSINSTALLDIR environment variable
        2. Use vswhere.exe (official VS installer locator)
        3. Fall back to hardcoded common paths
        
        This ensures maximum compatibility across different VS installations.
        """
        # Method 1: Check environment variable (fastest)
        vs_install_dir = os.environ.get("VSINSTALLDIR")
        if vs_install_dir:
            env_candidate = Path(vs_install_dir) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if env_candidate.exists():
                return env_candidate
        
        # Method 2: Use vswhere.exe (most reliable)
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if vswhere.exists():
            try:
                # Query vswhere for latest Visual Studio installation
                result = subprocess.run(
                    [str(vswhere), "-latest", "-property", "installationPath"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    install_path = Path(result.stdout.strip())
                    vcvars = install_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                    if vcvars.exists():
                        return vcvars
            except (subprocess.TimeoutExpired, Exception):
                # vswhere failed, continue to fallback
                pass
        
        # Method 3: Check common installation paths
        candidates = [
            # BuildTools (most common for CI/CD)
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            # Community Edition
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"),
            # Professional Edition
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"),
            # Enterprise Edition
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"),
            # VS 2019 fallback
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                # Quick validation that it's actually a vcvarsall.bat file
                try:
                    with open(candidate, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(200)
                        # Verify it looks like a vcvarsall.bat file
                        if 'vcvarsall' in content.lower() or '@echo off' in content:
                            return candidate
                except Exception:
                    continue

        return None

    def _get_python_executable(self) -> str:
        """
        Get the correct Python executable path.
        
        Priority:
        1. Venv python from launcher (venv/Scripts/python.exe or venv/bin/python)
        2. sys.executable as fallback
        
        This ensures subprocess runs use the correct environment with all dependencies.
        """
        # Try to find venv python relative to base_dir
        if platform.system() == "Windows":
            venv_python = self.base_dir / "venv" / "Scripts" / "python.exe"
        else:
            venv_python = self.base_dir / "venv" / "bin" / "python"
        
        if venv_python.exists():
            return str(venv_python)
        
        # Fallback to sys.executable if venv not found
        return sys.executable

    @staticmethod
    def _seedvr2_supported_weight_exts() -> set[str]:
        return {".gguf", ".safetensors"}

    def _seedvr2_cli_discovery_dirs(self) -> List[Path]:
        """
        Mirror SeedVR2 CLI discovery fallbacks when ComfyUI `folder_paths`
        is not present (the default for this app packaging).
        """
        return [
            self.base_dir / "models" / "SEEDVR2",
            self.base_dir / "models" / "seedvr2",
            self.base_dir / "models" / "SeedVR2",
        ]

    def _seedvr2_runtime_model_dirs(self, settings: Dict[str, Any]) -> List[Path]:
        dirs: List[Path] = []
        configured = normalize_path(settings.get("model_dir"))
        if configured:
            dirs.append(Path(configured))
        dirs.append(self.base_dir / "SeedVR2" / "models")
        dirs.extend(self._seedvr2_cli_discovery_dirs())

        # Preserve order but dedupe equivalent paths.
        seen: set[str] = set()
        unique_dirs: List[Path] = []
        for d in dirs:
            try:
                key = str(d.resolve()).lower()
            except Exception:
                key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(d)
        return unique_dirs

    def _resolve_seedvr2_model_file(self, model_name: str, settings: Dict[str, Any]) -> Optional[Path]:
        name = str(model_name or "").strip()
        if not name:
            return None
        for model_dir in self._seedvr2_runtime_model_dirs(settings):
            try:
                candidate = model_dir / name
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None

    def _discover_seedvr2_cli_models(self) -> List[str]:
        """
        Discover models likely accepted by SeedVR2 argparse:
        - Built-in registry entries in SeedVR2/src/utils/model_registry.py
        - Files visible under CLI discovery directories (models/SEEDVR2, etc.)
        """
        names: set[str] = set()

        registry_path = self.base_dir / "SeedVR2" / "src" / "utils" / "model_registry.py"
        if registry_path.exists():
            try:
                text = registry_path.read_text(encoding="utf-8", errors="ignore")
                matches = re.findall(r'^\s*"([^"]+)"\s*:\s*ModelInfo\(', text, flags=re.MULTILINE)
                for model_name in matches:
                    suffix = Path(model_name).suffix.lower()
                    if suffix in self._seedvr2_supported_weight_exts():
                        names.add(model_name)
            except Exception:
                pass

        for model_dir in self._seedvr2_cli_discovery_dirs():
            try:
                if not model_dir.exists():
                    continue
                for f in model_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in self._seedvr2_supported_weight_exts():
                        names.add(f.name)
            except Exception:
                continue

        return sorted(names)

    def _ensure_seedvr2_model_visible_to_cli(
        self,
        model_name: str,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Ensure custom models selected from non-CLI-scanned folders are visible to
        CLI argparse discovery by linking/copying into models/SEEDVR2.
        """
        selected = str(model_name or "").strip()
        if not selected:
            return False

        target_dir = self.base_dir / "models" / "SEEDVR2"
        target_path = target_dir / selected
        if target_path.exists():
            return False

        source_path = self._resolve_seedvr2_model_file(selected, settings)
        if not source_path:
            return False

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.link(str(source_path), str(target_path))
                method = "hardlink"
            except Exception:
                shutil.copy2(source_path, target_path)
                method = "copy"

            msg = f"[SeedVR2] Synced selected model for CLI discovery via {method}: {target_path}\n"
            print(msg, end="", flush=True)
            if on_progress:
                on_progress(msg)
            return True
        except Exception as e:
            msg = f"[SeedVR2] Warning: failed to sync model into CLI discovery path: {e}\n"
            print(msg, end="", flush=True)
            if on_progress:
                on_progress(msg)
            return False

    @staticmethod
    def _seedvr2_model_traits(model_name: str) -> Dict[str, str]:
        text = str(model_name or "").strip().lower()
        if "_7b_" in text:
            size = "7b"
        elif "_3b_" in text:
            size = "3b"
        else:
            size = ""
        variant = "sharp" if "_sharp" in text else "standard"
        extension = Path(text).suffix.lower()
        precision = ""
        if "-q8_0" in text:
            precision = "q8"
        elif "-q4_k_m" in text:
            precision = "q4"
        elif "_fp16" in text:
            precision = "fp16"
        elif "_fp8_" in text:
            precision = "fp8"
        return {
            "size": size,
            "variant": variant,
            "extension": extension,
            "precision": precision,
        }

    @classmethod
    def _pick_seedvr2_fallback_model(cls, requested_model: str, available_models: List[str]) -> Optional[str]:
        if not available_models:
            return None

        requested = str(requested_model or "").strip()
        available_by_lower = {str(name).lower(): str(name) for name in available_models if str(name).strip()}
        if requested and requested.lower() in available_by_lower:
            return available_by_lower[requested.lower()]

        # Explicit high-value aliases for common GGUF additions.
        alias_fallbacks: Dict[str, List[str]] = {
            "seedvr2_ema_7b_sharp-q8_0.gguf": [
                "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
                "seedvr2_ema_7b_sharp_fp16.safetensors",
                "seedvr2_ema_7b_fp16.safetensors",
            ],
            "seedvr2_ema_7b-q8_0.gguf": [
                "seedvr2_ema_7b-Q4_K_M.gguf",
                "seedvr2_ema_7b_fp16.safetensors",
            ],
            "seedvr2_ema_3b-q8_0.gguf": [
                "seedvr2_ema_3b-Q4_K_M.gguf",
                "seedvr2_ema_3b_fp16.safetensors",
            ],
        }
        for candidate in alias_fallbacks.get(requested.lower(), []):
            mapped = available_by_lower.get(candidate.lower())
            if mapped:
                return mapped

        req_traits = cls._seedvr2_model_traits(requested)
        best_name = str(available_models[0])
        best_score = -10**9
        for candidate in available_models:
            cand_name = str(candidate)
            cand_traits = cls._seedvr2_model_traits(cand_name)
            score = 0

            if req_traits["size"] and cand_traits["size"] == req_traits["size"]:
                score += 120
            if cand_traits["variant"] == req_traits["variant"]:
                score += 40
            if req_traits["extension"] and cand_traits["extension"] == req_traits["extension"]:
                score += 20

            if req_traits["precision"] and cand_traits["precision"] == req_traits["precision"]:
                score += 45
            elif req_traits["precision"] == "q8" and cand_traits["precision"] == "q4":
                score += 38
            elif req_traits["precision"] == "q4" and cand_traits["precision"] == "q8":
                score += 20
            elif req_traits["precision"] in {"fp16", "fp8"} and cand_traits["precision"] in {"fp16", "fp8"}:
                score += 18

            if cand_name.lower().endswith(".safetensors"):
                score += 5

            if score > best_score:
                best_score = score
                best_name = cand_name

        return best_name

    @staticmethod
    def _extract_seedvr2_invalid_model_error(log_text: str) -> Tuple[Optional[str], List[str]]:
        text = str(log_text or "")
        match = re.search(
            r"--dit_model:\s*invalid choice:\s*'([^']+)'.*?\(choose from ([^)]+)\)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None, []

        invalid_model = str(match.group(1) or "").strip()
        raw_choices = str(match.group(2) or "")
        choices = [choice.strip() for choice in re.findall(r"'([^']+)'", raw_choices) if choice.strip()]
        return invalid_model, choices

    def _prepare_seedvr2_model_selection(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Normalize selected model + model_dir for compatibility with CLI constraints.
        Returns True when settings/model visibility changed.
        """
        selected_model = str(settings.get("dit_model") or "").strip()
        if not selected_model:
            return False

        changed = False

        # Prefer the directory that already contains the selected file.
        selected_model_path = self._resolve_seedvr2_model_file(selected_model, settings)
        if selected_model_path:
            resolved_dir = str(selected_model_path.parent)
            current_dir = str(settings.get("model_dir") or "").strip()
            if resolved_dir and resolved_dir != current_dir:
                settings["model_dir"] = resolved_dir
                changed = True
                msg = f"[SeedVR2] Using detected model directory: {resolved_dir}\n"
                print(msg, end="", flush=True)
                if on_progress:
                    on_progress(msg)

        # Ensure model is visible to CLI argparse discovery.
        if self._ensure_seedvr2_model_visible_to_cli(selected_model, settings, on_progress=on_progress):
            changed = True

        available_models = self._discover_seedvr2_cli_models()
        available_by_lower = {m.lower(): m for m in available_models}
        canonical_match = available_by_lower.get(selected_model.lower())
        if canonical_match:
            if canonical_match != selected_model:
                settings["dit_model"] = canonical_match
                changed = True
            return changed

        fallback_model = self._pick_seedvr2_fallback_model(selected_model, available_models)
        if fallback_model and fallback_model != selected_model:
            settings["dit_model"] = fallback_model
            changed = True
            msg = (
                f"[SeedVR2] Model '{selected_model}' is not recognized by CLI. "
                f"Auto-selected compatible fallback: '{fallback_model}'.\n"
            )
            print(msg, end="", flush=True)
            if on_progress:
                on_progress(msg)

            fallback_path = self._resolve_seedvr2_model_file(fallback_model, settings)
            if fallback_path:
                resolved_dir = str(fallback_path.parent)
                current_dir = str(settings.get("model_dir") or "").strip()
                if resolved_dir and resolved_dir != current_dir:
                    settings["model_dir"] = resolved_dir
                    changed = True

        return changed

    def _build_seedvr2_cmd(
        self,
        cli_path: Path,
        settings: Dict[str, Any],
        output_format: Optional[str],
        preview_only: bool,
        output_override: Optional[str],
    ) -> List[str]:
        cmd: List[str] = [self._get_python_executable(), str(cli_path), settings["input_path"]]

        # Output path override
        if output_override:
            cmd.extend(["--output", output_override])

        if output_format:
            cmd.extend(["--output_format", output_format])

        if settings.get("model_dir"):
            cmd.extend(["--model_dir", settings["model_dir"]])

        if settings.get("dit_model"):
            cmd.extend(["--dit_model", settings["dit_model"]])

        # Processing params
        def _add_int(flag: str, key: str):
            if settings.get(key) is not None:
                cmd.extend([flag, str(int(settings[key]))])

        _add_int("--resolution", "resolution")
        _add_int("--max_resolution", "max_resolution")
        _add_int("--batch_size", "batch_size")
        if settings.get("uniform_batch_size"):
            cmd.append("--uniform_batch_size")
        _add_int("--seed", "seed")
        _add_int("--skip_first_frames", "skip_first_frames")
        _add_int("--load_cap", "load_cap")
        _add_int("--chunk_size", "chunk_size")  # SeedVR2 native streaming mode
        _add_int("--prepend_frames", "prepend_frames")
        _add_int("--temporal_overlap", "temporal_overlap")

        # Quality
        if settings.get("color_correction"):
            cmd.extend(["--color_correction", settings["color_correction"]])
        if settings.get("input_noise_scale") is not None:
            cmd.extend(["--input_noise_scale", str(settings["input_noise_scale"])])
        if settings.get("latent_noise_scale") is not None:
            cmd.extend(["--latent_noise_scale", str(settings["latent_noise_scale"])])

        # Devices
        if settings.get("cuda_device"):
            cmd.extend(["--cuda_device", settings["cuda_device"]])
        if settings.get("dit_offload_device"):
            cmd.extend(["--dit_offload_device", settings["dit_offload_device"]])
        if settings.get("vae_offload_device"):
            cmd.extend(["--vae_offload_device", settings["vae_offload_device"]])
        if settings.get("tensor_offload_device"):
            cmd.extend(["--tensor_offload_device", settings["tensor_offload_device"]])

        # BlockSwap
        _add_int("--blocks_to_swap", "blocks_to_swap")
        if settings.get("swap_io_components"):
            cmd.append("--swap_io_components")

        # VAE tiling
        if settings.get("vae_encode_tiled"):
            cmd.append("--vae_encode_tiled")
        _add_int("--vae_encode_tile_size", "vae_encode_tile_size")
        _add_int("--vae_encode_tile_overlap", "vae_encode_tile_overlap")
        if settings.get("vae_decode_tiled"):
            cmd.append("--vae_decode_tiled")
        _add_int("--vae_decode_tile_size", "vae_decode_tile_size")
        _add_int("--vae_decode_tile_overlap", "vae_decode_tile_overlap")
        if settings.get("tile_debug") and settings["tile_debug"] != "false":
            cmd.extend(["--tile_debug", settings["tile_debug"]])

        # Performance
        if settings.get("attention_mode"):
            cmd.extend(["--attention_mode", settings["attention_mode"]])
        if settings.get("compile_dit"):
            cmd.append("--compile_dit")
        if settings.get("compile_vae"):
            cmd.append("--compile_vae")
        if settings.get("compile_backend"):
            cmd.extend(["--compile_backend", settings["compile_backend"]])
        if settings.get("compile_mode"):
            cmd.extend(["--compile_mode", settings["compile_mode"]])
        if settings.get("compile_fullgraph"):
            cmd.append("--compile_fullgraph")
        cmd.extend([
            "--compile_dynamic",
            self._normalize_compile_dynamic_setting(settings.get("compile_dynamic"), default="none"),
        ])
        _add_int("--compile_dynamo_cache_size_limit", "compile_dynamo_cache_size_limit")
        _add_int("--compile_dynamo_recompile_limit", "compile_dynamo_recompile_limit")
        if bool(settings.get("split_phase_subprocesses", True)):
            cmd.append("--split_phase_subprocesses")

        # Caching
        if settings.get("cache_dit"):
            cmd.append("--cache_dit")
        if settings.get("cache_vae"):
            cmd.append("--cache_vae")

        # Debug
        if settings.get("debug"):
            cmd.append("--debug")
        
        # ADDED v2.5.22: FFmpeg 10-bit encoding support
        # Video backend selection (opencv or ffmpeg)
        if settings.get("video_backend"):
            backend = str(settings["video_backend"]).lower()
            if backend in ("opencv", "ffmpeg"):
                cmd.extend(["--video_backend", backend])
        
        # 10-bit color depth (requires ffmpeg backend)
        if settings.get("use_10bit"):
            cmd.append("--10bit")

        # Explicit SeedVR2 ffmpeg encoder settings from Output tab.
        # This avoids per-chunk codec drift when relying on CLI defaults.
        codec_raw = str(settings.get("video_codec", "") or "").strip().lower()
        codec_map = {
            "h264": "libx264",
            "avc": "libx264",
            "x264": "libx264",
            "libx264": "libx264",
            "h265": "libx265",
            "hevc": "libx265",
            "x265": "libx265",
            "libx265": "libx265",
            "vp9": "libvpx-vp9",
            "libvpx-vp9": "libvpx-vp9",
            "av1": "libsvtav1",
            "libsvtav1": "libsvtav1",
            "prores": "prores_ks",
            "prores_ks": "prores_ks",
        }
        seed_codec = codec_map.get(codec_raw, "")
        if not seed_codec:
            if bool(settings.get("use_10bit", False) or settings.get("seedvr2_use_10bit", False)):
                seed_codec = "libx265"
            elif codec_raw not in ("", "auto", "default"):
                seed_codec = codec_raw
        if seed_codec:
            cmd.extend(["--video_codec", seed_codec])

        quality_raw = settings.get("video_quality", settings.get("output_quality", None))
        quality_val: Optional[int] = None
        try:
            if quality_raw is not None and str(quality_raw).strip() != "":
                quality_val = int(float(quality_raw))
        except Exception:
            quality_val = None
        if quality_val is not None:
            # Keep in broad ffmpeg CRF range to avoid invalid CLI values.
            quality_val = max(0, min(63, quality_val))
            cmd.extend(["--video_quality", str(quality_val)])

        preset_raw = str(settings.get("video_preset", "") or "").strip().lower()
        allowed_presets = {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        }
        if preset_raw in allowed_presets:
            cmd.extend(["--video_preset", preset_raw])

        if seed_codec == "libx265":
            h265_tune_raw = str(settings.get("h265_tune", "none") or "none").strip().lower()
            if h265_tune_raw and h265_tune_raw != "none":
                cmd.extend(["--h265_tune", h265_tune_raw])

        if seed_codec == "libsvtav1":
            try:
                av1_film_grain = int(float(settings.get("av1_film_grain", 8) or 8))
            except Exception:
                av1_film_grain = 8
            av1_film_grain = max(0, min(50, av1_film_grain))
            av1_film_grain_denoise = bool(settings.get("av1_film_grain_denoise", False))
            cmd.extend(["--av1_film_grain", str(av1_film_grain)])
            cmd.extend(["--av1_film_grain_denoise", "1" if av1_film_grain_denoise else "0"])

        # Preview: prefer PNG for quick visualization
        if preview_only and not output_format:
            cmd.extend(["--output_format", "png"])

        return cmd

    def _maybe_wrap_with_vcvars(self, cmd: List[str], settings: Dict[str, Any], on_progress: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        On Windows, wrap command with vcvarsall.bat to activate C++ toolchain.
        
        The C++ toolchain is required for:
        - torch.compile (when compile_dit/compile_vae are enabled)
        - Some CUDA operations that need nvcc at runtime
        
        Behavior:
        - If compile flags are set: REQUIRE vcvars, disable compile if missing
        - If no compile flags: OPTIONALLY wrap with vcvars if available (best-effort C++ support)
        
        FIXED: Now surfaces warnings to UI via on_progress callback for transparency
        """
        def _strip_compile_flags(command: List[str]) -> List[str]:
            """
            Remove torch.compile-related CLI args from a command list.

            IMPORTANT: Some flags are boolean (no value) and others take a value.
            The previous implementation incorrectly skipped the next token for
            boolean flags like --compile_dit, which could delete unrelated args.
            """
            return Runner._strip_torch_compile_flags(command)

        if platform.system() != "Windows":
            return cmd
        
        compile_requested = settings.get("compile_dit") or settings.get("compile_vae")
        vcvars_path = self._find_vcvars()
        
        if not vcvars_path:
            if compile_requested:
                # Compile was requested but vcvars not found - disable compile flags
                warning_msg = "WARNING: VS Build Tools not found; disabling torch.compile for compatibility.\n" \
                             "Install 'Desktop development with C++' workload from Visual Studio Installer for torch.compile support.\n"
                self._log_lines.append(warning_msg.strip())
                
                # FIXED: Surface warning to UI so user knows compile was disabled
                if on_progress:
                    on_progress(warning_msg)
                
                # Reflect runtime behavior in settings (avoid misleading logs/metadata)
                settings["compile_dit"] = False
                settings["compile_vae"] = False

                # Remove compile-related flags from command
                return _strip_compile_flags(cmd)
            else:
                # No compile requested and vcvars not found - proceed without vcvars
                # Log a warning but don't block execution
                if not hasattr(self, '_vcvars_warning_shown'):
                    info_msg = "INFO: VS Build Tools not found. torch.compile will be unavailable.\n"
                    self._log_lines.append(info_msg.strip())
                    if on_progress:
                        on_progress(info_msg)
                    self._vcvars_warning_shown = True
                return cmd
        
        # FIXED: Only wrap with vcvars if compile is actually requested
        # Wrapping with cmd /c can cause subprocess output capture issues
        if not compile_requested:
            # No compile requested - don't wrap with vcvars (avoid output capture issues)
            if on_progress:
                on_progress("INFO: VS Build Tools available but torch.compile not enabled\n")
            return cmd
        
        # Compile requested and vcvars found - wrap command to enable C++ toolchain
        if on_progress:
            on_progress(f"INFO: Using VS Build Tools for torch.compile: {vcvars_path}\n")
        quoted_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)

        # IMPORTANT:
        # - We must not silently swallow vcvars failures. If vcvarsall.bat fails, `&&` prevents Python
        #   from running, which otherwise looks like "compile failed with no output".
        # - Capture vcvars output to a temp log and print it ONLY if vcvars/cl.exe checks fail.
        # - Do NOT print that log if the Python command itself fails (unrelated errors should surface normally).
        #
        # We rely on %TEMP% (set to our temp_dir in _run_seedvr2_subprocess) to avoid writing to system temp.
        vcvars_log = r"%TEMP%\seedvr2_vcvars.log"
        wrapped_script = (
            f'call "{vcvars_path}" x64 > "{vcvars_log}" 2>&1'
            f' & if errorlevel 1 (echo [SeedVR2][VCVARS_ERROR] vcvarsall.bat failed. Install/repair VS Build Tools (Desktop development with C++).'
            f' & type "{vcvars_log}" & exit /b 1)'
            f' & where cl.exe >> "{vcvars_log}" 2>&1'
            f' & if errorlevel 1 (echo [SeedVR2][VCVARS_ERROR] cl.exe not found after vcvarsall. Install MSVC toolset via VS Installer.'
            f' & type "{vcvars_log}" & exit /b 1)'
            f' & {quoted_cmd}'
        )
        return ["cmd", "/c", wrapped_script]

    def _maybe_clear_in_app_model(self, model_id: Optional[str]):
        """
        Best-effort memory trim when switching models in in-app mode.
        """
        if self._active_mode != "in_app":
            return
        if model_id is None:
            return
        if self._last_model_id and self._last_model_id != model_id:
            try:
                # Avoid importing torch here; only clear if torch is already loaded and CUDA initialized.
                from .gpu_utils import clear_cuda_cache
                clear_cuda_cache()
            except Exception:
                pass
        self._last_model_id = model_id

    # ------------------------------------------------------------------ #
    # RIFE runner
    # ------------------------------------------------------------------ #
    def run_rife(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """
        Run RIFE inference_video.py with given settings.
        
        FIXED: Now supports skip_first_frames and load_cap via ffmpeg preprocessing.
        RIFE CLI doesn't have these options natively, so we pre-trim the video.
        """
        cli_path = self.base_dir / "RIFE" / "inference_video.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"RIFE CLI not found at {cli_path}")

        input_path = normalize_path(settings.get("input_path"))
        if not input_path:
            raise ValueError("Input path is required for RIFE.")

        # FIXED: Pre-process video if skip_first_frames or load_cap is set
        # RIFE CLI doesn't support these natively, so we trim via ffmpeg first
        skip_frames = int(settings.get("skip_first_frames", 0))
        load_cap = int(settings.get("load_cap", 0))
        
        effective_input = input_path
        trimmed_video = None
        
        if (skip_frames > 0 or load_cap > 0) and detect_input_type(input_path) == "video":
            # Need to trim video via ffmpeg
            trimmed_video = self.temp_dir / f"rife_trimmed_{Path(input_path).stem}.mp4"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            if on_progress:
                on_progress(f"Pre-trimming video (skip {skip_frames}, cap {load_cap})...\n")
            
            # Build ffmpeg trim command
            trim_cmd = ["ffmpeg", "-y", "-i", input_path]
            
            if skip_frames > 0:
                # Get FPS to convert frames to seconds
                fps = get_media_fps(input_path) or 30.0
                start_time = skip_frames / fps
                trim_cmd.extend(["-ss", str(start_time)])
            
            if load_cap > 0:
                # Limit number of frames
                trim_cmd.extend(["-frames:v", str(load_cap)])
            
            # Copy codec for fast trim (no re-encode)
            trim_cmd.extend(["-c", "copy", str(trimmed_video)])
            
            try:
                subprocess.run(trim_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                effective_input = str(trimmed_video)
                
                if on_progress:
                    on_progress(f"INFO: Video trimmed: {trimmed_video.name}\n")
            except Exception as e:
                if on_progress:
                    on_progress(f"WARNING: Video trimming failed: {e}, using original input\n")
                # Fall back to original if trim fails
                effective_input = input_path

        output_format_pref = str(settings.get("output_format") or "").strip().lower()
        sequence_requested = bool(settings.get("png_output"))
        img_input_mode = bool(settings.get("img_mode"))
        if sequence_requested:
            wants_sequence_output = True
        elif output_format_pref == "png":
            wants_sequence_output = True
        elif output_format_pref in {"mp4", "mov", "mkv", "avi", "webm", "m4v", "wmv", "flv"}:
            wants_sequence_output = False
        else:
            wants_sequence_output = sequence_requested
        settings["png_output"] = bool(wants_sequence_output)
        output_override = settings.get("output_override") or None
        package_sequence_to_video = bool(img_input_mode and not wants_sequence_output)
        cli_png_output = bool(wants_sequence_output or img_input_mode)
        final_video_output: Optional[Path] = None
        if package_sequence_to_video:
            final_video_output = rife_output_path(
                effective_input,
                False,
                output_override,
                global_output_dir=str(self.output_dir),
                png_padding=settings.get("png_padding"),
                png_keep_basename=settings.get("png_keep_basename", False),
            )
            predicted_output = rife_output_path(
                effective_input,
                True,
                str(final_video_output.parent / f"_{final_video_output.stem}_rife_frames"),
                global_output_dir=str(final_video_output.parent),
                png_padding=settings.get("png_padding"),
                png_keep_basename=settings.get("png_keep_basename", False),
            )
        else:
            predicted_output = rife_output_path(
                effective_input,  # Use trimmed input for output naming
                cli_png_output,
                output_override,
                global_output_dir=str(self.output_dir),
                png_padding=settings.get("png_padding"),
                png_keep_basename=settings.get("png_keep_basename", False),
            )

        cmd = self._build_rife_cmd(
            cli_path,
            effective_input,
            predicted_output,
            settings,
            on_progress=on_progress,
        )
        
        # Wrap with vcvars for C++ toolchain support (Windows only, best-effort)
        # FIXED: Pass on_progress for transparent warning surfacing
        if self._active_mode == "subprocess":
            cmd = self._maybe_wrap_with_vcvars(cmd, settings, on_progress)

        # In-app mode (not cancelable)
        if self._active_mode == "in_app":
            buf = io.StringIO()
            rc = 0
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    sys.argv = cmd[1:]
                    runpy.run_path(str(cli_path), run_name="__main__")
            except SystemExit as exc:  # argparse exit
                rc = int(exc.code) if isinstance(exc.code, int) else 1
            except Exception as exc:
                buf.write(f"{exc}\n")
                rc = 1
            finally:
                sys.argv = [sys.executable]
            output_path = str(predicted_output)
            if cli_png_output:
                finalized_path, finalize_note = _finalize_rife_sequence_output(
                    predicted_output,
                    settings,
                    on_progress=on_progress,
                )
                output_path = str(finalized_path)
                if finalize_note:
                    buf.write(f"{finalize_note}\n")
            if package_sequence_to_video and final_video_output is not None:
                packaged_path, package_note = self._package_rife_sequence_to_video(
                    Path(output_path),
                    final_video_output,
                    settings,
                    on_progress=on_progress,
                )
                if package_note:
                    buf.write(f"{package_note}\n")
                if packaged_path is None:
                    rc = rc or 1
                else:
                    try:
                        shutil.rmtree(Path(output_path), ignore_errors=True)
                    except Exception:
                        pass
                    output_path = str(packaged_path)
            meta_payload = {
                "returncode": rc,
                "output": output_path,
                "args": settings,
            }
            if Path(output_path).exists() and Path(output_path).is_dir():
                write_png_metadata(Path(output_path), meta_payload)
            else:
                emit_metadata(Path(output_path), meta_payload)
            return RunResult(rc, output_path, buf.getvalue())

        env = os.environ.copy()
        env.setdefault("PYTHONWARNINGS", "ignore")
        legacy_alloc_conf = env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if legacy_alloc_conf and not env.get("PYTORCH_ALLOC_CONF"):
            env["PYTORCH_ALLOC_CONF"] = legacy_alloc_conf
        cuda_device = str(settings.get("cuda_device", "") or "").strip().lower()
        if cuda_device.startswith("cuda:"):
            cuda_device = cuda_device.split(":", 1)[1].strip()
        if "," in cuda_device:
            cuda_device = cuda_device.split(",", 1)[0].strip()
        if cuda_device.isdigit():
            # RIFE CLI does not expose a dedicated GPU id flag, so enforce via env.
            env["CUDA_VISIBLE_DEVICES"] = cuda_device
        elif cuda_device in {"cpu", "none"} or str(env.get("SECOURSES_GLOBAL_GPU_DEVICE", "")).strip().lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid  # type: ignore[arg-type]

        proc: Optional[subprocess.Popen] = None
        log_lines: List[str] = []
        start_time = time.time()  # Track execution time
        ratio_re = re.compile(
            r"\b(?:(?:frame|frames?|step|steps?|batch|batches?)\s*)?(\d+)\s*/\s*(\d+)\b",
            flags=re.IGNORECASE,
        )
        last_ratio_done = 0
        last_ratio_total = 0
        ema_step_sec: Optional[float] = None
        sequence_staging_dir: Optional[Path] = None
        sequence_sync_stop: Optional[threading.Event] = None
        sequence_sync_thread: Optional[threading.Thread] = None

        def _emit_eta_progress(done: int, total: int) -> None:
            nonlocal ema_step_sec
            safe_total = max(1, int(total))
            safe_done = max(0, min(int(done), safe_total))
            elapsed = max(1e-6, time.time() - start_time)
            if safe_done > 0:
                step_sec = elapsed / float(safe_done)
                if ema_step_sec is None:
                    ema_step_sec = step_sec
                else:
                    ema_step_sec = (ema_step_sec * 0.7) + (step_sec * 0.3)
            eta_text = "ETA unknown"
            if ema_step_sec is not None and safe_done > 0 and safe_done < safe_total:
                eta_s = max(0.0, float(safe_total - safe_done) * float(ema_step_sec))
                finish_local = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_s))
                eta_text = f"ETA {int(eta_s)}s (finish ~{finish_local})"
            elif safe_done >= safe_total:
                eta_text = "ETA 0s"
            pct = (float(safe_done) / float(safe_total)) * 100.0
            eta_line = f"FRAME_PROGRESS {safe_done}/{safe_total} | {pct:.1f}% | elapsed {int(elapsed)}s | {eta_text}"
            log_lines.append(eta_line)
            try:
                print(eta_line, flush=True)
            except Exception:
                pass
            if on_progress:
                on_progress(eta_line + "\n")
        if cli_png_output:
            sequence_staging_dir = self.base_dir / "RIFE" / "vid_out"
            _clear_rife_sequence_staging(sequence_staging_dir)
            predicted_output.mkdir(parents=True, exist_ok=True)
            sequence_sync_stop = threading.Event()

            def _sequence_sync_worker() -> None:
                while not sequence_sync_stop.is_set():
                    _sync_rife_sequence_staging(sequence_staging_dir, predicted_output)
                    sequence_sync_stop.wait(0.5)
                _sync_rife_sequence_staging(sequence_staging_dir, predicted_output)

            sequence_sync_thread = threading.Thread(
                target=_sequence_sync_worker,
                name="rife-sequence-sync",
                daemon=True,
            )
            sequence_sync_thread.start()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=self.base_dir / "RIFE",
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            with self._lock:
                self._active_process = proc

            assert proc.stdout is not None
            for line in proc.stdout:
                raw_line = line.rstrip()
                log_lines.append(raw_line)
                try:
                    print(raw_line, flush=True)
                except Exception:
                    pass

                ratio_match = ratio_re.search(raw_line)
                if ratio_match:
                    try:
                        done_val = int(ratio_match.group(1))
                        total_val = int(ratio_match.group(2))
                        if total_val > 0:
                            should_emit = (
                                done_val != last_ratio_done
                                or total_val != last_ratio_total
                                or done_val == total_val
                            )
                            if should_emit:
                                last_ratio_done = done_val
                                last_ratio_total = total_val
                                _emit_eta_progress(done_val, total_val)
                    except Exception:
                        pass
                if on_progress:
                    on_progress(line)
                with self._lock:
                    if self._active_process is None:
                        break
            proc.wait()
        finally:
            with self._lock:
                self._active_process = None
            if sequence_sync_stop is not None:
                sequence_sync_stop.set()
            if sequence_sync_thread is not None:
                sequence_sync_thread.join(timeout=5.0)
            if cli_png_output and sequence_staging_dir is not None:
                _sync_rife_sequence_staging(sequence_staging_dir, predicted_output)

        output_path = str(predicted_output)
        sequence_output_path: Optional[Path] = None
        if cli_png_output:
            finalized_path, finalize_note = _finalize_rife_sequence_output(
                predicted_output,
                settings,
                on_progress=on_progress,
            )
            sequence_output_path = finalized_path
            output_path = str(finalized_path)
            if finalize_note:
                log_lines.append(finalize_note)
            if sequence_staging_dir is not None:
                _clear_rife_sequence_staging(sequence_staging_dir)

        returncode_val = proc.returncode if proc else -1
        if returncode_val == 0 and package_sequence_to_video and sequence_output_path is not None and final_video_output is not None:
            packaged_path, package_note = self._package_rife_sequence_to_video(
                sequence_output_path,
                final_video_output,
                settings,
                on_progress=on_progress,
            )
            if package_note:
                log_lines.append(package_note)
            if packaged_path is None:
                returncode_val = 1
            else:
                try:
                    shutil.rmtree(sequence_output_path, ignore_errors=True)
                except Exception:
                    pass
                output_path = str(packaged_path)

        if (
            returncode_val == 0
            and (not package_sequence_to_video)
            and settings.get("fps_override")
            and Path(output_path).suffix.lower() == ".mp4"
        ):
            adjusted = ffmpeg_set_fps(Path(output_path), settings["fps_override"])
            output_path = str(adjusted)
        execution_time = time.time() - start_time
        
        meta_payload = {
            "returncode": returncode_val,
            "output": output_path,
            "args": settings,
            "status": "success" if returncode_val == 0 else ("cancelled" if self._canceled else "failed"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add error context for failures
        if returncode_val != 0 and log_lines:
            meta_payload["error_logs"] = log_lines[-50:]
        
        # Check both global telemetry AND per-run metadata settings
        should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
        if should_emit_metadata:
            try:
                # Emit for all runs (success, failure, cancellation)
                metadata_target = Path(output_path) if Path(output_path).exists() or Path(output_path).parent.exists() else self.output_dir
                if Path(output_path).exists() and Path(output_path).is_dir():
                    write_png_metadata(metadata_target if metadata_target.is_dir() else metadata_target.parent, meta_payload)
                else:
                    emit_metadata(metadata_target, meta_payload)
            except Exception:
                pass  # Don't fail run if metadata fails
        
        # Log command to executed_commands folder
        try:
            self._command_logger.log_command(
                tab_name="rife",
                command=cmd,
                settings=settings,
                returncode=returncode_val,
                output_path=output_path,
                error_logs=log_lines[-50:] if returncode_val != 0 else None,
                execution_time=execution_time,
                additional_info={
                    "mode": "subprocess",
                    "cancelled": self._canceled,
                    "png_output": wants_sequence_output,
                    "cli_png_output": cli_png_output,
                    "sequence_packaged_to_video": package_sequence_to_video,
                    "trimmed_input": str(trimmed_video) if trimmed_video else None
                }
            )
            if on_progress:
                on_progress("INFO: Command logged to executed_commands folder\n")
        except Exception as e:
            if on_progress:
                on_progress(f"WARNING: Failed to log command: {e}\n")
        
        # FIXED: Clean up trimmed temp file if we created one
        if trimmed_video and trimmed_video.exists():
            try:
                trimmed_video.unlink()
                if on_progress:
                    on_progress("INFO: Cleaned up temporary trimmed video\n")
            except Exception:
                pass  # Non-critical cleanup failure

        return RunResult(returncode_val, output_path, "\n".join(log_lines))

    def _rife_model_aliases(self, model_name: str) -> List[str]:
        """Build tolerant aliases for model folder lookup (e.g., rife-v4.26 -> 4.26, v4.26)."""
        raw = str(model_name or "").strip()
        if not raw:
            return []

        aliases: List[str] = []

        def _add(val: str):
            text = str(val or "").strip()
            if text and text not in aliases:
                aliases.append(text)

        _add(raw)
        lower = raw.lower()
        _add(lower)
        norm = lower.replace("_", "-")
        _add(norm)

        if norm.startswith("rife-"):
            tail = norm[len("rife-"):].strip()
            _add(tail)
            if tail.startswith("v") and len(tail) > 1:
                _add(tail[1:])

        if norm.startswith("v") and len(norm) > 1 and norm[1].isdigit():
            no_v = norm[1:]
            _add(no_v)
            _add(f"rife-v{no_v}")

        try:
            float(norm)
            _add(f"v{norm}")
            _add(f"rife-v{norm}")
        except Exception:
            pass

        if norm in {"anime", "rife-anime"}:
            _add("anime")
            _add("rife-anime")

        return aliases

    @staticmethod
    def _normalize_rife_video_codec(codec_raw: Any) -> str:
        text = str(codec_raw or "").strip().lower()
        codec_map = {
            "h264": "libx264",
            "avc": "libx264",
            "x264": "libx264",
            "libx264": "libx264",
            "h265": "libx265",
            "hevc": "libx265",
            "x265": "libx265",
            "libx265": "libx265",
            "vp9": "libvpx-vp9",
            "libvpx-vp9": "libvpx-vp9",
            "av1": "libsvtav1",
            "libsvtav1": "libsvtav1",
            "prores": "prores_ks",
            "prores_ks": "prores_ks",
        }
        return codec_map.get(text, "libx264")

    @staticmethod
    def _rife_video_codec_key(codec_raw: Any) -> str:
        codec_name = Runner._normalize_rife_video_codec(codec_raw)
        codec_map = {
            "libx264": "h264",
            "libx265": "h265",
            "libvpx-vp9": "vp9",
            "libsvtav1": "av1",
            "prores_ks": "prores",
        }
        return codec_map.get(codec_name, "h264")

    @staticmethod
    def _infer_rife_sequence_pattern(frame_dir: Path) -> Tuple[Optional[str], Optional[int]]:
        try:
            frame_paths = sorted(
                p for p in Path(frame_dir).iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        except Exception:
            return None, None

        if not frame_paths:
            return None, None

        first = frame_paths[0]
        match = re.match(r"^(.*?)(\d+)$", first.stem)
        if not match:
            return None, None

        prefix, digits = match.groups()
        return f"{prefix}%0{len(digits)}d{first.suffix.lower()}", int(digits)

    def _package_rife_sequence_to_video(
        self,
        sequence_dir: Path,
        output_path: Path,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Optional[Path], Optional[str]]:
        sequence_dir = Path(sequence_dir)
        output_path = Path(output_path)

        if not sequence_dir.exists() or not sequence_dir.is_dir():
            return None, f"Sequence output directory not found: {sequence_dir}"

        pattern_name, start_number = self._infer_rife_sequence_pattern(sequence_dir)
        if not pattern_name or start_number is None:
            return None, f"Could not detect numbered frames in: {sequence_dir}"

        try:
            output_fps = float(settings.get("fps_override") or 0.0)
        except Exception:
            output_fps = 0.0

        if output_fps <= 0:
            try:
                fps_mult = int(settings.get("fps_multiplier", 2) or 2)
            except Exception:
                fps_mult = 2
            fps_mult = max(1, fps_mult)
            output_fps = 30.0 * float(fps_mult)
            if on_progress:
                on_progress(
                    f"INFO: Image-sequence input has no source FPS metadata; packaging video at "
                    f"{output_fps:.2f}fps (30fps base x{fps_mult}).\n"
                )

        codec_key = self._rife_video_codec_key(settings.get("video_codec", "h264"))
        try:
            quality_val = int(float(settings.get("video_quality", settings.get("output_quality", 23)) or 23))
        except Exception:
            quality_val = 23
        quality_val = max(0, min(63, quality_val))

        video_preset = str(settings.get("video_preset", "slow") or "slow").strip().lower() or "slow"
        pixel_format = str(settings.get("pixel_format", "yuv420p") or "yuv420p").strip().lower() or "yuv420p"
        use_10bit = bool(settings.get("use_10bit", False) or settings.get("seedvr2_use_10bit", False))
        if codec_key == "h265" and use_10bit and "10le" not in pixel_format:
            pixel_format = "yuv420p10le"

        h265_tune = str(settings.get("h265_tune", "none") or "none").strip().lower() or "none"
        try:
            av1_film_grain = int(float(settings.get("av1_film_grain", 8) or 8))
        except Exception:
            av1_film_grain = 8
        av1_film_grain = max(0, min(50, av1_film_grain))
        av1_film_grain_denoise = bool(settings.get("av1_film_grain_denoise", False))

        codec_args = build_ffmpeg_video_encode_args(
            codec=codec_key,
            quality=quality_val,
            pixel_format=pixel_format,
            preset=video_preset,
            audio_codec="none",
            audio_bitrate=None,
            h265_tune=h265_tune,
            av1_film_grain=av1_film_grain,
            av1_film_grain_denoise=av1_film_grain_denoise,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_pattern = str(sequence_dir / pattern_name)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{float(output_fps):.6f}",
            "-start_number",
            str(int(start_number)),
            "-i",
            frame_pattern,
            *codec_args,
        ]
        if output_path.suffix.lower() == ".mp4":
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(output_path))

        if on_progress:
            on_progress(f"Packaging image sequence to video: {output_path.name}\n")

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            return None, f"Failed to package image sequence to video: {exc}"

        if proc.returncode != 0 or not output_path.exists():
            detail = (proc.stdout or "").strip()
            if len(detail) > 1000:
                detail = detail[-1000:]
            return None, (
                f"Failed to package image sequence to video: "
                f"{detail or f'ffmpeg exited with code {proc.returncode}'}"
            )

        return output_path, None

    def _resolve_rife_bundle_dir(self, candidate_dir: Path) -> Optional[Path]:
        """
        Resolve a model bundle dir containing flownet.pkl.

        Supports:
        - direct: <candidate>/flownet.pkl
        - one-level nested zip: <candidate>/<inner>/flownet.pkl
        """
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            return None
        if (candidate_dir / "flownet.pkl").exists():
            return candidate_dir

        children = [p for p in candidate_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if len(children) == 1 and (children[0] / "flownet.pkl").exists():
            return children[0]
        return None

    def _stage_rife_bundle_to_train_log(
        self,
        bundle_dir: Path,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Copy selected RIFE bundle files into RIFE/train_log root so inference_video.py
        can import `train_log.*` modules matching that model package.
        """
        train_log_dir = self.base_dir / "RIFE" / "train_log"
        train_log_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in bundle_dir.iterdir():
            if not src.is_file() or src.name.startswith("."):
                continue
            if src.suffix.lower() not in {".py", ".pkl", ".pth"}:
                continue
            try:
                shutil.copy2(src, train_log_dir / src.name)
                copied += 1
            except Exception:
                continue

        # Minimum requirement for RIFE load_model()
        if not (train_log_dir / "flownet.pkl").exists():
            return False

        if on_progress:
            on_progress(f"Using RIFE model bundle from: {bundle_dir}\n")
        return copied > 0

    def _build_rife_cmd(
        self,
        cli_path: Path,
        input_path: str,
        output_path: Path,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        cmd: List[str] = [self._get_python_executable(), str(cli_path)]

        if settings.get("img_mode"):
            cmd.extend(["--img", input_path])
        else:
            cmd.extend(["--video", input_path])

        cmd.extend(["--output", str(output_path)])

        # Model and device settings
        model_dir_raw = str(settings.get("model_dir") or "").strip()
        model_dir = model_dir_raw
        if model_dir_raw:
            # If override points to a full model bundle, stage it to train_log so
            # inference_video.py imports matching train_log.* python files.
            try:
                override_dir = Path(normalize_path(model_dir_raw))
            except Exception:
                override_dir = Path(model_dir_raw)
            bundle_dir = self._resolve_rife_bundle_dir(override_dir)
            if bundle_dir and (bundle_dir / "flownet.pkl").exists():
                staged = self._stage_rife_bundle_to_train_log(bundle_dir, on_progress=on_progress)
                if staged:
                    model_dir = "train_log"

        if not model_dir:
            model_name = str(settings.get("model", "") or "").strip()
            train_log_root = self.base_dir / "RIFE" / "train_log"
            models_root = self.base_dir / "RIFE" / "models"
            aliases = self._rife_model_aliases(model_name)

            # Explicit legacy selection token(s)
            legacy_tokens = {"legacy-train-log", "legacy", "train_log", "train-log"}
            if any(a.lower() in legacy_tokens for a in aliases):
                if (train_log_root / "flownet.pkl").exists():
                    model_dir = "train_log"

            # Preferred layout: RIFE/models/<model>/... (bundle contains .py + flownet.pkl)
            if not model_dir:
                for alias in aliases:
                    bundle_parent = models_root / alias
                    bundle_dir = self._resolve_rife_bundle_dir(bundle_parent)
                    if bundle_dir and (bundle_dir / "flownet.pkl").exists():
                        staged = self._stage_rife_bundle_to_train_log(bundle_dir, on_progress=on_progress)
                        if staged:
                            model_dir = "train_log"
                            break

            # Legacy per-model layout: RIFE/train_log/<model>/flownet.pkl
            if not model_dir:
                for alias in aliases:
                    named_dir = train_log_root / alias
                    named_bundle = self._resolve_rife_bundle_dir(named_dir)
                    if not named_bundle or not (named_bundle / "flownet.pkl").exists():
                        continue
                    if named_bundle == named_dir:
                        model_dir = f"train_log/{alias}"
                    else:
                        model_dir = str(named_bundle)
                    break

            # Legacy root layout: RIFE/train_log/flownet.pkl
            if not model_dir and (train_log_root / "flownet.pkl").exists():
                model_dir = "train_log"

            # Last-resort compatibility fallback
            if not model_dir and model_name:
                model_dir = f"train_log/{model_name}"
        if model_dir:
            cmd.extend(["--model", str(model_dir)])
        if settings.get("fp16_mode"):
            cmd.append("--fp16")
        if settings.get("uhd_mode"):
            cmd.append("--UHD")

        # Processing parameters
        if settings.get("scale") and settings["scale"] != 1.0:
            cmd.extend(["--scale", str(settings["scale"])])
        # RIFE default is x2; avoid passing "--multi 1" (some builds reject it) and
        # be tolerant of legacy string values like "x2".
        try:
            multi_raw = settings.get("fps_multiplier", 2)
            if isinstance(multi_raw, str):
                multi_s = multi_raw.strip().lower()
                if multi_s.startswith("x"):
                    multi_val = int(multi_s[1:])
                else:
                    multi_val = int(float(multi_s))
            else:
                multi_val = int(multi_raw)
        except Exception:
            multi_val = 2

        if multi_val > 1 and multi_val != 2:
            cmd.extend(["--multi", str(int(multi_val))])
        if settings.get("fps_override") and settings["fps_override"] > 0:
            cmd.extend(["--fps", str(settings["fps_override"])])
        if settings.get("exp") and settings["exp"] != 1:
            cmd.extend(["--exp", str(settings["exp"])])

        # Output options
        if settings.get("png_output"):
            cmd.append("--png")
            sequence_format = _normalize_rife_sequence_format(settings.get("sequence_format", "png"))
            cmd.extend(["--sequence-format", sequence_format])
            if sequence_format == "jpg":
                try:
                    sequence_quality = int(float(settings.get("sequence_quality", 95) or 95))
                except Exception:
                    sequence_quality = 95
                sequence_quality = max(1, min(100, sequence_quality))
                cmd.extend(["--jpg-quality", str(sequence_quality)])
        if settings.get("montage"):
            cmd.append("--montage")
        if settings.get("no_audio"):
            cmd.append("--no-audio")
        if settings.get("show_ffmpeg"):
            cmd.append("--show-ffmpeg")
        if settings.get("skip_static_frames"):
            cmd.append("--skip")

        # Encoding options (Output tab source-of-truth). This keeps RIFE output codec/pixfmt
        # aligned with user-selected video settings instead of hardcoded x264 defaults.
        video_codec = self._normalize_rife_video_codec(settings.get("video_codec", "h264"))
        cmd.extend(["--video-codec", video_codec])

        try:
            quality_val = int(float(settings.get("video_quality", settings.get("output_quality", 23)) or 23))
        except Exception:
            quality_val = 23
        quality_val = max(0, min(63, quality_val))
        cmd.extend(["--video-crf", str(quality_val)])

        video_preset = str(settings.get("video_preset", "slow") or "slow").strip().lower()
        if not video_preset:
            video_preset = "slow"
        cmd.extend(["--video-preset", video_preset])

        if video_codec == "libx265":
            h265_tune_raw = str(settings.get("h265_tune", "none") or "none").strip().lower()
            if h265_tune_raw and h265_tune_raw != "none":
                cmd.extend(["--h265-tune", h265_tune_raw])

        if video_codec in {"libsvtav1", "libaom-av1"}:
            try:
                av1_film_grain = int(float(settings.get("av1_film_grain", 8) or 8))
            except Exception:
                av1_film_grain = 8
            av1_film_grain = max(0, min(50, av1_film_grain))
            av1_film_grain_denoise = bool(settings.get("av1_film_grain_denoise", False))
            cmd.extend(["--av1-film-grain", str(av1_film_grain)])
            cmd.extend(["--av1-film-grain-denoise", "1" if av1_film_grain_denoise else "0"])

        pixel_format = str(settings.get("pixel_format", "yuv420p") or "yuv420p").strip().lower()
        if not pixel_format:
            pixel_format = "yuv420p"
        use_10bit = bool(settings.get("use_10bit", False) or settings.get("seedvr2_use_10bit", False))
        if video_codec == "libx265" and use_10bit and "10le" not in pixel_format:
            pixel_format = "yuv420p10le"
        cmd.extend(["--pixel-format", pixel_format])

        # Frame control: skip_first_frames and load_cap are now handled via ffmpeg preprocessing
        # in run_rife() before RIFE CLI is called, so no CLI args needed here.
        # RIFE's --skip flag is for static frame detection, not frame trimming.

        return cmd

    # ------------------------------------------------------------------ #
    # GAN/image-based upscaler runner
    # ------------------------------------------------------------------ #
    def run_gan(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """
        Run GAN-based upscaling using the proper GAN runner implementation.
        """
        from .gan_runner import run_gan_upscale, GanResult

        try:
            # Convert settings to GAN runner format
            model_name = settings.get("model", "")
            if not model_name:
                error_msg = "No GAN model specified"
                if on_progress:
                    on_progress(f"ERROR: {error_msg}\n")
                return RunResult(1, None, error_msg)

            input_path = normalize_path(settings.get("input_path"))
            if not input_path or not Path(input_path).exists():
                error_msg = f"Input path not found: {input_path}"
                if on_progress:
                    on_progress(f"ERROR: {error_msg}\n")
                return RunResult(1, None, error_msg)

            if on_progress:
                on_progress(f"Starting GAN upscaling with model: {model_name}\n")
                on_progress(f"Input: {input_path}\n")

            # Honor output overrides for chunking/batch workflows.
            # The GAN runner primarily supports choosing an output directory, so when an explicit
            # file path is provided we run into a safe directory and then move/rename the result.
            output_override_raw = settings.get("output_override")
            explicit_output_path: Optional[Path] = None
            output_dir_for_run = Path(self.output_dir)
            if output_override_raw:
                try:
                    override_path = Path(normalize_path(str(output_override_raw)))
                    if override_path.exists() and override_path.is_dir():
                        output_dir_for_run = override_path
                    else:
                        # Treat common media extensions as an explicit file-path override.
                        file_exts = {
                            ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".m4v", ".flv",
                            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
                        }
                        if override_path.suffix.lower() in file_exts:
                            explicit_output_path = override_path
                            output_dir_for_run = override_path.parent
                        else:
                            # No extension: treat as a directory override.
                            output_dir_for_run = override_path
                except Exception:
                    output_dir_for_run = Path(self.output_dir)
                    explicit_output_path = None

            # Safety: avoid writing the output into the same folder as the input when the
            # GAN runner would otherwise overwrite the input filename (common for chunk files).
            try:
                input_dir = Path(input_path).parent.resolve()
                if output_dir_for_run.resolve() == input_dir:
                    output_dir_for_run = output_dir_for_run / "__gan_out"
            except Exception:
                pass
            try:
                output_dir_for_run.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # Run the GAN processing using proper runner
            result: GanResult = run_gan_upscale(
                input_path=input_path,
                model_name=model_name,
                settings=settings,
                base_dir=self.base_dir,
                temp_dir=self.temp_dir,
                output_dir=output_dir_for_run,
                on_progress=on_progress,
                cancel_event=None
            )

            # Output path comes from result
            output_path = result.output_path
            if explicit_output_path and output_path and Path(output_path).exists():
                try:
                    explicit_output_path.parent.mkdir(parents=True, exist_ok=True)
                    dest = collision_safe_path(explicit_output_path)
                    shutil.move(str(output_path), str(dest))
                    output_path = str(dest)
                    result.output_path = output_path
                except Exception:
                    # Best-effort: keep the original output path on move failure.
                    pass

            # Emit metadata if successful and enabled
            should_emit_metadata = self._telemetry_enabled and settings.get("save_metadata", True)
            if output_path and result.returncode == 0 and should_emit_metadata:
                try:
                    emit_metadata(
                        Path(output_path),
                        {
                            "returncode": result.returncode,
                            "output": output_path,
                            "args": settings,
                            "model": model_name,
                        },
                    )
                except Exception as e:
                    if on_progress:
                        on_progress(f"Warning: Failed to emit metadata: {e}\n")

            return RunResult(result.returncode, output_path, result.log)

        except Exception as e:
            error_msg = f"GAN processing failed: {str(e)}"
            if on_progress:
                on_progress(f"ERROR: {error_msg}\n")
            return RunResult(1, None, error_msg)



