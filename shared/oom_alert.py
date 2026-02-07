"""
VRAM Out-Of-Memory (OOM) detection + user-facing guidance banner.

This project runs heavy GPU workloads via subprocesses. When a subprocess or in-app
run hits VRAM OOM, the raw error can be cryptic (especially in logs).

This helper:
- Detects common VRAM OOM signatures from text/exception
- Builds a big, actionable HTML banner for the Gradio UI
- Stores/clears the banner in the shared_state dict
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def is_vram_oom_text(text: Any) -> bool:
    """Best-effort detection of VRAM OOM from a log line / exception string."""
    if text is None:
        return False
    t = str(text).lower()
    if not t:
        return False

    # Strong signals (PyTorch / CUDA)
    if "torch.outofmemoryerror" in t:
        return True
    if "cuda out of memory" in t:
        return True
    if "cublas_status_alloc_failed" in t or "cudnn_status_alloc_failed" in t:
        return True
    if "allocation on device" in t:
        return True

    # Weaker signals: require GPU-ish context to reduce false positives.
    generic = ("out of memory" in t) or ("ran out of memory" in t) or ("failed to allocate" in t)
    gpu_context = ("cuda" in t) or ("gpu" in t) or ("vram" in t) or ("device" in t) or ("nvidia" in t)
    return bool(generic and gpu_context)


def extract_oom_snippet(text: str, max_lines: int = 18, context: int = 3, max_chars: int = 1400) -> str:
    """Extract a small snippet around the first OOM-ish line for display."""
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""

    hit_idx = None
    for i, line in enumerate(lines):
        if is_vram_oom_text(line):
            hit_idx = i
            break

    if hit_idx is None:
        # Fallback: show tail, which is usually where the exception is printed.
        snippet_lines = lines[-max_lines:]
    else:
        start = max(0, hit_idx - context)
        end = min(len(lines), hit_idx + context + max_lines)
        snippet_lines = lines[start:end]

    snippet = "\n".join(snippet_lines).strip()
    if len(snippet) > max_chars:
        snippet = snippet[-max_chars:]
        snippet = "…\n" + snippet
    return snippet


def _fmt_setting(settings: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not settings:
        return None
    val = settings.get(key)
    if val is None or val == "":
        return None
    return f"{key}={val}"


def build_vram_oom_html(
    *,
    model_label: Optional[str] = None,
    details_text: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    snippet = extract_oom_snippet(details_text or "")
    snippet_html = ""
    if snippet:
        snippet_html = (
            "<div class='vram-oom-snippet-wrap'>"
            "<div class='vram-oom-snippet-title'>Captured error snippet</div>"
            f"<pre class='vram-oom-snippet'>{_html_escape(snippet)}</pre>"
            "</div>"
        )

    # Show a compact settings line (when available) to help users correlate.
    settings_bits = []
    for k in (
        "dit_model",
        "upscale_factor",
        "max_resolution",
        "batch_size",
        "blocks_to_swap",
        "swap_io_components",
        "dit_offload_device",
        "vae_offload_device",
        "tensor_offload_device",
        "vae_encode_tiled",
        "vae_encode_tile_size",
        "vae_decode_tiled",
        "vae_decode_tile_size",
        "chunk_size",  # SeedVR2 native streaming (frames)
    ):
        s = _fmt_setting(settings, k)
        if s:
            settings_bits.append(s)
    settings_line = ""
    if settings_bits:
        settings_line = f"<div class='vram-oom-settings'>Current: {_html_escape(' | '.join(settings_bits[:10]))}</div>"

    model_line = ""
    if model_label:
        model_line = f"<div class='vram-oom-model'>Pipeline: <strong>{_html_escape(model_label)}</strong></div>"

    # Actionable guidance: keep the first section as "do these first".
    return f"""
<div class="vram-oom-banner" role="alert" aria-live="assertive">
  <div class="vram-oom-title">🚫 OUT OF VRAM (GPU Memory)</div>
  {model_line}
  {settings_line}
  <div class="vram-oom-subtitle">Your GPU ran out of memory while allocating tensors. Change settings below and retry.</div>

  <div class="vram-oom-grid">
    <div class="vram-oom-card">
      <div class="vram-oom-card-title">✅ Fastest fixes (try in this order)</div>
      <ul class="vram-oom-list">
        <li><strong>Lower Max Resolution (max edge)</strong> (e.g. 1920 → 1600 → 1280).</li>
        <li><strong>Lower Upscale Factor</strong> (e.g. 4x → 3x → 2x).</li>
        <li><strong>Lower Batch Size</strong> (SeedVR2: use a smaller 4n+1 like 5; 1 is also valid but slower).</li>
        <li><strong>Turn OFF Cache DiT / Cache VAE</strong> (caching increases baseline VRAM).</li>
        <li><strong>Turn OFF torch.compile</strong> (compile can increase memory footprint).</li>
      </ul>
    </div>

    <div class="vram-oom-card">
      <div class="vram-oom-card-title">🧠 Memory-saver controls (SeedVR2)</div>
      <ul class="vram-oom-list">
        <li><strong>Enable DiT Offload Device = cpu</strong> and <strong>VAE Offload Device = cpu</strong>.</li>
        <li><strong>Increase BlockSwap → Blocks to Swap</strong> (try 20–30 for 8GB GPUs).</li>
        <li><strong>Enable Swap I/O Components</strong> (max VRAM savings; slower).</li>
        <li><strong>Enable VAE Encode Tiled + VAE Decode Tiled</strong>.</li>
        <li><strong>Reduce VAE Tile Size</strong> (1024 → 768 → 512) if still OOM.</li>
      </ul>
    </div>
  </div>

  <details class="vram-oom-details">
    <summary class="vram-oom-summary">More options (advanced)</summary>
    <ul class="vram-oom-list">
      <li><strong>Use Chunking</strong>: in <em>Resolution &amp; Scene Split</em> tab enable PySceneDetect chunking (smaller chunk size + overlap) to keep VRAM bounded on long videos.</li>
      <li><strong>SeedVR2 Native Streaming</strong>: set <em>SeedVR2 Native Streaming (frames per chunk)</em> to a small number (e.g. 10–30) for very long videos.</li>
      <li><strong>Pick a smaller model</strong>: SeedVR2 3B uses much less VRAM than 7B; sharp variants can be a bit heavier.</li>
      <li><strong>Close other GPU apps</strong> (games, browsers, OBS) and use the app’s <em>Clear CUDA Cache</em> button, then retry.</li>
      <li><strong>Prefer Subprocess Mode</strong>: it fully cleans VRAM after each run and avoids in-app memory creep.</li>
    </ul>
  </details>

  {snippet_html}
</div>
""".strip()


def clear_vram_oom_alert(state: Optional[Dict[str, Any]]) -> None:
    """Clear the VRAM OOM banner stored in shared_state."""
    if not isinstance(state, dict):
        return
    alerts = state.setdefault("alerts", {})
    alerts["oom"] = {"visible": False, "html": "", "ts": None, "modal_shown": False}


def maybe_set_vram_oom_alert(
    state: Optional[Dict[str, Any]],
    *,
    model_label: Optional[str],
    text: Any,
    settings: Optional[Dict[str, Any]] = None,
) -> bool:
    """If text looks like VRAM OOM, set banner in shared_state and return True."""
    if not isinstance(state, dict):
        return False
    if not is_vram_oom_text(text):
        return False

    alerts = state.setdefault("alerts", {})
    prev = alerts.get("oom") if isinstance(alerts.get("oom"), dict) else {}
    modal_shown = bool(prev.get("modal_shown")) if isinstance(prev, dict) else False
    alerts["oom"] = {
        "visible": True,
        "html": build_vram_oom_html(
            model_label=model_label,
            details_text=str(text) if text is not None else "",
            settings=settings,
        ),
        "ts": time.time(),
        "modal_shown": modal_shown,
    }
    return True


def show_vram_oom_modal(
    state: Optional[Dict[str, Any]],
    *,
    title: str = "Out of VRAM (GPU)",
    duration: float | None = None,
) -> None:
    """
    Best-effort: show a Gradio modal popup containing the current VRAM OOM guidance HTML stored in shared_state.

    - Uses `gr.Warning(...)` so it is visually prominent.
    - Default `duration=None` means it stays until the user closes it.
    - If Gradio isn't available or the state doesn't contain HTML, this is a no-op / fallback message.
    """
    try:
        import gradio as gr  # Local import: keep this module usable outside the Gradio UI.

        info = (state or {}).get("alerts", {}).get("oom", {}) if isinstance(state, dict) else {}
        html = info.get("html", "") if isinstance(info, dict) else ""

        # Prevent spamming multiple modals in the same run.
        if isinstance(state, dict):
            alerts = state.setdefault("alerts", {})
            oom = alerts.setdefault("oom", {}) if isinstance(alerts, dict) else {}
            if isinstance(oom, dict) and oom.get("modal_shown"):
                return
            if isinstance(oom, dict):
                oom["modal_shown"] = True

        if html:
            gr.Warning(html, title=title, duration=duration)
        else:
            gr.Warning(
                "🚫 Out of VRAM (GPU). Try lowering max resolution/upscale factor, reducing batch size, "
                "enabling tiling/chunking, and closing other GPU apps, then retry.",
                title=title,
                duration=duration,
            )
    except Exception:
        # If queue isn't enabled or any UI error occurs, silently skip (banner still exists).
        return


