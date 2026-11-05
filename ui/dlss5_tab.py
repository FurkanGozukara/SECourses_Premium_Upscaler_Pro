"""DLSS 5 Neural Rendering tab: render existing videos and images, or enable it after every upscaler."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gradio as gr

from shared.dlss5.motion import MOTION_QUALITY
from shared.dlss5.runner import MOTION_CHOICES, PRECISION_CHOICES, PROFILE_CHOICES, SPEED_CHOICES
from shared.services.dlss5_service import DLSS5_ORDER, build_dlss5_callbacks
from shared.video_comparison_slider import get_video_comparison_js_on_load
from ui.media_preview import preview_updates
from ui.universal_preset_section import universal_preset_section, wire_universal_preset_events

PROFILE_INFO = {
    "Standard": "NVIDIA's default look: photoreal lighting, skin and materials.",
    "Natural": "Softer, closer to the source colours.",
    "Cinematic": "Stronger contrast and filmic tone.",
    "Neutral": "Structure only: no local tone or local structure push.",
}

HERO_HTML = """
<div class="dlss5-hero">
  <div class="dlss5-hero-title">✨ DLSS 5 Neural Rendering</div>
  <div class="dlss5-hero-sub">NVIDIA's generative DLSS 5 renderer for your existing videos and images: photoreal lighting,
  skin, hair and material detail, kept stable over time with motion-compensated history. Runs on every
  RTX 30 / 40 / 50 GPU; the model (292 MB) downloads automatically on the first run.</div>
  <div class="dlss5-chips">
    <span class="dlss5-chip">🎞️ Temporal video</span><span class="dlss5-chip">🖼️ Images &amp; folders</span>
    <span class="dlss5-chip">🔗 Post-process after any upscaler</span><span class="dlss5-chip">⛓️ Usable in Chains</span>
  </div>
</div>
"""


def dlss5_tab(*, preset_manager, global_settings: Dict[str, Any], shared_state: gr.State, base_dir: Path, output_dir: Path):
    service = build_dlss5_callbacks(global_settings=global_settings, base_dir=base_dir, output_dir=output_dir)
    seed_controls = shared_state.value.get("seed_controls", {}) if isinstance(shared_state.value, dict) else {}
    saved = seed_controls.get("dlss5_settings", {}) if isinstance(seed_controls, dict) else {}
    values = dict(service["defaults"])
    if isinstance(saved, dict):
        values.update({k: v for k, v in saved.items() if v is not None and k in values})

    def v(key):
        return values.get(key)

    with gr.Row(elem_classes=["dlss5-tab"]):
        with gr.Column(scale=3):
            gr.HTML(HERO_HTML)
            with gr.Group(elem_classes=["dlss5-card"]):
                with gr.Row():
                    with gr.Column(scale=3):
                        input_file = gr.File(label="Upload video or image", type="filepath", file_types=["video", "image"])
                        input_path = gr.Textbox(label="Input Path (video, image or image folder)", value=v("input_path") or "",
                                                placeholder="C:/videos/clip.mp4")
                    with gr.Column(scale=3):
                        input_image_preview = gr.Image(label="Input", type="filepath", interactive=False, height=230, visible=False)
                        input_video_preview = gr.Video(label="Input", interactive=False, height=230, visible=False)

            with gr.Group(elem_classes=["dlss5-card"]):
                gr.Markdown("#### 🎨 Look", elem_classes=["dlss5-card-title"])
                profile = gr.Radio(label="Style profile", choices=PROFILE_CHOICES, value=v("profile") or "Standard",
                                   info="DLSS 5 artistic-direction profiles.")
                profile_info = gr.Markdown(PROFILE_INFO.get(v("profile") or "Standard", ""), elem_classes=["resolution-info"])
                with gr.Row():
                    intensity = gr.Slider(label="Intensity", minimum=0.0, maximum=1.0, step=0.05, value=float(v("intensity")),
                                          info="1.0 = the full DLSS 5 render; lower blends back toward the source.")
                    detail_strength = gr.Slider(label="Detail strength", minimum=0.0, maximum=3.0, step=0.05,
                                                value=float(v("detail_strength")), info="Fine texture the render adds (high frequencies).")
                    colour_strength = gr.Slider(label="Colour & lighting strength", minimum=0.0, maximum=3.0, step=0.05,
                                                value=float(v("colour_strength")), info="Broad lighting / colour change (low frequencies).")

            with gr.Accordion("🎞️ Temporal stability (video)", open=False, elem_classes=["dlss5-card"]):
                with gr.Row():
                    temporal = gr.Checkbox(label="Temporal history (recommended)", value=bool(v("temporal")),
                                           info="Reprojects the previous rendered frame with optical flow and lets the network blend it: no flicker.")
                    motion_quality = gr.Dropdown(label="Motion estimation", choices=MOTION_CHOICES,
                                                 value=v("motion_quality") if v("motion_quality") in MOTION_CHOICES else "Balanced",
                                                 info="Optical-flow resolution: " + ", ".join(
                                                     f"{k} {d or 'full'} px" for k, (_p, d) in MOTION_QUALITY.items()))
                    scene_cut_threshold = gr.Slider(label="Scene-cut threshold", minimum=0.05, maximum=1.0, step=0.05,
                                                    value=float(v("scene_cut_threshold")),
                                                    info="History resets when a cut changes the picture this much.")

            with gr.Accordion("⚙️ Advanced", open=False, elem_classes=["dlss5-card"]):
                with gr.Row():
                    processing_scale = gr.Slider(label="Processing scale", minimum=1.0, maximum=2.0, step=0.25,
                                                 value=float(v("processing_scale")),
                                                 info="Render at a higher internal resolution, then downsample: finer detail, ~scale² slower.")
                    detail_radius = gr.Slider(label="Detail radius (px)", minimum=1.0, maximum=16.0, step=0.5,
                                              value=float(v("detail_radius")), info="Split point between detail and colour strength.")
                with gr.Row():
                    tone_strength = gr.Slider(label="Local tone", minimum=0.0, maximum=2.0, step=0.05, value=float(v("tone_strength")),
                                              info="Network local tone-mapping control (profile default 1.0).")
                    structure_strength = gr.Slider(label="Local structure", minimum=0.0, maximum=2.0, step=0.05,
                                                   value=float(v("structure_strength")),
                                                   info="Network local structure control; lower if the result looks over-detailed.")
                with gr.Row():
                    precision = gr.Dropdown(label="Precision", choices=PRECISION_CHOICES,
                                            value=v("precision") if v("precision") in PRECISION_CHOICES else PRECISION_CHOICES[0])
                    speed_mode = gr.Dropdown(label="Execution", choices=SPEED_CHOICES,
                                             value=v("speed_mode") if v("speed_mode") in SPEED_CHOICES else SPEED_CHOICES[0],
                                             info="CUDA Graphs replays the whole network per frame (same result, faster).")
                    max_frames = gr.Number(label="Max frames (0 = all)", value=int(v("max_frames") or 0), precision=0, minimum=0)
                    keep_model_loaded = gr.Checkbox(label="Keep model in VRAM", value=bool(v("keep_model_loaded")),
                                                    info="Off frees VRAM for the other upscalers after each run.")

            with gr.Group(elem_classes=["dlss5-card", "dlss5-post-card"]):
                gr.Markdown("#### 🔗 Post-processing for every upscaler", elem_classes=["dlss5-card-title"])
                global_postprocess = gr.Checkbox(
                    label="Apply DLSS 5 automatically after every upscale (SeedVR2, FlashVSR, SparkVSR, GAN, RTX, LTX 2.5, Open-H3)",
                    value=bool(v("global_postprocess")),
                    info="Uses the settings of this tab on each finished video; the upscaled file is kept and a *_DLSS5.mp4 "
                         "is added. Runs before Global RIFE, so RIFE interpolates the DLSS 5 frames.",
                )

            with gr.Accordion("📁 Batch & output", open=False, elem_classes=["dlss5-card"]):
                batch_enable = gr.Checkbox(label="Batch: render every video and image in a folder", value=bool(v("batch_enable")))
                with gr.Row():
                    batch_input_path = gr.Textbox(label="Batch input folder", value=v("batch_input_path") or "")
                    batch_output_path = gr.Textbox(label="Batch output folder (empty = <outputs>/<folder>_DLSS5)",
                                                   value=v("batch_output_path") or "")
                output_override = gr.Textbox(label="Output Override (single run)", value=v("output_override") or "",
                                             placeholder="Empty = a numbered run folder in the outputs folder")

            (preset_dropdown, preset_name_input, save_preset_btn, load_preset_btn, preset_status, reset_defaults_btn,
             delete_preset_btn, preset_callbacks) = universal_preset_section(
                preset_manager=preset_manager, shared_state=shared_state, tab_name="dlss5", inputs_list=[],
                base_dir=base_dir, models_list=seed_controls.get("available_models", ["default"]), open_accordion=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("### Output / Actions")
            status_box = gr.Markdown("Ready. Upload a video or image and press **Render DLSS 5**.", elem_classes=["runtime-status-box"])
            progress_indicator = gr.HTML("", visible=False)
            with gr.Row():
                render_btn = gr.Button("✨ Render DLSS 5", variant="primary", elem_classes=["action-btn", "sec-btn-emerald"], scale=3)
                preview_btn = gr.Button("⚡ Preview", elem_classes=["action-btn", "sec-btn-cyan"], scale=2)
                cancel_btn = gr.Button("⏹️ Cancel", variant="stop", elem_classes=["action-btn", "sec-btn-red"], scale=1)
            output_video = gr.Video(label="DLSS 5 Output", interactive=False, visible=False)
            output_image = gr.Image(label="DLSS 5 Output", type="filepath", interactive=False, visible=False)
            image_slider = gr.ImageSlider(label="Before / After", interactive=False, visible=False, max_height=900,
                                          buttons=["download", "fullscreen"], elem_classes=["native-image-comparison-slider"])
            video_comparison_html = gr.HTML(label="Video Comparison", value="", js_on_load=get_video_comparison_js_on_load(), visible=False)
            with gr.Accordion("Log", open=False):
                log_box = gr.Textbox(label="Log", lines=14, max_lines=24, interactive=False, autoscroll=True)

    inputs_list = [
        input_path, output_override, batch_enable, batch_input_path, batch_output_path, profile, intensity,
        detail_strength, colour_strength, detail_radius, processing_scale, temporal, motion_quality, scene_cut_threshold,
        tone_strength, structure_strength, precision, speed_mode, max_frames, keep_model_loaded, global_postprocess,
    ]
    assert len(inputs_list) == len(DLSS5_ORDER)

    def _on_upload(path):
        image_upd, video_upd = preview_updates(path)
        return (gr.update(value=path) if path else gr.update()), image_upd, video_upd

    input_file.change(fn=_on_upload, inputs=[input_file], outputs=[input_path, input_image_preview, input_video_preview],
                      queue=False, show_progress="hidden")
    input_path.submit(fn=lambda p: preview_updates(p), inputs=[input_path],
                      outputs=[input_image_preview, input_video_preview], queue=False, show_progress="hidden")
    profile.change(fn=lambda p: PROFILE_INFO.get(p, ""), inputs=[profile], outputs=[profile_info], queue=False,
                   show_progress="hidden")

    outputs = [status_box, log_box, progress_indicator, output_image, output_video, image_slider, video_comparison_html, shared_state]

    def run_render(upload, *args, progress=gr.Progress()):
        state = args[-1] if args and isinstance(args[-1], dict) else {}
        yield from service["run_action"](upload, *args[:-1], state=state, progress=progress)

    def run_preview(upload, *args, progress=gr.Progress()):
        state = args[-1] if args and isinstance(args[-1], dict) else {}
        vals = list(args[:-1])
        vals[DLSS5_ORDER.index("max_frames")] = 48
        vals[DLSS5_ORDER.index("batch_enable")] = False
        yield from service["run_action"](upload, *vals, state=state, progress=progress)

    render_btn.click(fn=run_render, inputs=[input_file] + inputs_list + [shared_state], outputs=outputs,
                     concurrency_limit=32, concurrency_id="app_processing_queue", trigger_mode="multiple")
    preview_btn.click(fn=run_preview, inputs=[input_file] + inputs_list + [shared_state], outputs=outputs,
                      concurrency_limit=32, concurrency_id="app_processing_queue", trigger_mode="multiple")
    cancel_btn.click(fn=service["cancel_action"], outputs=[status_box], queue=False)

    wire_universal_preset_events(
        preset_dropdown=preset_dropdown, preset_name_input=preset_name_input, save_btn=save_preset_btn,
        load_btn=load_preset_btn, preset_status=preset_status, reset_btn=reset_defaults_btn, delete_btn=delete_preset_btn,
        callbacks=preset_callbacks, inputs_list=inputs_list, shared_state=shared_state, tab_name="dlss5",
    )
    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
        "service": service,
        "input_file": input_file,
    }
