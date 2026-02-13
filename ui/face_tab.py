"""
Face Restoration Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from typing import Dict, Any

from shared.services.face_service import (
    build_face_callbacks, FACE_ORDER
)
from shared.models.seedvr2_meta import get_seedvr2_model_names
from ui.media_preview import preview_updates
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values
from shared.processing_queue import get_processing_queue_manager
from shared.queue_state import (
    snapshot_queue_state,
    snapshot_global_settings,
    merge_payload_state,
)


def _get_gan_model_names(base_dir) -> list:
    """Get GAN model names from the app's model folders (supports legacy layout)."""
    from pathlib import Path
    from shared.models import scan_gan_models
    return scan_gan_models(Path(base_dir))


def face_tab(preset_manager, global_settings: Dict[str, Any], shared_state: gr.State, base_dir=None):
    """
    Self-contained Face Restoration tab.
    Handles face restoration settings shared across models.
    
    Args:
        base_dir: Base directory for the application (required for GAN model discovery)
    """
    
    # Derive base_dir from global settings if not provided
    if base_dir is None:
        from pathlib import Path
        output_dir_path = Path(global_settings.get("output_dir", "./outputs"))
        # Navigate up from outputs to project root
        if output_dir_path.name == "outputs":
            base_dir = output_dir_path.parent
        else:
            # Fallback: try to find model folders in parent directories
            current = output_dir_path
            for _ in range(3):  # Search up to 3 levels
                if (current / "models").exists() or (current / "Image_Upscale_Models").exists():
                    base_dir = current
                    break
                current = current.parent
            else:
                # Last resort: use current working directory
                base_dir = Path.cwd()

    # Get available models
    seed_controls = shared_state.value.get("seed_controls", {})
    shared_models = seed_controls.get("available_models", [])
    if not isinstance(shared_models, list):
        shared_models = []
    combined_models = sorted(
        {
            str(model_name).strip()
            for model_name in shared_models
            if str(model_name).strip()
            and (
                str(model_name).lower().startswith("seedvr2_")
                or str(model_name).lower().endswith(".pth")
                or str(model_name).lower().endswith(".safetensors")
                or str(model_name).lower().endswith(".gguf")
            )
        }
    )
    if not combined_models:
        # Fallback for legacy state/migrations: discover model lists directly.
        seedvr2_models = get_seedvr2_model_names()
        gan_models = _get_gan_model_names(base_dir)
        combined_models = sorted(list({*seedvr2_models, *gan_models}))

    # Build service callbacks
    service = build_face_callbacks(preset_manager, global_settings, combined_models, shared_state)
    queue_manager = get_processing_queue_manager()

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    face_settings = seed_controls.get("face_settings", {})
    models_list = seed_controls.get("available_models", combined_models)
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in face_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults.get(k, defaults.get(k, "")) for k in FACE_ORDER]
    # Layout
    gr.Markdown("### 👤 Face Restoration Settings")
    gr.Markdown("*Configure face enhancement applied after upscaling across all models*")

    # Global face restoration toggle
    with gr.Group():
        gr.Markdown("#### 🌐 Global Face Restoration")
        global_face_enabled = gr.Checkbox(
            label="⚡ Enable Globally (All Upscalers)",
            value=global_settings.get("face_global", False),
            info="When enabled, face restoration is automatically applied to ALL video and image upscaling operations"
        )

        with gr.Group():
            gr.Markdown("""
            **ℹ️ Global Mode Info:**
            - Face restoration will be applied to SeedVR2, GAN, and RIFE processing
            - Settings below control the enhancement strength and behavior
            - Universal presets manage these settings across the app
            - Processing time increases with face restoration enabled
            """)

        apply_global_btn = gr.Button(
            "💾 Save Global Face Setting",
            variant="secondary",
            size="lg"
        )
        global_status = gr.Markdown("")

    # Legacy model context is kept as a hidden field for preset/order compatibility.
    model_selector_value = values[0] if values else (combined_models[0] if combined_models else "default")
    if combined_models and model_selector_value not in combined_models:
        model_selector_value = combined_models[0]
    model_selector = gr.Textbox(
        value=model_selector_value,
        visible=False,
        interactive=False,
    )

    with gr.Tabs():
        # Standalone processing (image/video + batch)
        with gr.TabItem("🧪 Standalone Processing"):
            gr.Markdown("#### Restore faces on an image/video directly (no upscaling)")
            gr.Markdown("*Uses the same face restoration backend used by the other pipelines.*")

            with gr.Row():
                # Left: input
                with gr.Column(scale=3):
                    with gr.Row():
                        input_file = gr.File(
                            label="Upload Image or Video",
                            type="filepath",
                            file_types=["image", "video"],
                        )
                        with gr.Column():
                            input_image_preview = gr.Image(
                                label="📸 Input Preview (Image)",
                                type="filepath",
                                interactive=False,
                                height=250,
                                visible=False,
                            )
                            input_video_preview = gr.Video(
                                label="🎬 Input Preview (Video)",
                                interactive=False,
                                height=250,
                                visible=False,
                            )

                    standalone_input_path = gr.Textbox(
                        label="Input Path (alternative to upload)",
                        value=values[18] if len(values) > 18 else "",
                        placeholder="C:/path/to/image.png or C:/path/to/video.mp4",
                        info="Upload wins if both are set. For batch, enable Batch Processing below.",
                    )

                    standalone_output_override = gr.Textbox(
                        label="Output Override (optional)",
                        value=values[19] if len(values) > 19 else "",
                        placeholder="Leave empty for auto output in Outputs folder",
                        info="Can be a file path (recommended) or a directory path (suffixless).",
                    )

                    input_cache_msg = gr.Markdown("", visible=False)

                    with gr.Accordion("📦 Batch Processing (multiple files)", open=False):
                        standalone_batch_enable = gr.Checkbox(
                            label="Enable Batch Processing",
                            value=values[20] if len(values) > 20 else False,
                            info="Process all supported media in a directory (recursive).",
                        )
                        standalone_batch_input = gr.Textbox(
                            label="Batch Input Path",
                            value=values[21] if len(values) > 21 else "",
                            placeholder="Folder containing images/videos (or a single media file)",
                        )
                        standalone_batch_output = gr.Textbox(
                            label="Batch Output Folder Override (optional)",
                            value=values[22] if len(values) > 22 else "",
                            placeholder="Leave empty for default outputs folder",
                        )

                    process_btn = gr.Button(
                        "✨ Restore Faces",
                        variant="primary",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-upscale"],
                    )

                # Right: output
                with gr.Column(scale=2):
                    process_status = gr.Markdown(value="Ready.")
                    process_log = gr.Textbox(
                        label="📋 Log",
                        value="",
                        lines=10,
                        buttons=["copy"],
                    )
                    restored_image = gr.Image(
                        label="🖼️ Restored Image",
                        interactive=False,
                        buttons=["download"],
                        visible=False,
                    )
                    restored_video = gr.Video(
                        label="🎬 Restored Video",
                        interactive=False,
                        buttons=["download"],
                        visible=False,
                    )
                    batch_outputs = gr.File(
                        label="📦 Batch Outputs",
                        file_count="multiple",
                        type="filepath",
                        interactive=False,
                    )

            # Input caching + previews
            def _cache_upload(val, state):
                state = state or {}
                try:
                    state.setdefault("seed_controls", {})
                    state["seed_controls"]["last_input_path"] = val or ""
                except Exception:
                    pass
                img_prev, vid_prev = preview_updates(val)
                return (
                    val or "",
                    img_prev,
                    vid_prev,
                    gr.update(value="✅ Input cached for face restoration.", visible=True),
                    state,
                )

            input_file.upload(
                fn=_cache_upload,
                inputs=[input_file, shared_state],
                outputs=[standalone_input_path, input_image_preview, input_video_preview, input_cache_msg, shared_state],
            )

            input_file.change(
                fn=lambda p: preview_updates(p),
                inputs=[input_file],
                outputs=[input_image_preview, input_video_preview],
            )

            standalone_input_path.change(
                fn=lambda p: preview_updates(p),
                inputs=[standalone_input_path],
                outputs=[input_image_preview, input_video_preview],
            )

        # Model Selection
        with gr.TabItem("🤖 Model Selection"):
            gr.Markdown("#### Face Restoration Model")
            
            # Get available backends
            from shared.face_restore import get_available_backends
            available_backends = get_available_backends()
            
            if not available_backends:
                gr.Markdown("""
                **⚠️ No Face Restoration Models Available**
                
                Please install one of the following:
                - **GFPGAN**: `pip install gfpgan`
                - **CodeFormer**: Coming soon
                
                Until installed, face restoration will be skipped.
                """)
                backend_choices = ["auto"]
            else:
                backend_choices = ["auto"] + available_backends
            
            with gr.Group():
                # Note: face_backend is NOT in FACE_ORDER - it's for display only
                # The actual restoration model is controlled by restoration_model in the Restoration tab
                face_backend = gr.Dropdown(
                    label="Face Restoration Backend (Info Only)",
                    choices=backend_choices,
                    value="auto",
                    info="'auto' uses first available backend. GFPGAN is recommended. This is informational - configure in Face Restoration tab.",
                    interactive=False
                )
                
                backend_info = gr.Markdown("")
                
                def update_backend_info(backend):
                    if backend == "auto":
                        if available_backends:
                            return f"**Auto**: Will use {available_backends[0]} (first available)"
                        else:
                            return "**Auto**: No backends available - face restoration will be skipped"
                    elif backend == "gfpgan":
                        return """
                        **GFPGAN v1.3**
                        - High quality face restoration
                        - Works on photos and videos
                        - GPU accelerated
                        - Best for realistic faces
                        """
                    elif backend == "codeformer":
                        return """
                        **CodeFormer**
                        - Advanced face restoration
                        - Better for heavily degraded faces
                        - Controllable fidelity
                        - Coming soon
                        """
                    return ""
                
                face_backend.change(
                    fn=update_backend_info,
                    inputs=face_backend,
                    outputs=backend_info
                )
        
        # Face Detection Settings
        with gr.TabItem("🔍 Face Detection"):
            gr.Markdown("#### Face Detection Configuration")

            with gr.Group():
                face_detector = gr.Dropdown(
                    label="Face Detection Model",
                    choices=["retinaface", "yunet", "opencv", "dlib"],
                    value=values[1],  # FACE_ORDER index 1 = face_detector
                    info="Algorithm for detecting faces in images/videos"
                )

                detection_confidence = gr.Slider(
                    label="Detection Confidence Threshold",
                    minimum=0.1, maximum=1.0, step=0.05,
                    value=values[2],  # FACE_ORDER index 2 = detection_confidence
                    info="Minimum confidence for face detection (higher = fewer false positives)"
                )

                min_face_size = gr.Number(
                    label="Minimum Face Size (pixels)",
                    value=values[3],  # FACE_ORDER index 3 = min_face_size
                    precision=0,
                    info="Skip faces smaller than this size"
                )

                max_faces = gr.Number(
                    label="Maximum Faces Per Frame",
                    value=values[4],  # FACE_ORDER index 4 = max_faces
                    precision=0,
                    info="Limit faces processed per frame (0 = unlimited)"
                )

        # Restoration Settings
        with gr.TabItem("✨ Face Restoration"):
            gr.Markdown("#### Enhancement Parameters")

            with gr.Group():
                restoration_model = gr.Dropdown(
                    label="Restoration Model",
                    choices=["gfpgan", "restoreformer", "codeformer", "auto"],
                    value=values[5],  # FACE_ORDER index 5 = restoration_model
                    info="'auto' selects best model based on input quality"
                )

                face_strength = gr.Slider(
                    label="Face Enhancement Strength",
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=values[6],  # FACE_ORDER index 6 = face_strength
                    info="How strongly to apply face restoration (0 = no change, 1 = maximum enhancement)"
                )

                restore_blindly = gr.Checkbox(
                    label="Restore All Faces Blindly",
                    value=values[7],  # FACE_ORDER index 7 = restore_blindly
                    info="Apply restoration to all detected faces without quality checks"
                )

                upscale_faces = gr.Checkbox(
                    label="Upscale Face Region First",
                    value=values[8],  # FACE_ORDER index 8 = upscale_faces
                    info="Pre-upscale face areas before restoration"
                )

        # Advanced Settings
        with gr.TabItem("⚙️ Advanced Settings"):
            gr.Markdown("#### Fine-tuning & Performance")

            with gr.Group():
                face_padding = gr.Slider(
                    label="Face Padding Ratio",
                    minimum=0.1, maximum=1.0, step=0.1,
                    value=values[9],  # FACE_ORDER index 9 = face_padding
                    info="Extra area around face for context"
                )

                face_landmarks = gr.Checkbox(
                    label="Use Face Landmarks",
                    value=values[10],  # FACE_ORDER index 10 = use_landmarks (key name differs)
                    info="Guide restoration using facial feature detection"
                )

                color_correction = gr.Checkbox(
                    label="Apply Color Correction",
                    value=values[11],  # FACE_ORDER index 11 = color_correction
                    info="Match face colors to surrounding skin tone"
                )

                gpu_acceleration = gr.Checkbox(
                    label="Enable GPU Acceleration",
                    value=values[12],  # FACE_ORDER index 12 = gpu_acceleration
                    info="Use GPU for face processing (faster but uses more VRAM)"
                )

                batch_face_processing = gr.Checkbox(
                    label="Batch Face Processing",
                    value=values[13],  # FACE_ORDER index 13 = batch_faces (key name differs)
                    info="Process multiple faces simultaneously"
                )

        # Quality & Output
        with gr.TabItem("🎨 Quality & Output"):
            gr.Markdown("#### Output Quality Settings")

            with gr.Group():
                output_quality = gr.Slider(
                    label="Output Quality",
                    minimum=0.1, maximum=1.0, step=0.05,
                    value=values[14],  # FACE_ORDER index 14 = output_quality
                    info="Restoration output quality (higher = better but slower)"
                )

                preserve_original = gr.Checkbox(
                    label="Preserve Original When Better",
                    value=values[15],  # FACE_ORDER index 15 = preserve_original
                    info="Keep original face if restoration doesn't improve quality"
                )

                artifact_reduction = gr.Checkbox(
                    label="Artifact Reduction",
                    value=values[16],  # FACE_ORDER index 16 = artifact_reduction
                    info="Apply additional filtering to reduce processing artifacts"
                )

                save_face_masks = gr.Checkbox(
                    label="Save Face Masks",
                    value=values[17],  # FACE_ORDER index 17 = save_face_masks
                    info="Save debug masks showing detected face regions"
                )

    # UNIVERSAL PRESET MANAGEMENT
    with gr.Accordion("💾 Preset Management", open=True):
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
            tab_name="face",
            inputs_list=[],
            base_dir=base_dir,
            models_list=models_list,
            open_accordion=False,  # Already in accordion
        )

    # Performance info
    with gr.Accordion("📊 Performance Impact", open=False):
        gr.Markdown("""
        #### Face Restoration Performance Notes

        **Processing Time Increase:**
        - Single face: ~2-3x slower
        - Multiple faces: ~3-5x slower
        - HD video: Significantly longer

        **Memory Usage:**
        - GPU: Additional 2-4GB VRAM
        - CPU: Additional 4-8GB RAM

        **Best Practices:**
        - Enable only when needed
        - Use global mode for batch processing
        - Lower strength for subtle improvements
        - Use GPU acceleration when available
        """)

    # Collect inputs for callbacks
    inputs_list = [
        model_selector, face_detector, detection_confidence, min_face_size, max_faces,
        restoration_model, face_strength, restore_blindly, upscale_faces,
        face_padding, face_landmarks, color_correction, gpu_acceleration, batch_face_processing,
        output_quality, preserve_original, artifact_reduction, save_face_masks
        , standalone_input_path, standalone_output_override, standalone_batch_enable, standalone_batch_input, standalone_batch_output
    ]

    # Wire up callbacks
    
    # Global face toggle
    apply_global_btn.click(
        fn=lambda enabled, state: service["set_face_global"](enabled, state),
        inputs=[global_face_enabled, shared_state],
        outputs=[global_status, shared_state]
    )

    def _queued_waiting_output(state, ticket_id: str, position: int):
        safe_state = state or {}
        pos = max(1, int(position)) if position else "?"
        return (
            gr.update(value=f"Queue waiting: {ticket_id} (position {pos})"),
            gr.update(value=f"Queued and waiting for active processing slot. Queue position: {pos}."),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _queued_cancelled_output(state, ticket_id: str):
        safe_state = state or {}
        return (
            gr.update(value=f"Queue item removed: {ticket_id}"),
            gr.update(value="This queued request was removed before processing started."),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def _queue_disabled_busy_output(state):
        safe_state = state or {}
        return (
            gr.update(value="Processing already in progress (queue disabled)."),
            gr.update(value="Enable 'Enable Queue' in Global Settings to stack additional requests."),
            gr.update(),
            gr.update(),
            gr.update(),
            safe_state,
        )

    def run_process_with_queue(upload, *args):
        live_state = args[-1] if (args and isinstance(args[-1], dict)) else {}
        queued_state = snapshot_queue_state(live_state)
        queued_global_settings = snapshot_global_settings(global_settings)
        queue_enabled = bool(queued_global_settings.get("queue_enabled", True))
        ticket = queue_manager.submit("Face", "Restore")
        acquired_slot = queue_manager.is_active(ticket.job_id)

        try:
            if not queue_enabled:
                if not acquired_slot:
                    queue_manager.cancel_waiting([ticket.job_id])
                    yield _queue_disabled_busy_output(live_state)
                    return
                payload = service["run_action"](
                    upload,
                    *args[:-1],
                    state=queued_state,
                    global_settings_snapshot=queued_global_settings,
                )
                yield merge_payload_state(payload, live_state)
                return

            wait_notice_sent = False
            while not ticket.start_event.wait(timeout=0.5):
                if ticket.cancel_event.is_set():
                    yield _queued_cancelled_output(live_state, ticket.job_id)
                    return
                if not wait_notice_sent:
                    try:
                        pos = queue_manager.waiting_position(ticket.job_id)
                        pos_text = max(1, int(pos)) if pos else "?"
                        gr.Info(f"Queued: {ticket.job_id} (position {pos_text})")
                    except Exception:
                        pass
                    wait_notice_sent = True

            if ticket.cancel_event.is_set() and not queue_manager.is_active(ticket.job_id):
                yield _queued_cancelled_output(live_state, ticket.job_id)
                return

            acquired_slot = True
            payload = service["run_action"](
                upload,
                *args[:-1],
                state=queued_state,
                global_settings_snapshot=queued_global_settings,
            )
            yield merge_payload_state(payload, live_state)
        finally:
            if acquired_slot:
                queue_manager.complete(ticket.job_id)
            else:
                queue_manager.cancel_waiting([ticket.job_id])

    # Standalone processing
    process_btn.click(
        fn=run_process_with_queue,
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[process_status, process_log, restored_image, restored_video, batch_outputs, shared_state],
        concurrency_limit=32,
        concurrency_id="app_processing_queue",
        trigger_mode="multiple",
    )

    # UNIVERSAL PRESET EVENT WIRING
    wire_universal_preset_events(
        preset_dropdown=preset_dropdown,
        preset_name_input=preset_name_input,
        save_btn=save_preset_btn,
        load_btn=load_preset_btn,
        preset_status=preset_status,
        reset_btn=reset_defaults_btn,
        delete_btn=delete_preset_btn,
        callbacks=preset_callbacks,
        inputs_list=inputs_list,
        shared_state=shared_state,
        tab_name="face",
    )

    return {
        "inputs_list": inputs_list,
        "preset_dropdown": preset_dropdown,
        "preset_status": preset_status,
    }

