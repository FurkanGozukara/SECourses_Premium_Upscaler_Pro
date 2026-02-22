"""
Shared UI Layout Components

Provides reusable two-column layouts and common UI patterns to maintain
consistency across tabs and prevent code duplication.
"""

import gradio as gr
from typing import Callable, List, Optional, Tuple, Any
from shared.video_comparison_slider import get_video_comparison_js_on_load


def two_column_layout(
    left_content_fn: Callable,
    right_content_fn: Callable,
    left_scale: int = 3,
    right_scale: int = 2
):
    """
    Create a standardized two-column layout.
    
    Args:
        left_content_fn: Function that builds left column content
        right_content_fn: Function that builds right column content
        left_scale: Scale ratio for left column (default 3)
        right_scale: Scale ratio for right column (default 2)
        
    Returns:
        Tuple of (left_components, right_components) returned by content functions
    """
    with gr.Row():
        with gr.Column(scale=left_scale):
            left_components = left_content_fn()
        
        with gr.Column(scale=right_scale):
            right_components = right_content_fn()
    
    return left_components, right_components


def create_input_section(
    label: str = "Input Configuration",
    file_types: Optional[List[str]] = None,
    upload_label: str = "Upload File",
    path_label: str = "Input Path",
    path_placeholder: str = "C:/path/to/file",
    path_info: str = "Direct path to input file or folder",
    default_path: str = "",
    show_batch: bool = True,
    batch_default_enable: bool = False,
    batch_input_default: str = "",
    batch_output_default: str = ""
) -> Tuple[gr.File, gr.Textbox, gr.Markdown, gr.Checkbox, gr.Textbox, gr.Textbox]:
    """
    Create standardized input section with file upload and batch processing controls.
    
    Returns:
        Tuple of (input_file, input_path, input_cache_msg, batch_enable, batch_input, batch_output)
    """
    with gr.Accordion(f"📁 {label}", open=True):
        input_file = gr.File(
            label=upload_label,
            type="filepath",
            file_types=file_types or ["video", "image"]
        )
        
        input_path = gr.Textbox(
            label=path_label,
            value=default_path,
            placeholder=path_placeholder,
            info=path_info
        )
        
        input_cache_msg = gr.Markdown("", visible=False)
        
        if show_batch:
            batch_enable = gr.Checkbox(
                label="Enable Batch Processing",
                value=batch_default_enable,
                info="Process multiple files from directory"
            )
            batch_input = gr.Textbox(
                label="Batch Input Folder",
                value=batch_input_default,
                placeholder="Folder containing files to process",
                info="Directory with files to process in batch mode"
            )
            batch_output = gr.Textbox(
                label="Batch Output Folder Override",
                value=batch_output_default,
                placeholder="Optional override for batch outputs",
                info="Custom output directory for batch results"
            )
        else:
            batch_enable = gr.Checkbox(value=False, visible=False)
            batch_input = gr.Textbox(value="", visible=False)
            batch_output = gr.Textbox(value="", visible=False)
    
    return input_file, input_path, input_cache_msg, batch_enable, batch_input, batch_output


def create_output_section(
    show_video: bool = True,
    show_image: bool = True,
    show_gallery: bool = False
) -> Tuple[gr.Markdown, gr.Markdown, gr.Textbox, Optional[gr.Video], Optional[gr.Image], Optional[gr.Gallery], gr.ImageSlider, gr.HTML]:
    """
    Create standardized output section with status, progress, and comparison displays.
    
    Returns:
        Tuple of (status_box, progress_indicator, log_box, output_video, output_image, 
                 batch_gallery, image_slider, video_comparison_html)
    """
    gr.Markdown("### 🎯 Output / Results")
    
    status_box = gr.Markdown(value="Ready for processing.", visible=False, elem_classes=["runtime-status-box"])
    progress_indicator = gr.Markdown(value="", visible=False, elem_classes=["runtime-progress-box"])
    
    log_box = gr.Textbox(
        label="📋 Processing Log",
        value="",
        lines=10,
        buttons=["copy"]
    )
    
    output_video = None
    if show_video:
        output_video = gr.Video(
            label="🎬 Upscaled Video",
            interactive=False,
            buttons=["download"]
        )
    
    output_image = None
    if show_image:
        output_image = gr.Image(
            label="🖼️ Upscaled Image",
            interactive=False,
            buttons=["download"]
        )
    
    batch_gallery = None
    if show_gallery:
        batch_gallery = gr.Gallery(
            label="📦 Batch Results",
            visible=False,
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain",
            buttons=["download"]
        )
    
    # Enhanced comparison slider
    image_slider = gr.ImageSlider(
        label="🔍 Before/After Comparison",
        interactive=False,
        slider_position=50,
        max_height=1000,
        buttons=["download", "fullscreen"],
        elem_classes=["native-image-comparison-slider"],
    )
    
    # Video comparison HTML
    video_comparison_html = gr.HTML(
        label="🎬 Video Comparison Slider",
        value="",
        js_on_load=get_video_comparison_js_on_load(),
        visible=False
    )
    
    return status_box, progress_indicator, log_box, output_video, output_image, batch_gallery, image_slider, video_comparison_html


def create_action_buttons(
    show_preview: bool = True,
    show_cancel: bool = True,
    ffmpeg_available: bool = True
) -> Tuple[gr.Button, gr.Checkbox, Optional[gr.Button], Optional[gr.Button]]:
    """
    Create standardized action button section.
    
    Returns:
        Tuple of (upscale_btn, cancel_confirm, cancel_btn, preview_btn)
    """
    with gr.Row():
        upscale_btn = gr.Button(
            "🚀 Start Processing" if ffmpeg_available else "❌ Processing (ffmpeg required)",
            variant="primary" if ffmpeg_available else "stop",
            size="lg",
            interactive=ffmpeg_available,
            elem_classes=["action-btn", "action-btn-upscale"],
        )
        
        cancel_confirm = None
        cancel_btn = None
        if show_cancel:
            cancel_confirm = gr.Checkbox(
                label="⚠️ Confirm cancel (subprocess only)",
                value=False,
                info="Enable to confirm cancellation"
            )
            cancel_btn = gr.Button(
                "⏹️ Cancel",
                variant="stop",
                scale=1,
                elem_classes=["action-btn", "action-btn-cancel"],
            )
        
        preview_btn = None
        if show_preview:
            preview_btn = gr.Button(
                "👁️ Preview First Frame" if ffmpeg_available else "❌ Preview (ffmpeg required)",
                size="lg",
                interactive=ffmpeg_available,
                elem_classes=["action-btn", "action-btn-preview"],
            )
    
    return upscale_btn, cancel_confirm, cancel_btn, preview_btn


def create_utility_buttons() -> Tuple[gr.Button, gr.Button, gr.Checkbox]:
    """
    Create standardized utility button section.
    
    Returns:
        Tuple of (open_outputs_btn, clear_temp_btn, delete_confirm)
    """
    with gr.Row():
        open_outputs_btn = gr.Button(
            "📂 Open Outputs Folder",
            size="lg",
            elem_classes=["action-btn", "action-btn-open"],
        )
        clear_temp_btn = gr.Button(
            "🗑️ Clear Temp Files",
            size="lg",
            elem_classes=["action-btn", "action-btn-clear"],
        )
    
    delete_confirm = gr.Checkbox(
        label="⚠️ Confirm delete temp (required for safety)",
        value=False,
        info="Enable to confirm deletion"
    )
    
    return open_outputs_btn, clear_temp_btn, delete_confirm


def create_gpu_warning_banner(cuda_available: bool, gpu_hint: str, model_name: str = "Model") -> None:
    """
    Create standardized GPU availability warning banner.
    
    Args:
        cuda_available: Whether CUDA is available
        gpu_hint: GPU detection hint message
        model_name: Name of the model/pipeline for context
    """
    if not cuda_available:
        gr.Markdown(
            f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
            f'<strong>⚠️ GPU Acceleration Unavailable</strong><br>'
            f'{gpu_hint}<br><br>'
            f'{model_name} processing requires GPU acceleration. CPU fallback is significantly slower (10-100x).<br>'
            f'Install CUDA-enabled PyTorch for optimal performance.'
            f'</div>',
            elem_classes="warning-text"
        )
