"""
Unified Comparison System - Single Entry Point for All Comparison Features

This module consolidates all comparison functionality from:
- video_comparison.py (basic HTML comparison)
- video_comparison_slider.py (advanced slider)
- video_comparison_advanced.py (side-by-side generation)

Provides a single, parameterized API that automatically selects the
best comparison method based on input type and user preferences.
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any, List

# Import from existing modules
from .video_comparison_slider import create_video_comparison_html as _slider_html
from .video_comparison_advanced import (
    create_side_by_side_video,
    create_stacked_video,
    ComparisonConfig
)


ComparisonMode = Literal["slider", "side_by_side", "stacked", "native", "auto"]


def create_unified_comparison(
    input_path: Optional[str],
    output_path: Optional[str],
    mode: ComparisonMode = "auto",
    height: int = 600,
    enable_fullscreen: bool = True,
    pin_reference: bool = False,
    pinned_reference_path: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Unified comparison creator that automatically chooses the best method.
    
    Args:
        input_path: Path to input file (original)
        output_path: Path to output file (processed)
        mode: Comparison mode:
            - "auto": Automatically choose best method
            - "slider": HTML slider comparison (videos)
            - "native": Use Gradio's ImageSlider (images)
            - "side_by_side": Generate side-by-side video
            - "stacked": Generate vertically stacked video
        height: Display height in pixels
        enable_fullscreen: Enable fullscreen button
        pin_reference: Use pinned reference instead of current input
        pinned_reference_path: Path to pinned reference file
        
    Returns:
        (html_update, image_slider_update) - gr.update() objects for components
    """
    
    # Validate inputs
    if not input_path or not output_path:
        return gr.update(visible=False), gr.update(visible=False)
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists() or not output_file.exists():
        return gr.update(visible=False), gr.update(visible=False)
    
    # Determine input/output types
    is_input_video = input_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    is_output_video = output_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    is_input_image = input_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']
    is_output_image = output_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']
    
    # Handle pinned reference
    reference_path = pinned_reference_path if pin_reference and pinned_reference_path else input_path
    
    # Auto-select mode if requested
    if mode == "auto":
        if is_input_video and is_output_video:
            mode = "slider"  # Default to slider for videos
        elif is_input_image and is_output_image:
            mode = "native"  # Use Gradio's native ImageSlider for images
        else:
            # Mixed types - can't compare
            return gr.update(visible=False), gr.update(visible=False)
    
    # Execute comparison based on mode
    if mode == "native" and is_input_image and is_output_image:
        # Use Gradio's native ImageSlider for images - return update, not component
        slider_update = gr.update(
            value=(reference_path, output_path),
            visible=True
        )
        return gr.update(visible=False), slider_update
    
    elif mode == "slider" and is_input_video and is_output_video:
        # Use custom HTML slider for videos
        html_content = _slider_html(
            original_video=reference_path,
            upscaled_video=output_path,
            height=height,
            slider_position=50.0
        )
        return gr.update(value=html_content, visible=True), gr.update(visible=False)
    
    elif mode == "side_by_side" and is_input_video and is_output_video:
        # Generate side-by-side comparison video
        from .path_utils import collision_safe_path
        
        comparison_path = collision_safe_path(
            output_file.parent / f"{output_file.stem}_comparison_sidebyside.mp4"
        )
        
        success, comp_path, error = create_side_by_side_video(
            input_video=reference_path,
            output_video=output_path,
            comparison_output=str(comparison_path)
        )
        
        if success and comp_path:
            # Return as video player HTML
            html_content = f"""
            <div style="text-align: center; padding: 20px;">
                <h3>Side-by-Side Comparison</h3>
                <video controls style="max-width: 100%; height: {height}px;">
                    <source src="file={comp_path}" type="video/mp4">
                </video>
                <p style="color: #666; margin-top: 10px;">Saved to: {Path(comp_path).name}</p>
            </div>
            """
            return gr.update(value=html_content, visible=True), gr.update(visible=False)
        else:
            # Fallback to slider if generation failed
            html_content = _slider_html(reference_path, output_path, height)
            return gr.update(value=html_content, visible=True), gr.update(visible=False)
    
    elif mode == "stacked" and is_input_video and is_output_video:
        # Generate vertically stacked comparison video
        from .path_utils import collision_safe_path
        
        comparison_path = collision_safe_path(
            output_file.parent / f"{output_file.stem}_comparison_stacked.mp4"
        )
        
        success, comp_path, error = create_stacked_video(
            input_video=reference_path,
            output_video=output_path,
            comparison_output=str(comparison_path)
        )
        
        if success and comp_path:
            # Return as video player HTML
            html_content = f"""
            <div style="text-align: center; padding: 20px;">
                <h3>Stacked Comparison (Top: Original, Bottom: Upscaled)</h3>
                <video controls style="max-width: 100%; max-height: {height}px;">
                    <source src="file={comp_path}" type="video/mp4">
                </video>
                <p style="color: #666; margin-top: 10px;">Saved to: {Path(comp_path).name}</p>
            </div>
            """
            return gr.update(value=html_content, visible=True), gr.update(visible=False)
        else:
            # Fallback to slider
            html_content = _slider_html(reference_path, output_path, height)
            return gr.update(value=html_content, visible=True), gr.update(visible=False)
    
    # Fallback: return nothing if incompatible
    return gr.update(visible=False), gr.update(visible=False)


def get_comparison_modes_for_type(is_video: bool) -> List[str]:
    """
    Get available comparison modes for given content type.
    
    Args:
        is_video: True if content is video, False for images
        
    Returns:
        List of valid comparison mode strings
    """
    if is_video:
        return ["auto", "slider", "side_by_side", "stacked"]
    else:
        return ["auto", "native"]


def build_comparison_selector(
    input_path: Optional[str],
    output_path: Optional[str],
    mode: ComparisonMode = "auto",
    **kwargs
) -> Tuple[Optional[str], gr.ImageSlider]:
    """
    Legacy compatibility wrapper for existing code.
    
    Returns:
        (html_string, image_slider_update)
    """
    html_comp, image_slider = create_unified_comparison(
        input_path, output_path, mode, **kwargs
    )
    
    # Convert to legacy format
    if html_comp:
        return html_comp.value if hasattr(html_comp, 'value') else str(html_comp), gr.update(visible=False)
    elif image_slider:
        return "", image_slider
    else:
        return "", gr.update(visible=False)


# Expose commonly used functions directly
def create_video_comparison_slider(
    original_video: str,
    upscaled_video: str,
    height: int = 600
) -> str:
    """Quick helper for video slider HTML"""
    return _slider_html(original_video, upscaled_video, height)


def create_image_comparison_slider(
    input_image: str,
    output_image: str
):
    """Quick helper for image slider update"""
    _, slider_update = create_unified_comparison(
        input_image,
        output_image,
        mode="native"
    )
    return slider_update if slider_update else gr.update(visible=False)

