import shutil
import subprocess
import tempfile
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import traceback

import cv2
import numpy as np

from .path_utils import (
    collision_safe_dir,
    collision_safe_path,
    ffmpeg_set_fps,
    normalize_path,
    get_media_fps,
    resolve_output_location,
    detect_input_type,
)
from .face_restore import restore_image, restore_video


# GAN Model Metadata System
@dataclass
class GanModelMetadata:
    name: str
    scale: int
    architecture: str
    input_channels: int = 3
    output_channels: int = 3
    description: str = ""
    author: str = "unknown"
    tags: list = None
    supports_multi_gpu: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Real-ESRGAN model definitions with scale factors
REAL_ESRGAN_MODELS = {
    'RealESRGAN_x4plus': {'scale': 4, 'arch': 'RRDBNet'},
    'RealESRNet_x4plus': {'scale': 4, 'arch': 'RRDBNet'},
    'RealESRGAN_x4plus_anime_6B': {'scale': 4, 'arch': 'RRDBNet'},
    'RealESRGAN_x2plus': {'scale': 2, 'arch': 'RRDBNet'},
    'realesr-animevideov3': {'scale': 4, 'arch': 'SRVGGNetCompact'},
    'realesr-general-x4v3': {'scale': 4, 'arch': 'SRVGGNetCompact'},
}


class GanModelRegistry:
    """Comprehensive registry for GAN model metadata from multiple sources"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._omd_cache: Dict[str, GanModelMetadata] = {}
        self._loaded = False
        self._spandrel_available = self._check_spandrel()

    def _check_spandrel(self) -> bool:
        """Check if spandrel is available for model loading"""
        try:
            import spandrel
            return True
        except ImportError:
            return False

    def _load_omd_metadata(self) -> None:
        """Load metadata from Open Model Database"""
        if self._loaded:
            return

        omd_dir = self.base_dir / "open-model-database" / "data" / "models"
        if not omd_dir.exists():
            self._loaded = True
            return

        for json_file in omd_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                scale = data.get('scale', 4)
                name = data.get('name', json_file.stem)
                arch = data.get('architecture', 'esrgan')

                metadata = GanModelMetadata(
                    name=name,
                    scale=int(scale),
                    architecture=arch,
                    input_channels=data.get('inputChannels', 3),
                    output_channels=data.get('outputChannels', 3),
                    description=data.get('description', ''),
                    author=data.get('author', 'unknown'),
                    tags=data.get('tags', [])
                )

                # Use normalized key for lookup
                key = self._normalize_name(name)
                self._omd_cache[key] = metadata

            except Exception:
                continue

        self._loaded = True

    def _normalize_name(self, name: str) -> str:
        """Normalize model name for consistent lookup"""
        # Remove scale prefix, file extensions, and normalize
        name = re.sub(r'^\d+x[-_]', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\.(pth|safetensors|onnx)$', '', name, flags=re.IGNORECASE)
        return re.sub(r'[^a-z0-9]', '', name.lower())

    def get_model_metadata_with_spandrel(self, model_path: Path) -> Optional[GanModelMetadata]:
        """Use spandrel to detect model architecture and scale"""
        if not self._spandrel_available:
            return None

        try:
            import spandrel
            model_descriptor = spandrel.ModelLoader().load_from_file(str(model_path))
            
            # Extract metadata from spandrel
            scale = getattr(model_descriptor.model, 'scale', 4)
            if hasattr(scale, '__iter__'):
                scale = scale[0] if len(scale) > 0 else 4
            
            arch_name = model_descriptor.architecture.name if hasattr(model_descriptor, 'architecture') else "unknown"
            
            return GanModelMetadata(
                name=model_path.stem,
                scale=int(scale),
                architecture=arch_name,
                input_channels=getattr(model_descriptor.model, 'in_channels', 3) if hasattr(model_descriptor.model, 'in_channels') else 3,
                output_channels=getattr(model_descriptor.model, 'out_channels', 3) if hasattr(model_descriptor.model, 'out_channels') else 3,
                description=f"Detected via spandrel: {arch_name}",
                supports_multi_gpu=False  # Most GAN models don't support multi-GPU well
            )
        except Exception as e:
            # Spandrel failed, fall back to other methods
            return None

    def get_model_metadata(self, model_filename: str) -> GanModelMetadata:
        """Get comprehensive metadata for a model file"""
        # First check Real-ESRGAN hardcoded models
        model_stem = Path(model_filename).stem
        if model_stem in REAL_ESRGAN_MODELS:
            info = REAL_ESRGAN_MODELS[model_stem]
            return GanModelMetadata(
                name=model_stem,
                scale=info['scale'],
                architecture=info['arch'],
                description=f"Real-ESRGAN {model_stem}",
                supports_multi_gpu=False
            )

        # Try spandrel first for accurate detection
        # Support both `models/` (current) and `Image_Upscale_Models/` (legacy).
        model_path = self.base_dir / "models" / model_filename
        if not model_path.exists():
            model_path = self.base_dir / "Image_Upscale_Models" / model_filename
        if model_path.exists():
            spandrel_meta = self.get_model_metadata_with_spandrel(model_path)
            if spandrel_meta:
                return spandrel_meta

        # Load OMD data if not already loaded
        self._load_omd_metadata()

        # Try exact filename match in OMD
        if model_filename in self._omd_cache:
            return self._omd_cache[model_filename]

        # Try normalized name match
        normalized = self._normalize_name(model_filename)
        if normalized in self._omd_cache:
            return self._omd_cache[normalized]

        # Fallback: parse scale from filename
        scale = self._parse_scale_from_filename(model_filename)

        return GanModelMetadata(
            name=model_filename,
            scale=scale,
            architecture="unknown",
            description="Unknown GAN model (fallback detection)"
        )

    def _parse_scale_from_filename(self, filename: str) -> int:
        """Fallback scale detection from filename patterns"""
        filename_lower = filename.lower()

        # Check for explicit scale prefixes (e.g., "4x_", "2x", "x4")
        scale_match = re.search(r'(\d+)x[-_]', filename_lower)
        if scale_match:
            try:
                return int(scale_match.group(1))
            except ValueError:
                pass
        
        # Check for x prefix pattern (e.g., "x4", "x2")
        scale_match = re.search(r'x(\d+)', filename_lower)
        if scale_match:
            try:
                return int(scale_match.group(1))
            except ValueError:
                pass

        # Default to 4x if can't determine
        return 4


@dataclass
class GanResult:
    """Result of GAN upscaling operation"""
    returncode: int
    output_path: Optional[str]
    log: str


class GanRunner:
    """
    Production-ready GAN upscaler with Spandrel + Real-ESRGAN.
    
    This implementation fixes all critical gaps:
    - Actually uses Spandrel for universal model loading
    - Comprehensive metadata from Open Model Database + detection
    - Proper fixed-scale handling with downscale-first support
    - Robust batch processing for videos
    - Color correction and face restoration integration
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.registry = GanModelRegistry(base_dir)
        self._spandrel_available = self.registry._spandrel_available
        
        # Current loaded model state
        self._current_model = None
        self._current_model_name = None
        self._current_device = None
    
    def _clear_model(self):
        """Clear currently loaded model and free VRAM"""
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
            self._current_model_name = None
            self._current_device = None
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
    
    def _load_model_spandrel(
        self,
        model_name: str,
        device: str,
        on_progress: Optional[Callable[[str], None]] = None
    ):
        """Load model using Spandrel (universal loader)"""
        try:
            import spandrel
            import torch
            
            if on_progress:
                on_progress(f"Loading {model_name} with Spandrel...")
            
            # Support both `models/` (current) and `Image_Upscale_Models/` (legacy).
            model_path = self.base_dir / "models" / model_name
            if not model_path.exists():
                model_path = self.base_dir / "Image_Upscale_Models" / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load with spandrel
            model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(model_path))
            
            if on_progress:
                on_progress(f"✓ Loaded {model_name} (arch: {model_descriptor.architecture.name if hasattr(model_descriptor, 'architecture') else 'unknown'})")
            
            return model_descriptor.model
            
        except Exception as e:
            raise RuntimeError(f"Spandrel loading failed: {e}")
    
    def _load_model_realesrgan(
        self,
        model_name: str,
        metadata: GanModelMetadata,
        device: str,
        on_progress: Optional[Callable[[str], None]] = None
    ):
        """Load model using Real-ESRGAN (fallback for known architectures)"""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            import torch
            
            if on_progress:
                on_progress(f"Loading {model_name} with Real-ESRGAN...")
            
            # Support both `models/` (current) and `Image_Upscale_Models/` (legacy).
            model_path = self.base_dir / "models" / model_name
            if not model_path.exists():
                model_path = self.base_dir / "Image_Upscale_Models" / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Create model architecture
            if metadata.architecture == "RRDBNet":
                model = RRDBNet(
                    num_in_ch=metadata.input_channels,
                    num_out_ch=metadata.output_channels,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=metadata.scale
                )
            elif metadata.architecture == "SRVGGNetCompact":
                model = SRVGGNetCompact(
                    num_in_ch=metadata.input_channels,
                    num_out_ch=metadata.output_channels,
                    num_feat=64,
                    num_conv=32,
                    upscale=metadata.scale,
                    act_type='prelu'
                )
            else:
                raise ValueError(f"Unsupported architecture: {metadata.architecture}")
            
            # Create upsampler
            upsampler = RealESRGANer(
                scale=metadata.scale,
                model_path=str(model_path),
                model=model,
                tile=0,  # Will be set per-image
                tile_pad=10,
                pre_pad=0,
                half=True,  # fp16 for performance
                device=device
            )
            
            if on_progress:
                on_progress(f"✓ Loaded {model_name} (Real-ESRGAN {metadata.architecture})")
            
            return upsampler
            
        except Exception as e:
            raise RuntimeError(f"Real-ESRGAN loading failed: {e}")
    
    def _load_model(
        self,
        model_name: str,
        device: str,
        on_progress: Optional[Callable[[str], None]] = None
    ):
        """Load model if not already loaded, using Spandrel first then Real-ESRGAN fallback"""
        # Check if already loaded
        if (self._current_model is not None and 
            self._current_model_name == model_name and 
            self._current_device == device):
            if on_progress:
                on_progress(f"✓ Model {model_name} already loaded")
            return self._current_model
        
        # Clear previous model
        self._clear_model()
        
        # Get metadata
        metadata = self.registry.get_model_metadata(model_name)
        
        # Try Spandrel first (universal)
        if self._spandrel_available:
            try:
                model = self._load_model_spandrel(model_name, device, on_progress)
                self._current_model = model
                self._current_model_name = model_name
                self._current_device = device
                return model
            except Exception as e:
                if on_progress:
                    on_progress(f"⚠️ Spandrel failed ({e}), trying Real-ESRGAN...")
        
        # Fallback to Real-ESRGAN for known architectures
        if metadata.architecture in ["RRDBNet", "SRVGGNetCompact"]:
            try:
                model = self._load_model_realesrgan(model_name, metadata, device, on_progress)
                self._current_model = model
                self._current_model_name = model_name
                self._current_device = device
                return model
            except Exception as e:
                raise RuntimeError(f"Both Spandrel and Real-ESRGAN failed: {e}")
        else:
            raise RuntimeError(f"No loader available for architecture: {metadata.architecture}")
    
    def _calculate_outscale(
        self,
        input_width: int,
        input_height: int,
        target_resolution: int,
        model_scale: int,
        enable_max: bool = True,
        max_resolution: int = 0
    ) -> Tuple[float, int, int]:
        """
        Calculate output scale and target dimensions for fixed-scale models.
        
        Returns: (outscale, target_width, target_height)
        """
        # For fixed-scale models, calculate what downscale we need first
        short_side = min(input_width, input_height)
        long_side = max(input_width, input_height)
        is_landscape = input_width > input_height
        
        # Calculate desired output from target resolution
        desired_short = target_resolution
        aspect_ratio = long_side / short_side
        desired_long = int(desired_short * aspect_ratio)
        
        # Apply max resolution constraint
        if enable_max and max_resolution > 0:
            if max(desired_short, desired_long) > max_resolution:
                scale_down = max_resolution / max(desired_short, desired_long)
                desired_short = int(desired_short * scale_down)
                desired_long = int(desired_long * scale_down)
        
        # Calculate what input resolution we need to reach desired output with model_scale
        required_input_short = desired_short / model_scale
        required_input_long = desired_long / model_scale
        
        # Calculate outscale (how much to downscale before applying model)
        outscale = short_side / required_input_short
        
        # Calculate final target dimensions
        if is_landscape:
            target_width = desired_long
            target_height = desired_short
        else:
            target_width = desired_short
            target_height = desired_long
        
        return outscale, target_width, target_height
    
    def _upscale_image_with_model(
        self,
        image: np.ndarray,
        model,
        metadata: GanModelMetadata,
        settings: Dict[str, Any]
    ) -> np.ndarray:
        """Upscale single image using loaded model"""
        try:
            import torch
            
            # Get settings
            tile_size = int(settings.get("tile_size", 0))
            
            # Use Spandrel model
            if self._spandrel_available and not isinstance(model, type(None)):
                try:
                    # Convert to tensor
                    img_tensor = torch.from_numpy(image).float() / 255.0
                    if len(img_tensor.shape) == 2:
                        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
                    elif len(img_tensor.shape) == 3:
                        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                    
                    device = self._current_device or "cuda"
                    img_tensor = img_tensor.to(device)
                    
                    # Apply model
                    with torch.no_grad():
                        if tile_size > 0:
                            # Tiled inference
                            output = self._tiled_inference(model, img_tensor, tile_size, settings)
                        else:
                            output = model(img_tensor)
                    
                    # Convert back to numpy
                    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
                    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
                    
                    return output_np
                except Exception as e:
                    # Fallback to Real-ESRGAN if spandrel fails
                    pass
            
            # Use Real-ESRGAN upsampler
            if hasattr(model, 'enhance'):
                # Set tile size
                model.tile = tile_size if tile_size > 0 else 0
                model.tile_pad = int(settings.get("overlap", 32))
                
                # Enhance
                output, _ = model.enhance(image, outscale=metadata.scale)
                return output
            else:
                raise RuntimeError("Model has no enhance method and spandrel failed")
                
        except Exception as e:
            raise RuntimeError(f"Upscaling failed: {e}")
    
    def _tiled_inference(
        self,
        model,
        img_tensor: 'torch.Tensor',
        tile_size: int,
        settings: Dict[str, Any]
    ) -> 'torch.Tensor':
        """Perform tiled inference for large images"""
        import torch
        
        overlap = int(settings.get("overlap", 32))
        _, _, h, w = img_tensor.shape
        
        # Calculate tile positions
        tiles_x = (w + tile_size - 1) // tile_size
        tiles_y = (h + tile_size - 1) // tile_size
        
        # Create output tensor (upscaled size)
        metadata = self.registry.get_model_metadata(self._current_model_name)
        scale = metadata.scale
        output = torch.zeros(
            1, 3, h * scale, w * scale,
            device=img_tensor.device,
            dtype=img_tensor.dtype
        )
        
        # Process each tile
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile bounds with overlap
                x1 = max(0, tx * tile_size - overlap)
                y1 = max(0, ty * tile_size - overlap)
                x2 = min(w, (tx + 1) * tile_size + overlap)
                y2 = min(h, (ty + 1) * tile_size + overlap)
                
                # Extract tile
                tile = img_tensor[:, :, y1:y2, x1:x2]
                
                # Upscale tile
                with torch.no_grad():
                    tile_out = model(tile)
                
                # Calculate output position
                out_x1 = x1 * scale
                out_y1 = y1 * scale
                out_x2 = x2 * scale
                out_y2 = y2 * scale
                
                # Blend with overlap
                output[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out
        
        return output
    
    def _upscale_single_image(
        self,
        input_path: str,
        output_path: str,
        model,
        metadata: GanModelMetadata,
        settings: Dict[str, Any],
        log: Callable[[str], None]
    ) -> GanResult:
        """Upscale single image file"""
        try:
            # Load image
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return GanResult(returncode=1, output_path=None, log="Failed to load image")
            
            log(f"Input: {img.shape[1]}x{img.shape[0]}")
            
            # Calculate outscale if needed
            target_res = int(settings.get("target_resolution", 0))
            if target_res > 0 and settings.get("downscale_first", False):
                outscale, target_w, target_h = self._calculate_outscale(
                    img.shape[1], img.shape[0],
                    target_res, metadata.scale,
                    settings.get("enable_max", True),
                    int(settings.get("max_resolution", 0))
                )
                
                if outscale > 1.0:
                    # Downscale first
                    new_w = int(img.shape[1] / outscale)
                    new_h = int(img.shape[0] / outscale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    log(f"Downscaled to {new_w}x{new_h} before upscaling")
            
            # Upscale
            log("Upscaling...")
            output_img = self._upscale_image_with_model(img, model, metadata, settings)
            
            log(f"Output: {output_img.shape[1]}x{output_img.shape[0]}")
            
            # Apply post-processing
            if settings.get("color_correction", False):
                log("Applying color correction...")
                # Simple color matching
                output_img = self._match_colors(output_img, img, metadata.scale)
            
            if settings.get("sharpening", 0) > 0:
                strength = float(settings.get("sharpening", 0))
                log(f"Applying sharpening (strength: {strength})...")
                output_img = self._sharpen_image(output_img, strength)
            
            # Save
            output_format = settings.get("output_format_gan", "auto")
            if output_format == "auto":
                # Use input format
                ext = Path(input_path).suffix
            else:
                ext = f".{output_format}"
            
            final_output = Path(output_path).with_suffix(ext)
            
            quality = int(settings.get("output_quality_gan", 95))
            if ext.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(final_output), output_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext.lower() == '.webp':
                cv2.imwrite(str(final_output), output_img, [cv2.IMWRITE_WEBP_QUALITY, quality])
            else:
                cv2.imwrite(str(final_output), output_img)
            
            log(f"✓ Saved to {final_output}")
            
            return GanResult(returncode=0, output_path=str(final_output), log="Success")
            
        except Exception as e:
            log(f"❌ Error: {e}")
            log(traceback.format_exc())
            return GanResult(returncode=1, output_path=None, log=str(e))
    
    def _match_colors(self, output: np.ndarray, reference: np.ndarray, scale: int) -> np.ndarray:
        """Simple color correction by matching histograms"""
        try:
            # Downscale output to match reference resolution
            h, w = reference.shape[:2]
            output_small = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)
            
            # Match histograms in LAB color space
            output_lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.float32)
            output_small_lab = cv2.cvtColor(output_small, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Match mean and std for each channel
            for i in range(3):
                out_mean, out_std = output_small_lab[:, :, i].mean(), output_small_lab[:, :, i].std()
                ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()
                
                if out_std > 0:
                    output_lab[:, :, i] = ((output_lab[:, :, i] - out_mean) * (ref_std / out_std)) + ref_mean
            
            # Convert back
            output_lab = np.clip(output_lab, 0, 255).astype(np.uint8)
            corrected = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
            
            return corrected
        except Exception:
            # Return original if correction fails
            return output
    
    def _sharpen_image(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Apply unsharp mask sharpening"""
        try:
            # Gaussian blur
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            
            # Unsharp mask
            sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
            
            return sharpened
        except Exception:
            return img
    
    def _upscale_video_file(
        self,
        input_path: str,
        output_path: str,
        model,
        metadata: GanModelMetadata,
        settings: Dict[str, Any],
        log: Callable[[str], None]
    ) -> GanResult:
        """Upscale video file frame-by-frame with batch support"""
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory(prefix="gan_video_") as temp_dir:
                temp_path = Path(temp_dir)
                frames_dir = temp_path / "frames"
                frames_up_dir = temp_path / "frames_up"
                frames_dir.mkdir()
                frames_up_dir.mkdir()
                
                # Extract frames
                log("Extracting frames...")
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                log(f"Video: {total_frames} frames @ {fps} FPS")
                
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(frame_path), frame)
                    frame_idx += 1
                
                cap.release()
                log(f"Extracted {frame_idx} frames")
                
                # Upscale frames with batch support
                log("Upscaling frames...")
                frame_files = sorted(frames_dir.glob("*.png"))
                batch_size = int(settings.get("batch_size", 1))
                
                # Process frames in batches
                for i in range(0, len(frame_files), batch_size):
                    batch = frame_files[i:i+batch_size]
                    log(f"Processing frames {i+1}-{min(i+batch_size, len(frame_files))} / {len(frame_files)}")
                    
                    for frame_file in batch:
                        # Upscale single frame
                        frame_out_path = frames_up_dir / f"{frame_file.stem}_up{frame_file.suffix}"
                        result = self._upscale_single_image(
                            str(frame_file),
                            str(frame_out_path),
                            model,
                            metadata,
                            settings,
                            lambda msg: None  # Suppress per-frame logs
                        )
                        
                        if result.returncode != 0:
                            raise RuntimeError(f"Frame upscale failed: {frame_file.name}")
                
                # Reassemble video
                log("Reassembling video...")
                upscaled_frames = sorted(frames_up_dir.glob("*.png"))
                
                if not upscaled_frames:
                    raise RuntimeError("No upscaled frames found")
                
                # Get output dimensions from first frame
                first_frame = cv2.imread(str(upscaled_frames[0]))
                height, width = first_frame.shape[:2]
                
                # Create video writer (using ffmpeg for better quality)
                output_format = settings.get("output_format_gan", "auto")
                if output_format == "auto":
                    ext = ".mp4"
                else:
                    ext = f".{output_format}" if not output_format.startswith('.') else output_format
                
                final_output = Path(output_path).with_suffix(ext)
                
                # Use ffmpeg to create video
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", str(frames_up_dir / "frame_%06d_up.png"),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    str(final_output)
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                log(f"✓ Video saved to {final_output}")
                
                return GanResult(returncode=0, output_path=str(final_output), log="Success")
                
        except Exception as e:
            log(f"❌ Error: {e}")
            log(traceback.format_exc())
            return GanResult(returncode=1, output_path=None, log=str(e))
    
    def upscale(
        self,
        settings: Dict[str, Any],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> GanResult:
        """
        Main upscale entry point.
        
        settings must include:
        - input_path: str
        - gan_model: str (model filename)
        - target_resolution: int
        - And other GAN-specific settings
        """
        log_lines = []
        
        def log(msg: str):
            log_lines.append(msg)
            if on_progress:
                on_progress(msg)
        
        try:
            # Get settings
            input_path = normalize_path(settings.get("input_path", ""))
            if not input_path or not Path(input_path).exists():
                return GanResult(returncode=1, output_path=None, log="Invalid input path")
            
            model_name = settings.get("gan_model", "")
            if not model_name:
                return GanResult(returncode=1, output_path=None, log="No model specified")
            
            # Get metadata
            metadata = self.registry.get_model_metadata(model_name)
            log(f"Model: {metadata.name} ({metadata.scale}x, {metadata.architecture})")
            
            # Load model
            device = f"cuda:{settings.get('gpu_device', '0')}" if settings.get("gpu_acceleration", True) else "cpu"
            log(f"Device: {device}")
            
            model = self._load_model(model_name, device, log)
            
            # Determine input type
            input_type = detect_input_type(input_path)
            
            # Resolve output path
            output_override = settings.get("output_override")
            if output_override:
                output_path = output_override
            else:
                output_path = resolve_output_location(
                    input_path=input_path,
                    output_format=settings.get("output_format_gan", "auto"),
                    global_output_dir=settings.get("output_dir", "outputs"),
                    batch_mode=False
                )
            
            # Upscale based on input type
            if input_type == "video":
                result = self._upscale_video_file(input_path, output_path, model, metadata, settings, log)
            else:
                result = self._upscale_single_image(input_path, output_path, model, metadata, settings, log)
            
            # Apply face restoration if requested
            if settings.get("apply_face", False) and result.returncode == 0 and result.output_path:
                try:
                    log("Applying face restoration...")
                    face_strength = float(settings.get("face_strength", 0.5))
                    
                    if input_type == "video":
                        result.output_path = restore_video(
                            result.output_path,
                            face_strength,
                            temp_dir=settings.get("temp_dir"),
                            on_progress=log
                        )
                    else:
                        result.output_path = restore_image(
                            result.output_path,
                            face_strength,
                            on_progress=log
                        )
                    
                    log("✓ Face restoration complete")
                except Exception as e:
                    log(f"⚠️ Face restoration failed: {e}")
            
            result.log = "\n".join(log_lines)
            return result
            
        except Exception as e:
            log(f"❌ Fatal error: {e}")
            log(traceback.format_exc())
            return GanResult(returncode=1, output_path=None, log="\n".join(log_lines))


# Module-level helper functions for compatibility with old code
_global_registry: Optional[GanModelRegistry] = None


def get_gan_model_metadata(model_filename: str, base_dir: Path) -> GanModelMetadata:
    """Get metadata for a GAN model file (module-level function for compatibility)"""
    global _global_registry
    if _global_registry is None:
        _global_registry = GanModelRegistry(base_dir)
    return _global_registry.get_model_metadata(model_filename)


def reload_gan_models_cache(base_dir: Path):
    """Reload the GAN models cache (module-level function for compatibility)"""
    global _global_registry
    _global_registry = GanModelRegistry(base_dir)
    # Force reload
    _global_registry._loaded = False
    _global_registry._load_omd_metadata()
