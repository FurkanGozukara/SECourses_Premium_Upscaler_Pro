"""
Comparison Manager for Pin Reference and Enhanced Comparison Features

Provides:
- Pin reference persistence across sessions
- Fullscreen comparison mode
- Difference overlay visualization
- Side-by-side comparison layouts
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np


class ComparisonManager:
    """
    Manager for comparison features including pinned references.
    
    Features:
    - Pin reference images/videos for iterative comparison
    - Persist pinned references across sessions
    - Generate difference overlays
    - Manage comparison cache
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize comparison manager.
        
        Args:
            cache_dir: Directory to store pinned references and cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pin_file = self.cache_dir / "pinned_reference.json"
        self.current_pin: Optional[Dict[str, Any]] = None
        self._load_pin()
    
    def _load_pin(self):
        """Load pinned reference from disk"""
        if self.pin_file.exists():
            try:
                with self.pin_file.open("r") as f:
                    self.current_pin = json.load(f)
            except Exception:
                self.current_pin = None
    
    def _save_pin(self):
        """Save pinned reference to disk"""
        try:
            with self.pin_file.open("w") as f:
                json.dump(self.current_pin, f, indent=2)
        except Exception:
            pass
    
    def pin_reference(
        self,
        file_path: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Pin a reference file for comparison.
        
        Args:
            file_path: Path to reference file
            name: Optional name for this pin
            metadata: Optional metadata to store
            
        Returns:
            (success, message)
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        # Copy to cache
        pin_copy = self.cache_dir / f"pinned_{path.name}"
        try:
            import shutil
            shutil.copy2(path, pin_copy)
        except Exception as e:
            return False, f"Failed to copy reference: {e}"
        
        self.current_pin = {
            "path": str(pin_copy),
            "original_path": str(path),
            "name": name or path.stem,
            "metadata": metadata or {},
            "type": "video" if path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"] else "image"
        }
        
        self._save_pin()
        
        return True, f"✅ Pinned reference: {self.current_pin['name']}"
    
    def get_pinned_reference(self) -> Optional[Dict[str, Any]]:
        """
        Get current pinned reference.
        
        Returns:
            Dictionary with pin info or None if no pin
        """
        if self.current_pin and Path(self.current_pin["path"]).exists():
            return self.current_pin
        return None
    
    def unpin_reference(self) -> Tuple[bool, str]:
        """
        Unpin current reference.
        
        Returns:
            (success, message)
        """
        if not self.current_pin:
            return False, "No reference pinned"
        
        # Delete cached file
        try:
            Path(self.current_pin["path"]).unlink(missing_ok=True)
        except Exception:
            pass
        
        name = self.current_pin.get("name", "reference")
        self.current_pin = None
        self._save_pin()
        
        return True, f"✅ Unpinned: {name}"
    
    def create_difference_overlay(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None,
        amplify: float = 3.0
    ) -> Optional[str]:
        """
        Create difference overlay visualization.
        
        Args:
            image1_path: First image (original)
            image2_path: Second image (upscaled)
            output_path: Optional output path (auto-generated if None)
            amplify: Amplification factor for differences (1.0 = raw diff, higher = more visible)
            
        Returns:
            Path to difference overlay image, or None if failed
        """
        try:
            # Load images
            img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
            
            if img1 is None or img2 is None:
                return None
            
            # Resize img1 to match img2 if different sizes
            if img1.shape != img2.shape:
                img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate absolute difference
            diff = cv2.absdiff(img1, img2).astype(np.float32)
            
            # Amplify differences
            diff = np.clip(diff * amplify, 0, 255).astype(np.uint8)
            
            # Create heatmap for better visualization
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            
            # Blend with original for context
            alpha = 0.6
            overlay = cv2.addWeighted(img2, 1 - alpha, diff_colored, alpha, 0)
            
            # Add difference stats
            mean_diff = diff_gray.mean()
            max_diff = diff_gray.max()
            
            # Add text overlay with stats
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Mean Diff: {mean_diff:.1f} | Max Diff: {max_diff}"
            cv2.putText(overlay, text, (20, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, text, (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Save
            if output_path is None:
                output_path = str(Path(image2_path).with_stem(f"{Path(image2_path).stem}_diff"))
            
            cv2.imwrite(output_path, overlay)
            
            return output_path
            
        except Exception:
            return None


# Global instance for easy access
_global_manager: Optional[ComparisonManager] = None


def get_comparison_manager(cache_dir: Optional[Path] = None) -> ComparisonManager:
    """Get or create global comparison manager"""
    global _global_manager
    
    if _global_manager is None:
        if cache_dir is None:
            cache_dir = Path.cwd() / "temp" / "comparison_cache"
        _global_manager = ComparisonManager(cache_dir)
    
    return _global_manager

