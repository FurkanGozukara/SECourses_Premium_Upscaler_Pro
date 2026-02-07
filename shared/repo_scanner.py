"""
Repository Scanner

Scans external repositories (SeedVR2, Real-ESRGAN, open-model-database) for:
- Recent commits and changelog
- New features that could be integrated
- Breaking changes or deprecations
- Model metadata and capabilities
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger("RepoScanner")


def scan_git_commits(repo_path: Path, count: int = 10) -> List[Dict[str, Any]]:
    """
    Scan recent git commits from a repository.
    
    Args:
        repo_path: Path to git repository
        count: Number of recent commits to retrieve
        
    Returns:
        List of commit dictionaries with hash, author, date, message
    """
    commits = []
    
    if not repo_path.exists() or not (repo_path / ".git").exists():
        logger.warning(f"Repository not found or not a git repo: {repo_path}")
        return commits
    
    try:
        # Use git log with JSON-like output format
        result = subprocess.run(
            [
                "git", "log",
                f"-{count}",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso"
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append({
                        "hash": parts[0][:8],  # Short hash
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4]
                    })
        
        logger.info(f"Scanned {len(commits)} commits from {repo_path.name}")
        
    except FileNotFoundError:
        logger.warning("git command not found - cannot scan commits")
    except subprocess.TimeoutExpired:
        logger.warning(f"Git log timed out for {repo_path}")
    except Exception as e:
        logger.error(f"Failed to scan git commits: {e}")
    
    return commits


def scan_seedvr2_repo(base_dir: Path) -> Dict[str, Any]:
    """
    Scan SeedVR2 repository for recent changes and features.
    
    Returns:
        Dict with commits, features, and notes
    """
    seedvr2_path = base_dir / "SeedVR2"
    
    result = {
        "path": str(seedvr2_path),
        "exists": seedvr2_path.exists(),
        "is_git_repo": (seedvr2_path / ".git").exists(),
        "commits": [],
        "features_found": [],
        "notes": []
    }
    
    if not result["exists"]:
        result["notes"].append("SeedVR2 repository not found")
        return result
    
    # Scan recent commits
    result["commits"] = scan_git_commits(seedvr2_path, count=10)
    
    # Scan for specific features by checking file structure
    cli_path = seedvr2_path / "inference_cli.py"
    if cli_path.exists():
        try:
            with open(cli_path, "r", encoding="utf-8", errors="ignore") as f:
                cli_content = f.read()
            
            # Look for key features in CLI
            if "--chunk_size" in cli_content:
                result["features_found"].append("Native streaming/chunking (--chunk_size)")
            if "--cache_dit" in cli_content:
                result["features_found"].append("DiT caching for speed")
            if "--cache_vae" in cli_content:
                result["features_found"].append("VAE caching")
            if "flash_attn" in cli_content or "flash-attn" in cli_content:
                result["features_found"].append("Flash Attention support")
            if "--compile" in cli_content:
                result["features_found"].append("Torch compile optimization")
            if "--blocks_to_swap" in cli_content:
                result["features_found"].append("BlockSwap memory optimization")
            if "--attention_mode" in cli_content:
                result["features_found"].append("Configurable attention backend")
            if "--temporal_overlap" in cli_content:
                result["features_found"].append("Temporal overlap for smoother video")
            if "--prepend_frames" in cli_content:
                result["features_found"].append("Frame prepending for stability")
            
            logger.info(f"SeedVR2 CLI analysis: {len(result['features_found'])} features detected")
            
        except Exception as e:
            logger.warning(f"Failed to analyze SeedVR2 CLI: {e}")
    
    return result


def scan_realesrgan_repo(base_dir: Path) -> Dict[str, Any]:
    """
    Scan Real-ESRGAN repository for GAN implementation patterns.
    
    Returns:
        Dict with available models, inference patterns, and features
    """
    realesrgan_path = base_dir / "Real-ESRGAN"
    
    result = {
        "path": str(realesrgan_path),
        "exists": realesrgan_path.exists(),
        "commits": [],
        "builtin_models": [],
        "features_found": []
    }
    
    if not result["exists"]:
        result["notes"] = ["Real-ESRGAN repository not found"]
        return result
    
    # Scan commits if git repo
    if (realesrgan_path / ".git").exists():
        result["commits"] = scan_git_commits(realesrgan_path, count=10)
    
    # Look for inference implementation
    inference_path = realesrgan_path / "inference_realesrgan.py"
    if inference_path.exists():
        try:
            with open(inference_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Extract builtin model names
            if "RealESRGANer" in content:
                result["features_found"].append("RealESRGANer inference class")
            if "tile" in content.lower():
                result["features_found"].append("Tile-based processing")
            if "face_enhance" in content or "face_restoration" in content:
                result["features_found"].append("Face enhancement support")
            if "fp16" in content or "half" in content:
                result["features_found"].append("FP16/half precision")
            
        except Exception as e:
            logger.warning(f"Failed to scan Real-ESRGAN inference: {e}")
    
    return result


def scan_open_model_database(base_dir: Path) -> Dict[str, Any]:
    """
    Scan open-model-database for GAN model metadata.
    
    Returns:
        Dict with available model metadata and architectures
    """
    omdb_path = base_dir / "open-model-database"
    
    result = {
        "path": str(omdb_path),
        "exists": omdb_path.exists(),
        "models_found": 0,
        "architectures": [],
        "sample_models": []
    }
    
    if not result["exists"]:
        return result
    
    # Scan data/models directory for JSON metadata
    models_dir = omdb_path / "data" / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.json"))
        result["models_found"] = len(model_files)
        
        # Sample first 10 models
        for model_file in model_files[:10]:
            try:
                with open(model_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                arch = metadata.get("architecture", "unknown")
                if arch not in result["architectures"]:
                    result["architectures"].append(arch)
                
                result["sample_models"].append({
                    "name": metadata.get("name", model_file.stem),
                    "architecture": arch,
                    "scale": metadata.get("scale", "unknown")
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse model metadata {model_file}: {e}")
    
    return result


def generate_repo_scan_report(base_dir: Path) -> str:
    """
    Generate comprehensive report of all scanned repositories.
    
    Args:
        base_dir: Base directory containing repositories
        
    Returns:
        Multi-line report string
    """
    report_lines = []
    report_lines.append("=== Repository Scan Report ===\n")
    
    # Scan SeedVR2
    seedvr2_scan = scan_seedvr2_repo(base_dir)
    report_lines.append("📦 SeedVR2 Repository:")
    report_lines.append(f"  Path: {seedvr2_scan['path']}")
    report_lines.append(f"  Status: {'✅ Found' if seedvr2_scan['exists'] else '❌ Not found'}")
    
    if seedvr2_scan["exists"]:
        report_lines.append(f"  Git: {'✅ Repository' if seedvr2_scan['is_git_repo'] else '⚠️ Not a git repo'}")
        
        if seedvr2_scan["commits"]:
            report_lines.append(f"\n  📝 Recent Commits ({len(seedvr2_scan['commits'])}):")
            for commit in seedvr2_scan["commits"][:5]:  # Show first 5
                report_lines.append(f"    • {commit['hash']} - {commit['message'][:80]}")
                report_lines.append(f"      {commit['date']} by {commit['author']}")
        
        if seedvr2_scan["features_found"]:
            report_lines.append(f"\n  🔧 Features Detected:")
            for feature in seedvr2_scan["features_found"]:
                report_lines.append(f"    ✅ {feature}")
    
    report_lines.append("\n")
    
    # Scan Real-ESRGAN
    realesrgan_scan = scan_realesrgan_repo(base_dir)
    report_lines.append("📦 Real-ESRGAN Repository:")
    report_lines.append(f"  Path: {realesrgan_scan['path']}")
    report_lines.append(f"  Status: {'✅ Found' if realesrgan_scan['exists'] else '❌ Not found'}")
    
    if realesrgan_scan["exists"] and realesrgan_scan["features_found"]:
        report_lines.append(f"  🔧 Features:")
        for feature in realesrgan_scan["features_found"]:
            report_lines.append(f"    ✅ {feature}")
    
    report_lines.append("\n")
    
    # Scan open-model-database
    omdb_scan = scan_open_model_database(base_dir)
    report_lines.append("📦 Open Model Database:")
    report_lines.append(f"  Path: {omdb_scan['path']}")
    report_lines.append(f"  Status: {'✅ Found' if omdb_scan['exists'] else '❌ Not found'}")
    
    if omdb_scan["exists"]:
        report_lines.append(f"  🧩 Models: {omdb_scan['models_found']} metadata files found")
        if omdb_scan["architectures"]:
            report_lines.append(f"  🏗️ Architectures: {', '.join(omdb_scan['architectures'][:10])}")
    
    return "\n".join(report_lines)
