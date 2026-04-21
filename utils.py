#!/usr/bin/env python3
"""Utility functions for video processing."""

import cv2
import torch
from pathlib import Path


def get_video_info(video_path):
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info


def check_gpu():
    """Check GPU status and memory."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("✗ No GPU available - will use CPU")
        return False


def format_video_info(info):
    """Pretty print video info."""
    return f"""
Video Information:
  Resolution: {info['width']}x{info['height']}
  FPS: {info['fps']}
  Total Frames: {info['total_frames']}
  Duration: {info['duration_seconds']:.1f} seconds
"""


def validate_video(video_path):
    """Validate video file."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".flv"}:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return path
