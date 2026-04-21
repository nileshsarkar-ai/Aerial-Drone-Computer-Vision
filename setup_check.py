#!/usr/bin/env python3
"""Setup verification script."""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    packages = [
        "cv2",
        "torch",
        "torchvision",
        "ultralytics",
        "numpy",
        "PIL",
        "tqdm",
    ]

    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - NOT INSTALLED")
            missing.append(pkg)

    return len(missing) == 0


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("\n✗ No GPU detected (CPU mode available)")
            return False
    except Exception as e:
        print(f"\n✗ GPU check failed: {e}")
        return False


def check_files():
    """Check if all required files exist."""
    required = [
        "aerial_detection.py",
        "batch_processing.py",
        "utils.py",
        "requirements.txt",
        "README.md",
        "config.yaml",
    ]

    all_exist = True
    for fname in required:
        if Path(fname).exists():
            print(f"✓ {fname}")
        else:
            print(f"✗ {fname} - MISSING")
            all_exist = False

    return all_exist


def main():
    print("=" * 50)
    print("Aerial Detection Setup Check")
    print("=" * 50)

    print("\n📦 Checking Dependencies...")
    deps_ok = check_dependencies()

    print("\n🎮 Checking GPU...")
    gpu_ok = check_gpu()

    print("\n📁 Checking Files...")
    files_ok = check_files()

    print("\n" + "=" * 50)
    if deps_ok and files_ok:
        print("✓ Setup Complete!")
        if gpu_ok:
            print("\nReady to process videos:")
            print("  python aerial_detection.py <video.mp4> --model x")
        else:
            print("\nGPU not available - CPU mode will be slower")
            print("  python aerial_detection.py <video.mp4> --device cpu")
    else:
        print("✗ Setup Incomplete")
        if not deps_ok:
            print("\nRun: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
