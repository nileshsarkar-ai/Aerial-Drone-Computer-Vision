#!/usr/bin/env python3
"""Batch processing for multiple videos."""

import argparse
from pathlib import Path
from aerial_detection import load_model, process_video
from tqdm import tqdm


def process_directory(input_dir, output_dir, model_name="yolov9e", confidence=0.15):
    """Process all video files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
    videos = [f for f in input_dir.glob("*") if f.suffix.lower() in video_extensions]

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(videos)} video(s) to process\n")

    model = load_model(model_name)

    for video_file in tqdm(videos, desc="Processing Videos"):
        output_path = output_dir / f"detected_{video_file.name}"
        try:
            process_video(model, video_file, output_path, confidence)
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos")
    parser.add_argument("input_dir", help="Directory containing video files")
    parser.add_argument("--output", "-o", default="./output_videos",
                       help="Output directory")
    parser.add_argument("--model", "-m", default="yolov9e", help="YOLO model name")
    parser.add_argument("--confidence", "-c", type=float, default=0.45,
                       help="Detection confidence threshold")

    args = parser.parse_args()
    process_directory(args.input_dir, args.output, args.model, args.confidence)
