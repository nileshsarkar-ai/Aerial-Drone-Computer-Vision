#!/usr/bin/env python3
"""Batch processing for multiple videos."""

import argparse
from pathlib import Path
from aerial_detection import load_models, process_video
from tqdm import tqdm


def process_directory(input_dir, output_dir, conf_people=0.15, conf_world=0.1):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
    videos = [f for f in input_dir.glob("*") if f.suffix.lower() in video_extensions]

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(videos)} video(s) to process\n")

    m_coco, m_world = load_models()

    for video_file in tqdm(videos, desc="Videos"):
        output_path = output_dir / f"detected_{video_file.name}"
        try:
            process_video(m_coco, m_world, video_file, output_path,
                          conf_people, conf_world)
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch aerial detection")
    parser.add_argument("input_dir")
    parser.add_argument("--output",      "-o", default="./output_videos")
    parser.add_argument("--conf-people", "-p", type=float, default=0.15)
    parser.add_argument("--conf-world",  "-w", type=float, default=0.1)
    args = parser.parse_args()
    process_directory(args.input_dir, args.output, args.conf_people, args.conf_world)
