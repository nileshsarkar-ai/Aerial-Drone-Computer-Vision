#!/usr/bin/env python3
"""
Aerial Footage Object Detection Pipeline
Detects objects in aerial/drone footage and outputs annotated video with bounding boxes.
Optimized for A100 GPU with 50GB VRAM.
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import argparse


class AerialDetectionPipeline:
    def __init__(self, model_size="x", confidence=0.15, device="cuda"):
        self.device = device
        self.confidence = confidence

        print(f"Loading YOLOv8-{model_size} model on {device}...")
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.model.to(device)

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def process_video(self, input_path, output_path, box_color=(255, 0, 0), thickness=2):
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nInput: {input_path.name}")
        print(f"  Resolution: {width}x{height} @ {fps:.1f}fps  |  Frames: {total_frames}")

        # Use avc1 (H.264) for maximum player compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            # Fallback to mp4v if avc1 not available
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        total_dets = 0
        frame_count = 0

        with tqdm(total=total_frames, desc="Detecting") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.confidence, verbose=False)
                n_dets = len(results[0].boxes)
                total_dets += n_dets

                # Use YOLOv8 built-in annotation (boxes + labels, guaranteed)
                annotated = results[0].plot(
                    line_width=thickness,
                    font_size=0.6,
                    labels=True,
                    conf=True,
                    boxes=True,
                )

                # Tint boxes blue by boosting blue channel in box regions
                if n_dets > 0:
                    annotated = self._tint_boxes_blue(annotated, results[0])

                out.write(annotated)
                frame_count += 1
                pbar.set_postfix(dets=total_dets)
                pbar.update(1)

        cap.release()
        out.release()

        print(f"\n✓ Saved: {output_path}")
        print(f"  Frames: {frame_count}  |  Total detections: {total_dets}")

    def _tint_boxes_blue(self, frame, result):
        """Redraw box outlines in blue to match project spec."""
        if result.boxes is None or len(result.boxes) == 0:
            return frame
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)
        return frame


def main():
    parser = argparse.ArgumentParser(description="Aerial Footage Object Detection")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output", "-o", default="output_detected.mp4")
    parser.add_argument("--model", "-m", default="x", choices=["n", "s", "m", "l", "x", "xl"])
    parser.add_argument("--confidence", "-c", type=float, default=0.15,
                        help="Detection confidence (default: 0.15 for aerial footage)")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    pipeline = AerialDetectionPipeline(
        model_size=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    pipeline.process_video(args.input_video, args.output)


if __name__ == "__main__":
    main()
