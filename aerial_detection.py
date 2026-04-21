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
    def __init__(self, model_size="x", confidence=0.45, device="cuda"):
        """
        Initialize detection pipeline.

        Args:
            model_size: YOLOv8 model size - 'n', 's', 'm', 'l', 'x', 'xl'
                       For A100, recommend 'x' or 'xl' for best accuracy
            confidence: Detection confidence threshold (0.0-1.0)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.confidence = confidence

        print(f"Loading YOLOv8-{model_size} model on {device}...")
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.model.to(device)

        # Check GPU memory
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def process_video(self, input_path, output_path, box_color=(255, 0, 0), thickness=2):
        """
        Process video and output annotated frames with bounding boxes.

        Args:
            input_path: Path to input video
            output_path: Path to save output video
            box_color: BGR color for bounding boxes (default: blue)
            thickness: Line thickness for boxes
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_path}")

        # Open video
        cap = cv2.VideoCapture(str(input_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nInput Video:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")

        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_count = 0
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection
                results = self.model(frame, conf=self.confidence, verbose=False)

                # Draw bounding boxes
                annotated_frame = self._draw_boxes(frame.copy(), results[0], box_color, thickness)

                # Write frame
                out.write(annotated_frame)
                frame_count += 1
                pbar.update(1)

        cap.release()
        out.release()

        print(f"\n✓ Output saved to: {output_path}")
        print(f"  Processed {frame_count} frames")

    def _draw_boxes(self, frame, result, color, thickness):
        """Draw bounding boxes on frame."""
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)

                # Draw blue bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Draw label
                label = f"{class_names[cls_id]} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]

                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 4),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

        return frame


def main():
    parser = argparse.ArgumentParser(
        description="Aerial Footage Object Detection"
    )
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output", "-o", default="output_detected.mp4", help="Output video path")
    parser.add_argument("--model", "-m", default="x", choices=["n", "s", "m", "l", "x", "xl"],
                       help="YOLOv8 model size (default: x for A100)")
    parser.add_argument("--confidence", "-c", type=float, default=0.45,
                       help="Detection confidence threshold")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AerialDetectionPipeline(
        model_size=args.model,
        confidence=args.confidence,
        device=args.device
    )

    # Process video
    pipeline.process_video(args.input_video, args.output)


if __name__ == "__main__":
    main()
