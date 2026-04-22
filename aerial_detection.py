#!/usr/bin/env python3
"""
Aerial Footage Object Detection Pipeline
YOLOv9e model with cvzone labels and blue bounding boxes.
"""

import cv2
import cvzone
import math
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

BLUE = (255, 50, 50)   # BGR blue for boxes


def load_model(model_name="yolov9e"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = YOLO(f"{model_name}.pt")
    model.to(device)
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return model


def process_frame(img, results):
    """Draw blue boxes + cvzone labels on a single frame — same style as reference code."""
    total = 0
    for r in results:
        for box in r.boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class
            cls = int(box.cls[0])
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

            # Blue rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 2)

            # cvzone label tag
            cvzone.putTextRect(
                img,
                f"{name} {conf}",
                (max(0, x1), max(20, y1)),
                scale=0.8,
                thickness=1,
                colorR=BLUE,
                colorT=(255, 255, 255),
                offset=5,
            )
            total += 1
    return img, total


def process_video(model, input_path, output_path, confidence=0.15):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nInput : {input_path.name}")
    print(f"  {width}x{height} @ {fps:.0f}fps  |  {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_n = 0
    det_total = 0

    with tqdm(total=total, desc="Detecting") as pbar:
        while True:
            success, img = cap.read()
            if not success:
                break

            results = model(img, stream=True, conf=confidence, verbose=False)
            img, n = process_frame(img, results)

            out.write(img)
            det_total += n
            frame_n   += 1
            pbar.set_postfix(dets=det_total)
            pbar.update(1)

    cap.release()
    out.release()

    print(f"✓ Saved : {output_path}")
    print(f"  {frame_n} frames  |  {det_total} total detections")


def main():
    parser = argparse.ArgumentParser(description="Aerial Object Detection – YOLOv9e + cvzone")
    parser.add_argument("input_video", help="Input video path")
    parser.add_argument("--output",     "-o", default="output_detected.mp4")
    parser.add_argument("--model",      "-m", default="yolov9e",
                        help="YOLO model: yolov8n/s/m/l/x, yolov9c/e (default: yolov9e)")
    parser.add_argument("--confidence", "-c", type=float, default=0.15)
    args = parser.parse_args()

    model = load_model(args.model)
    process_video(model, args.input_video, args.output, args.confidence)


if __name__ == "__main__":
    main()
