#!/usr/bin/env python3
"""
Aerial Drone Footage Detection Pipeline
Uses YOLO-World (open-vocabulary) to detect everything in drone footage:
buildings, trees, vehicles, people, roads, sports fields — anything.
"""

import cv2
import cvzone
import math
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLOWorld

# ── Aerial / campus drone classes ─────────────────────────────────────────────
AERIAL_CLASSES = [
    # People
    "person", "pedestrian", "crowd",
    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle", "van",
    # Structures
    "building", "rooftop", "house", "tower", "stadium", "warehouse",
    # Nature
    "tree", "bush", "grass", "lawn", "garden", "vegetation", "forest",
    # Infrastructure
    "road", "pathway", "sidewalk", "bridge", "parking lot", "driveway",
    # Campus / landmarks
    "sports field", "basketball court", "tennis court", "swimming pool",
    "fountain", "construction site",
    # Other aerial
    "shadow", "fence", "wall",
]

# ── Color per category group (BGR) ────────────────────────────────────────────
def get_color(label: str) -> tuple:
    label = label.lower()
    if any(w in label for w in ["person", "pedestrian", "crowd"]):
        return (0, 255, 255)        # yellow
    if any(w in label for w in ["car", "truck", "bus", "motor", "bicycle", "van"]):
        return (255, 50, 50)        # blue
    if any(w in label for w in ["building", "rooftop", "house", "tower", "stadium", "warehouse"]):
        return (50, 50, 255)        # red
    if any(w in label for w in ["tree", "bush", "grass", "lawn", "garden", "veg", "forest"]):
        return (50, 200, 50)        # green
    if any(w in label for w in ["road", "path", "side", "bridge", "parking", "drive"]):
        return (200, 200, 50)       # cyan
    return (255, 50, 255)           # magenta for everything else


def load_model(confidence=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLO-World (open-vocabulary) on", device)
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(AERIAL_CLASSES)
    model.to(device)
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Detecting {len(AERIAL_CLASSES)} aerial classes")
    return model


def process_frame(img, results, thickness=2):
    total = 0
    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf  = math.ceil(box.conf[0] * 100) / 100
            cls   = int(box.cls[0])
            label = names[cls] if cls < len(names) else str(cls)
            color = get_color(label)

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # cvzone label
            cvzone.putTextRect(
                img,
                f"{label} {conf}",
                (max(0, x1), max(20, y1)),
                scale=0.7,
                thickness=1,
                colorR=color,
                colorT=(255, 255, 255),
                offset=4,
            )
            total += 1
    return img, total


def process_video(model, input_path, output_path, confidence=0.1):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap    = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nInput : {input_path.name}")
    print(f"  {width}x{height} @ {fps:.0f}fps  |  {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_n   = 0
    det_total = 0
    class_counts = {}

    with tqdm(total=total, desc="Detecting") as pbar:
        while True:
            success, img = cap.read()
            if not success:
                break

            results = model(img, stream=True, conf=confidence, verbose=False)
            img, n  = process_frame(img, results)

            # Tally classes seen
            for r in results:
                for box in r.boxes:
                    cls   = int(box.cls[0])
                    label = r.names.get(cls, str(cls))
                    class_counts[label] = class_counts.get(label, 0) + 1

            out.write(img)
            det_total += n
            frame_n   += 1
            pbar.set_postfix(dets=det_total)
            pbar.update(1)

    cap.release()
    out.release()

    print(f"✓ Saved : {output_path}")
    print(f"  {frame_n} frames  |  {det_total} total detections")
    if class_counts:
        print("  Breakdown:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<20} {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Aerial Drone Detection – YOLO-World open-vocabulary"
    )
    parser.add_argument("input_video",  help="Input video path")
    parser.add_argument("--output",     "-o", default="output_detected.mp4")
    parser.add_argument("--confidence", "-c", type=float, default=0.1,
                        help="Detection confidence (default 0.1 for aerial)")
    args = parser.parse_args()

    model = load_model(args.confidence)
    process_video(model, args.input_video, args.output, args.confidence)


if __name__ == "__main__":
    main()
