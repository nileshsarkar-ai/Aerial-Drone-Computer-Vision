#!/usr/bin/env python3
"""
Aerial Drone Detection Pipeline – Dual-Model
YOLOv9e   → people, vehicles (best at small moving objects)
YOLO-World → buildings, trees, roads, structures (open-vocabulary)
Both run every frame; results merged and drawn together.
"""

import cv2
import cvzone
import math
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from ultralytics import YOLO, YOLOWorld

# ── Classes each model handles ────────────────────────────────────────────────

# YOLOv9e: COCO classes we care about in aerial/drone footage
COCO_KEEP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign",
}

# YOLO-World: open-vocabulary aerial/campus classes
WORLD_CLASSES = [
    "building", "rooftop", "tower", "house", "stadium", "warehouse",
    "construction site", "wall", "fence",
    "tree", "bush", "grass", "lawn", "garden", "vegetation",
    "road", "pathway", "sidewalk", "bridge", "parking lot",
    "sports field", "basketball court", "tennis court", "swimming pool",
    "fountain", "shadow",
]

# ── Colors per category (BGR) ─────────────────────────────────────────────────
def get_color(label: str) -> tuple:
    l = label.lower()
    if any(w in l for w in ["person", "pedestrian"]):
        return (0, 255, 255)          # yellow
    if any(w in l for w in ["car","truck","bus","motor","bicycle","traffic","stop","van"]):
        return (255, 100, 0)          # blue
    if any(w in l for w in ["building","rooftop","tower","house","stadium","warehouse"]):
        return (50, 50, 255)          # red
    if any(w in l for w in ["construct","wall","fence"]):
        return (0, 165, 255)          # orange
    if any(w in l for w in ["tree","bush","grass","lawn","garden","veg"]):
        return (50, 200, 50)          # green
    if any(w in l for w in ["road","path","side","bridge","parking","drive"]):
        return (200, 200, 0)          # teal
    if any(w in l for w in ["sport","court","field","pool","fountain"]):
        return (255, 0, 200)          # magenta
    return (200, 200, 200)            # grey


def draw_detection(img, x1, y1, x2, y2, label, conf, thickness=2):
    color = get_color(label)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cvzone.putTextRect(
        img,
        f"{label} {conf}",
        (max(0, x1), max(20, y1)),
        scale=0.65,
        thickness=1,
        colorR=color,
        colorT=(255, 255, 255),
        offset=4,
    )


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}  |  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")

    print("Loading YOLOv9e  (people / vehicles)...")
    m1 = YOLO("yolov9e.pt")
    m1.to(device)

    print("Loading YOLO-World (buildings / trees / structures)...")
    m2 = YOLOWorld("yolov8x-worldv2.pt")
    m2.set_classes(WORLD_CLASSES)
    m2.to(device)

    return m1, m2


def process_frame(img, m_coco, m_world, conf_people=0.15, conf_world=0.1):
    detections = []

    # ── Model 1: YOLOv9e — people & vehicles ─────────────────────────────────
    for r in m_coco(img, stream=True, conf=conf_people, verbose=False):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in COCO_KEEP:
                continue
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            detections.append((x1, y1, x2, y2, label, conf))

    # ── Model 2: YOLO-World — structures / nature / infrastructure ───────────
    for r in m_world(img, stream=True, conf=conf_world, verbose=False):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            detections.append((x1, y1, x2, y2, label, conf))

    for x1, y1, x2, y2, label, conf in detections:
        draw_detection(img, x1, y1, x2, y2, label, conf)

    return img, len(detections), [d[4] for d in detections]


def process_video(m_coco, m_world, input_path, output_path,
                  conf_people=0.15, conf_world=0.1):
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

    det_total   = 0
    class_tally = Counter()

    with tqdm(total=total, desc="Detecting") as pbar:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, n, labels = process_frame(img, m_coco, m_world,
                                           conf_people, conf_world)
            out.write(img)
            det_total += n
            class_tally.update(labels)
            pbar.set_postfix(dets=det_total)
            pbar.update(1)

    cap.release()
    out.release()

    print(f"✓ Saved : {output_path}")
    print(f"  Frames: {total}  |  Total detections: {det_total}")
    print("  Class breakdown:")
    for cls, cnt in class_tally.most_common():
        print(f"    {cls:<22} {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Aerial Drone Detection – YOLOv9e + YOLO-World"
    )
    parser.add_argument("input_video")
    parser.add_argument("--output",        "-o", default="output_detected.mp4")
    parser.add_argument("--conf-people",   "-p", type=float, default=0.15)
    parser.add_argument("--conf-world",    "-w", type=float, default=0.1)
    args = parser.parse_args()

    m_coco, m_world = load_models()
    process_video(m_coco, m_world, args.input_video, args.output,
                  args.conf_people, args.conf_world)


if __name__ == "__main__":
    main()
