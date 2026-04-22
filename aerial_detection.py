#!/usr/bin/env python3
"""
Aerial Drone Detection Pipeline – Dual-Model + Tracking + NMS
YOLOv9e   → people, vehicles  (with ByteTrack for stable IDs across frames)
YOLO-World → buildings, trees, structures (open-vocabulary)
Cross-model IoU NMS removes duplicate boxes between the two models.
"""

import cv2
import cvzone
import math
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from ultralytics import YOLO, YOLOWorld

# ── Classes ───────────────────────────────────────────────────────────────────
COCO_KEEP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign",
}

WORLD_CLASSES = [
    "building", "rooftop", "tower", "house", "stadium", "warehouse",
    "construction site", "wall", "fence",
    "tree", "bush", "grass", "lawn", "vegetation",
    "road", "pathway", "sidewalk", "bridge", "parking lot",
    "sports field", "basketball court", "tennis court", "swimming pool",
    "fountain",
]

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
def get_color(label: str) -> tuple:
    l = label.lower()
    if any(w in l for w in ["person", "pedestrian", "bicycle"]):
        return (0, 220, 255)
    if any(w in l for w in ["car","truck","bus","motor","traffic","stop","van"]):
        return (255, 100, 0)
    if any(w in l for w in ["building","rooftop","tower","house","stadium","warehouse"]):
        return (60, 60, 255)
    if any(w in l for w in ["construct","wall","fence"]):
        return (0, 165, 255)
    if any(w in l for w in ["tree","bush","grass","lawn","veg"]):
        return (40, 190, 40)
    if any(w in l for w in ["road","path","side","bridge","parking","drive"]):
        return (190, 190, 0)
    if any(w in l for w in ["sport","court","field","pool","fountain"]):
        return (255, 0, 180)
    return (180, 180, 180)


# ── Cross-model IoU NMS ───────────────────────────────────────────────────────
def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0

def nms_merge(detections, iou_thresh=0.45):
    """Remove duplicates across both models using IoU."""
    if not detections:
        return detections
    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d[5], reverse=True)
    kept = []
    for det in detections:
        box = det[:4]
        if all(iou(box, k[:4]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


# ── Draw ──────────────────────────────────────────────────────────────────────
def draw_detection(img, x1, y1, x2, y2, label, conf, track_id=None):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = (max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2))
    if x2 <= x1 or y2 <= y1:
        return
    color = get_color(label)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    tag = f"#{track_id} {label} {conf}" if track_id else f"{label} {conf}"
    cvzone.putTextRect(
        img, tag,
        (max(0, x1), max(20, y1)),
        scale=0.6, thickness=1,
        colorR=color, colorT=(255,255,255),
        offset=3,
    )


# ── Models ────────────────────────────────────────────────────────────────────
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")

    print("Loading YOLOv9e  (people / vehicles + ByteTrack)...")
    m_coco = YOLO("yolov9e.pt")
    m_coco.to(device)

    print("Loading YOLO-World (buildings / structures / nature)...")
    m_world = YOLOWorld("yolov8x-worldv2.pt")
    m_world.set_classes(WORLD_CLASSES)
    m_world.to(device)
    return m_coco, m_world


# ── Per-frame processing ──────────────────────────────────────────────────────
def process_frame(img, m_coco, m_world, conf_people=0.25, conf_world=0.15):
    detections = []   # (x1,y1,x2,y2, label, conf, track_id|None)

    # Model 1 – YOLOv9e with ByteTrack (stable IDs, no flicker)
    track_results = m_coco.track(img, conf=conf_people, persist=True,
                                  tracker="bytetrack.yaml", verbose=False)
    for r in track_results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in COCO_KEEP:
                continue
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            tid  = int(box.id[0]) if box.id is not None else None
            detections.append((x1,y1,x2,y2, label, conf, tid))

    # Model 2 – YOLO-World (structures/nature)
    for r in m_world(img, conf=conf_world, verbose=False):
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            x1,y1,x2,y2 = (int(v) for v in box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            detections.append((x1,y1,x2,y2, label, conf, None))

    # Cross-model NMS — remove overlapping duplicates
    detections = nms_merge(detections)

    for x1,y1,x2,y2,label,conf,tid in detections:
        draw_detection(img, x1,y1,x2,y2, label, conf, tid)

    return img, len(detections), [d[4] for d in detections]


# ── Video loop ────────────────────────────────────────────────────────────────
def process_video(m_coco, m_world, input_path, output_path,
                  conf_people=0.25, conf_world=0.15):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap    = cv2.VideoCapture(str(input_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nInput : {input_path.name}  |  {width}x{height} @ {fps:.0f}fps  |  {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    det_total   = 0
    class_tally = Counter()

    with tqdm(total=total, desc="Detecting") as pbar:
        while True:
            success, img = cap.read()
            if not success:
                break
            img, n, labels = process_frame(img, m_coco, m_world, conf_people, conf_world)
            out.write(img)
            det_total += n
            class_tally.update(labels)
            pbar.set_postfix(dets=det_total)
            pbar.update(1)

    cap.release()
    out.release()

    print(f"✓ Saved : {output_path}  |  {det_total} total detections")
    print("  Breakdown:")
    for cls, cnt in class_tally.most_common():
        print(f"    {cls:<22} {cnt}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output",       "-o", default="output_detected.mp4")
    parser.add_argument("--conf-people",  "-p", type=float, default=0.25)
    parser.add_argument("--conf-world",   "-w", type=float, default=0.15)
    args = parser.parse_args()

    m_coco, m_world = load_models()
    process_video(m_coco, m_world, args.input_video, args.output,
                  args.conf_people, args.conf_world)


if __name__ == "__main__":
    main()
