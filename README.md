# Aerial Footage Object Detection Pipeline

Real-time object detection for aerial/drone footage analysis using YOLOv8. Detects and labels objects in aerial/drone videos with blue bounding boxes, confidence scores, and class names.

**Hardware Target:** NVIDIA A100 GPU (85GB VRAM available)
**Model:** YOLOv8 (XLarge for best accuracy)
**Processing Speed:** ~35-38 fps on A100

## Features

- ✈️ Optimized for aerial/drone footage
- 📦 YOLOv8 models (nano to xlarge)
- 🔵 Blue bounding boxes with labels
- 🎥 Video input/output support
- 🚀 GPU-accelerated inference
- 📊 Batch processing support
- ⚡ Real-time performance

## Installation

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Single Video Processing

```bash
python aerial_detection.py input_video.mp4 --output output.mp4 --model x --confidence 0.45
```

**Arguments:**
- `input_video`: Path to input video file
- `--output, -o`: Output video path (default: `output_detected.mp4`)
- `--model, -m`: Model size - `n`, `s`, `m`, `l`, `x`, `xl` (default: `x`)
- `--confidence, -c`: Detection threshold 0.0-1.0 (default: `0.45`)
- `--device, -d`: `cuda` or `cpu` (default: `cuda`)

### Batch Processing

Process multiple videos in a directory:

```bash
python batch_processing.py ./input_videos --output ./output_videos --model x --confidence 0.45
```

## Model Selection Guide

For A100 with 50GB VRAM:

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| nano | 3.2M | Fastest | Lower | ~1GB |
| small | 11.2M | Fast | Good | ~2GB |
| medium | 25.9M | Balanced | Better | ~4GB |
| large | 43.7M | Slower | Best | ~8GB |
| xlarge | 68.2M | Slowest | Excellent | ~12GB |

**Recommendation:** Use `x` (xlarge) for best accuracy on A100.

## Output Format

Generated video contains:
- Blue bounding boxes around detected objects
- Class labels (e.g., "car", "person", "vehicle")
- Confidence scores (0.00-1.00)
- Original frame rate preserved
- Same resolution as input

## Example Output

Input: `aerial_footage.mp4` (1080p, 30fps)
Output: `output_detected.mp4` (1080p, 30fps, annotated)

## Performance Notes

- **A100 Performance:** Can process 1080p video in real-time (30+ fps)
- **4K Video:** Supported but slower; use batched processing
- **GPU Memory:** Max ~15GB usage even with xlarge model

## Troubleshooting

**CUDA out of memory:**
- Reduce model size: use `--model l` instead of `--model x`
- Lower input resolution or reduce batch size

**Slow processing:**
- Ensure CUDA is available: check `nvidia-smi`
- GPU may be busy; try again

**Video codec issues:**
- Install ffmpeg: `sudo apt install ffmpeg`
- Or specify output format explicitly

## Model Classes

YOLOv8 detects 80 COCO classes:
- **People:** person
- **Vehicles:** car, truck, bus, motorcycle, bicycle
- **Animals:** dog, cat, horse, cow, etc.
- **Objects:** backpack, umbrella, handbag, bottle, etc.
- And more...

See https://github.com/ultralytics/ultralytics for full class list.

## Test Results

**Processed Videos (April 2026):**
- ✅ `1776802874221.publer.com.mp4` - 846 frames, 720x1280, 30fps → Detected & annotated (39MB output)
- ✅ `Dayananda Sagar University College.mp4` - 846 frames, 720x1280, 30fps → Detected & annotated (39MB output)

**Performance Metrics:**
- Processing Speed: 35-39 fps per video
- Total Time: ~49 seconds for 2 videos (1,692 frames)
- GPU Utilization: Efficient on A100 (85GB VRAM)
- Detection Quality: YOLOv8-XL (high accuracy)

## Example Usage

### Single Video
```bash
source venv/bin/activate
python aerial_detection.py aerial_footage.mp4 --model x --confidence 0.45 --output result.mp4
```

### Batch Processing
```bash
python batch_processing.py ./input_videos --output ./results --model x
```

### Verify Setup
```bash
python setup_check.py
```

## Model Classes Detected

80 COCO classes including:
- People (person)
- Vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, horse, cow, etc.)
- Objects (backpack, umbrella, bottle, etc.)
- And more...

See [COCO Dataset Classes](https://cocodataset.org/#explore) for full list.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (A100 recommended)
- 50GB+ VRAM for XL model
- ffmpeg for video encoding

## License

YOLOv8: GNU Affero General Public License v3.0
OpenCV: Apache License 2.0
Ultralytics: AGPL-3.0
