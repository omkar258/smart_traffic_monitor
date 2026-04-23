# Real-Time Smart Traffic Monitoring System
### Using YOLOv8 + Deep Learning | Python | OpenCV

---

## Project Overview

This system detects and counts vehicles, pedestrians, and traffic signals in
**real time** from any video stream (file, webcam, or RTSP camera) using the
state-of-the-art **YOLOv8** object detection model. It estimates **traffic
density** (Low / Medium / High) and generates analytics charts automatically.

---

## Project Structure

```
smart_traffic_monitor/
├── main.py              ← unified entry point (video / webcam / image)
├── demo_image.py        ← quick synthetic-scene demo (no video needed)
├── evaluate.py          ← batch evaluation on a folder of files
├── requirements.txt     ← pip dependencies
│
├── src/
│   ├── config.py        ← all tunable parameters
│   ├── detector.py      ← YOLOv8 inference + annotation engine
│   ├── visualizer.py    ← matplotlib charts & CSV reports
│   └── utils.py         ← helpers: resize, download, CSV log
│
├── data/
│   └── sample_images/   ← put your test images/videos here
│
├── models/              ← downloaded YOLO weights are stored here
│
└── outputs/
    ├── detections/      ← annotated frames / output videos
    └── reports/         ← charts, CSV logs, session report
```

---

## Features

| Feature | Details |
|---|---|
| Object Detection | YOLOv8n (nano, fast) – upgradeable to s / m / l |
| Detected Classes | Car, Truck, Bus, Motorcycle, Bicycle, Person, Traffic Light |
| Traffic Density | LOW (0-5 vehicles) / MEDIUM (6-15) / HIGH (>15) |
| Input Sources | Video file, Webcam, Single image |
| HUD Overlay | Live counters + density label on every frame |
| Analytics | Bar chart, Pie chart, Vehicle timeline plot |
| CSV Logging | Per-frame detection log |
| Output Video | Annotated MP4 saved automatically |

---

## Quick Setup (5 Minutes)

### Step 1 — Clone / Copy the project

```
D:\deep_learning\smart_traffic_monitor\
```

### Step 2 — Create & activate a virtual environment (recommended)

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (NVIDIA):** If you have a CUDA GPU, install the matching `torch` first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

### Step 4 — Run the quick demo

```bash
python demo_image.py
```

This downloads `yolov8n.pt` (~6 MB) automatically on first run, then runs
inference on a synthetic traffic scene and shows the result.

---

## How to Run

### Process a video file

```bash
python main.py --source data/sample_images/sample_traffic.mp4
```

### Use webcam

```bash
python main.py --source 0
```

### Single image

```bash
python main.py --source data/sample_images/traffic.jpg
```

### Headless / server (no display window)

```bash
python main.py --source video.mp4 --no-display
```

### All CLI options

```
--source       Path to file or webcam index (default: auto-download sample)
--conf         Confidence threshold 0–1    (default: 0.40)
--iou          NMS IoU threshold 0–1       (default: 0.45)
--device       cpu | cuda | mps           (default: auto-detect)
--no-display   Disable OpenCV window
--no-save      Do not write output video
--no-log       Disable CSV logging
--max-frames   Stop after N frames
```

---

## Expected Output

After running on a traffic video you will see:

- **OpenCV window** – live annotated video with bounding boxes and HUD
- `outputs/detections/output_<timestamp>.mp4` – full annotated video
- `outputs/reports/session_report.txt` – text summary
- `outputs/reports/class_distribution.png` – bar chart
- `outputs/reports/density_pie.png` – density pie chart
- `outputs/reports/vehicle_timeline.png` – vehicle count over time
- `outputs/reports/detections_log.csv` – per-frame CSV

---

## Sample Output Description

```
Vehicle count HUD (top-left):
  Vehicles   : 12
  Pedestrians:  3
  Density    : MEDIUM

Per-class breakdown (bottom-left):
  Car: 7   Truck: 3   Bus: 2   Person: 3
```

---

## Performance

| Hardware | Model | Approx FPS |
|---|---|---|
| CPU (i5 10th gen) | yolov8n | 8–12 |
| NVIDIA RTX 3060 | yolov8n | 60+ |
| NVIDIA RTX 3060 | yolov8s | 45+ |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| Video window not showing | Remove `--no-display` flag |
| CUDA OOM error | Use `--device cpu` |
| Low FPS on CPU | Use `yolov8n.pt` (already default) |

---

## Technologies Used

- **Python 3.10+**
- **Ultralytics YOLOv8**
- **PyTorch**
- **OpenCV**
- **Matplotlib / Seaborn**
- **COCO-pretrained weights** (80 classes, no training needed)
