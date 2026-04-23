"""
utils.py – Shared helper utilities
===================================
• generate_sample_video()  – create a synthetic traffic video (no internet)
• frame_resize()           – resize maintaining aspect ratio
• timestamp_str()          – formatted timestamp string
• ensure_dirs()            – create all output dirs
• log_detection_csv()      – append detection row to CSV
"""

import os
import cv2
import csv
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR, DETECT_DIR, REPORT_DIR, DATA_DIR


# ──────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required output directories."""
    for d in [OUTPUT_DIR, DETECT_DIR, REPORT_DIR, DATA_DIR,
              os.path.join(DATA_DIR, "sample_images")]:
        os.makedirs(d, exist_ok=True)
    print("[INFO] Output directories ready.")


# ──────────────────────────────────────────────────────────────
def timestamp_str() -> str:
    """Return a filesystem-safe timestamp  e.g. '2024-09-15_14-32-05'"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ──────────────────────────────────────────────────────────────
def frame_resize(frame, width: int = None, height: int = None):
    """
    Resize frame while preserving aspect ratio.
    Specify exactly one of width or height.
    """
    h, w = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is not None:
        scale = width / w
        new_wh = (width, int(h * scale))
    else:
        scale = height / h
        new_wh = (int(w * scale), height)
    return cv2.resize(frame, new_wh, interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────
def _draw_scene(canvas, vehicles, pedestrians):
    """Draw a synthetic road scene onto canvas (in-place)."""
    H, W = canvas.shape[:2]

    # sky
    canvas[:H // 3, :] = (200, 210, 230)

    # road surface
    canvas[H // 3:, :] = (78, 78, 78)

    # lane dividers
    cv2.line(canvas, (0, H // 3), (W, H // 3), (180, 180, 180), 2)
    for x in range(0, W, 60):
        cv2.line(canvas, (x, H // 2), (x + 35, H // 2), (0, 220, 220), 3)

    # vehicles
    for v in vehicles:
        x1, y1 = int(v["x"]), int(v["y"])
        x2, y2 = x1 + v["w"], y1 + v["h"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), v["color"], -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 2)
        # windshield
        cv2.rectangle(canvas,
                      (x1 + 8, y1 + 4),
                      (x2 - 8, y1 + max(4, v["h"] // 3)),
                      (200, 230, 255), -1)

    # pedestrians
    for p in pedestrians:
        px, py = int(p["x"]), int(p["y"])
        cv2.rectangle(canvas, (px, py), (px + 14, py + 38), (200, 175, 155), -1)
        cv2.circle(canvas, (px + 7, py - 8), 8, (215, 185, 160), -1)


def generate_sample_video(dest_dir: str = None, n_frames: int = 300) -> str:
    """
    Generate a synthetic 10-second, 30 FPS, 1280x720 traffic video using
    OpenCV only — NO internet connection required.

    Animated coloured rectangles represent vehicles and pedestrians to
    demonstrate the full detection pipeline.

    Returns the saved .mp4 file path.
    """
    if dest_dir is None:
        dest_dir = os.path.join(DATA_DIR, "sample_images")
    os.makedirs(dest_dir, exist_ok=True)

    out_path = os.path.join(dest_dir, "sample_traffic.mp4")
    if os.path.exists(out_path):
        print(f"[INFO] Sample video already present: {out_path}")
        return out_path

    W, H = 1280, 720
    FPS = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    random.seed(42)
    # (lane_y, width, height, BGR color, speed px/frame)
    SPECS = [
        (H // 3 + 20,  120, 60, (0,  80, 200), 4.5),
        (H // 3 + 20,  120, 60, (30, 60, 190), 3.0),
        (H // 3 + 100, 180, 80, (0,  40, 160), 2.5),
        (H // 3 + 100, 160, 75, (20, 20, 180), 5.0),
        (H // 3 + 20,  100, 55, (10, 90, 210), 6.0),
        (H // 3 + 100, 125, 58, (0,  60, 195), 3.8),
        (H // 3 + 180, 130, 62, (15, 50, 185), 4.2),
        (H // 3 + 180, 175, 82, (5,  30, 170), 2.0),
    ]

    vehicles = []
    for lane_y, vw, vh, color, spd in SPECS:
        vehicles.append({
            "x": -vw - random.randint(0, 600),
            "y": lane_y, "w": vw, "h": vh,
            "color": color, "speed": spd,
        })

    pedestrians = [
        {"x": random.randint(0, 400), "y": H // 3 + 260, "speed": 0.8},
        {"x": random.randint(500, 900), "y": H // 3 + 265, "speed": 0.5},
    ]

    print(f"[INFO] Generating synthetic traffic video ({n_frames} frames) …")
    for frame_no in range(n_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_scene(canvas, vehicles, pedestrians)

        for v in vehicles:
            v["x"] += v["speed"]
            if v["x"] > W + 20:
                v["x"] = -v["w"] - random.randint(50, 300)

        for p in pedestrians:
            p["x"] += p["speed"]
            if p["x"] > W + 20:
                p["x"] = -20.0

        writer.write(canvas)
        if (frame_no + 1) % 60 == 0:
            print(f"  {(frame_no+1)/n_frames*100:.0f}% ...", end="\r", flush=True)

    writer.release()
    print(f"\n[INFO] Sample video saved --> {out_path}")
    return out_path


def download_sample_video(dest_dir: str = None) -> str:
    """Backward-compatible alias -> generate_sample_video() (no download)."""
    return generate_sample_video(dest_dir)


# ──────────────────────────────────────────────────────────────
def log_detection_csv(frame_no: int, summary: dict, csv_path: str = None):
    """
    Append one row per frame to a CSV log file.

    Columns: frame, timestamp, vehicles, pedestrians, density
    """
    if csv_path is None:
        csv_path = os.path.join(REPORT_DIR, "detections_log.csv")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["frame", "timestamp", "vehicles",
                             "pedestrians", "density"])
        writer.writerow([
            frame_no,
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
            summary.get("vehicle_count", 0),
            summary.get("pedestrian_count", 0),
            summary.get("density_label", ""),
        ])


# ──────────────────────────────────────────────────────────────
def get_video_writer(out_path: str, cap: cv2.VideoCapture,
                     fps_override: float = None):
    """
    Create an OpenCV VideoWriter matching the input capture dimensions.

    Parameters
    ----------
    out_path     : output .mp4 file path
    cap          : the source VideoCapture object (to read w/h/fps)
    fps_override : use a fixed FPS instead of the source FPS
    """
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))
