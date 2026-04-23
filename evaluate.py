"""
evaluate.py
===========
Batch evaluation script:
  • Runs inference on every image/video inside data/sample_images/
  • Computes per-class detection summary across all files
  • Generates analytics charts in outputs/reports/

Usage:
    python evaluate.py
    python evaluate.py --source data/sample_images
"""

import os
import sys
import argparse
import cv2
from pathlib import Path
from collections import defaultdict

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from config     import DATA_DIR, DETECT_DIR
from detector   import TrafficDetector, MODEL_PATH
from visualizer import plot_class_distribution, plot_density_pie, generate_summary_report
from utils      import ensure_dirs, timestamp_str


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def evaluate(source_dir: str):
    ensure_dirs()
    detector = TrafficDetector(model_path=MODEL_PATH)

    files = list(Path(source_dir).iterdir())
    media = [f for f in files if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]

    if not media:
        print(f"[WARN] No image/video files found in {source_dir}")
        return

    print(f"[INFO] Found {len(media)} file(s) to evaluate.\n")

    total_frames      = 0
    total_vehicles    = 0
    total_pedestrians = 0
    class_totals      = defaultdict(int)
    density_counts    = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    vehicle_counts    = []

    for f in media:
        print(f"  ▸ {f.name}")
        if f.suffix.lower() in IMAGE_EXTS:
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            frames = [frame]
        else:
            cap = cv2.VideoCapture(str(f))
            frames = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                frames.append(fr)
            cap.release()

        for frame in frames:
            detections = detector.detect(frame)
            summary    = detector.get_summary(detections)
            total_frames      += 1
            total_vehicles    += summary["vehicle_count"]
            total_pedestrians += summary["pedestrian_count"]
            vehicle_counts.append(summary["vehicle_count"])
            density_counts[summary["density_label"]] += 1
            for cls, cnt in summary["class_counts"].items():
                class_totals[cls] += cnt

    session_stats = {
        "total_frames"      : total_frames,
        "total_vehicles"    : total_vehicles,
        "total_pedestrians" : total_pedestrians,
        "avg_vehicle_count" : total_vehicles / max(total_frames, 1),
        "peak_vehicle_count": max(vehicle_counts, default=0),
        "density_counts"    : density_counts,
        "class_totals"      : dict(class_totals),
    }

    generate_summary_report(session_stats)
    if class_totals:
        plot_class_distribution(dict(class_totals))
    if any(density_counts.values()):
        plot_density_pie(density_counts)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation on sample data")
    parser.add_argument(
        "--source", type=str,
        default=os.path.join(DATA_DIR, "sample_images"),
        help="Directory containing images/videos",
    )
    args = parser.parse_args()
    evaluate(args.source)


if __name__ == "__main__":
    main()
