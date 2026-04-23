"""
main.py
=======
Real-Time Smart Traffic Monitoring System
-----------------------------------------
Entry point supporting three modes:

  1. VIDEO   – process a saved video file
  2. WEBCAM  – live stream from camera
  3. IMAGE   – single image inference

Usage examples
--------------
# Process a video file:
    python main.py --source data/sample_images/sample_traffic.mp4

# Use webcam (index 0):
    python main.py --source 0

# Single image:
    python main.py --source data/sample_images/traffic.jpg

# Custom confidence threshold:
    python main.py --source video.mp4 --conf 0.45

# No display window (headless / server):
    python main.py --source video.mp4 --no-display

Run  `python main.py --help`  for full option list.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ── ensure src/ is importable ──────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from config import (
    DETECT_DIR, REPORT_DIR, CONF_THRESH, IOU_THRESH,
    SHOW_DISPLAY, SAVE_OUTPUT, MODEL_PATH,
)
from detector    import TrafficDetector
from visualizer  import (
    plot_class_distribution,
    plot_density_pie,
    plot_vehicle_timeline,
    generate_summary_report,
)
from utils import (
    ensure_dirs, timestamp_str,
    get_video_writer, log_detection_csv,
    download_sample_video,
)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Smart Traffic Monitoring System (YOLOv8)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to video/image file, or webcam index (0,1,…). "
             "Leave blank to auto-download a sample traffic video.",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESH,
        help=f"Detection confidence threshold (default: {CONF_THRESH})",
    )
    parser.add_argument(
        "--iou", type=float, default=IOU_THRESH,
        help=f"NMS IoU threshold (default: {IOU_THRESH})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device: 'cpu', 'cuda', or 'mps'",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable pop-up video window (headless mode)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do NOT save annotated output video",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip CSV logging",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = process all)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════
def process_image(detector: TrafficDetector, image_path: str, show: bool, save: bool):
    """Run detection on a single image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    print(f"[INFO] Processing image: {image_path}")
    detections = detector.detect(frame)
    summary    = detector.get_summary(detections)
    annotated  = detector.annotate(frame, detections, summary)

    print(f"\n{'─'*40}")
    print(f"  Vehicles   : {summary['vehicle_count']}")
    print(f"  Pedestrians: {summary['pedestrian_count']}")
    print(f"  Density    : {summary['density_label']}")
    print(f"  Classes    : {summary['class_counts']}")
    print(f"{'─'*40}\n")

    if save:
        stem     = Path(image_path).stem
        out_path = os.path.join(DETECT_DIR, f"{stem}_detected_{timestamp_str()}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Annotated image saved → {out_path}")

        # Charts
        plot_class_distribution(summary["class_counts"])

    if show:
        cv2.imshow("Smart Traffic Monitor", annotated)
        print("[INFO] Press any key to close the window …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
# VIDEO / WEBCAM MODE
# ══════════════════════════════════════════════════════════════
def process_video(
    detector   : TrafficDetector,
    source     : str | int,
    show       : bool,
    save       : bool,
    log_csv    : bool,
    max_frames : int,
):
    """Run detection on each frame of a video or webcam stream."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # ── output video writer ──────────────────────────────────
    writer = None
    if save:
        out_path = os.path.join(
            DETECT_DIR,
            f"output_{timestamp_str()}.mp4",
        )
        writer = get_video_writer(out_path, cap)
        print(f"[INFO] Recording output → {out_path}")

    # ── session accumulators ─────────────────────────────────
    frame_no        = 0
    vehicle_timeline = []
    density_counts   = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    class_totals     = defaultdict(int)
    total_vehicles   = 0
    total_pedestrians= 0
    fps_timer        = time.time()

    print("[INFO] Starting detection loop … press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / video.")
            break

        # ── detect ──────────────────────────────────────────
        detections = detector.detect(frame)
        summary    = detector.get_summary(detections)
        annotated  = detector.annotate(frame, detections, summary)

        frame_no         += 1
        v_cnt             = summary["vehicle_count"]
        p_cnt             = summary["pedestrian_count"]
        total_vehicles   += v_cnt
        total_pedestrians+= p_cnt
        vehicle_timeline.append(v_cnt)
        density_counts[summary["density_label"]] += 1
        for cls, cnt in summary["class_counts"].items():
            class_totals[cls] += cnt

        # ── FPS overlay ──────────────────────────────────────
        elapsed = time.time() - fps_timer
        fps_val = 1.0 / elapsed if elapsed > 0 else 0
        fps_timer = time.time()
        cv2.putText(
            annotated,
            f"FPS: {fps_val:.1f}  Frame: {frame_no}",
            (annotated.shape[1] - 220, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
        )

        # ── log CSV ──────────────────────────────────────────
        if log_csv:
            log_detection_csv(frame_no, summary)

        # ── display ──────────────────────────────────────────
        if show:
            cv2.imshow("Smart Traffic Monitor  [Q = quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User quit.")
                break

        # ── write frame ──────────────────────────────────────
        if writer:
            writer.write(annotated)

        # ── progress ─────────────────────────────────────────
        if frame_no % 30 == 0:
            print(
                f"  Frame {frame_no:5d} | "
                f"Vehicles: {v_cnt:3d} | "
                f"Density: {summary['density_label']}"
            )

        if max_frames and frame_no >= max_frames:
            print(f"[INFO] Reached max-frames limit ({max_frames}).")
            break

    # ── cleanup ──────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── end-of-session analytics ─────────────────────────────
    session_stats = {
        "total_frames"     : frame_no,
        "total_vehicles"   : total_vehicles,
        "total_pedestrians": total_pedestrians,
        "avg_vehicle_count": total_vehicles / max(frame_no, 1),
        "peak_vehicle_count": max(vehicle_timeline, default=0),
        "density_counts"    : density_counts,
        "class_totals"      : dict(class_totals),
    }

    generate_summary_report(session_stats)

    if vehicle_timeline:
        plot_vehicle_timeline(vehicle_timeline)
    if any(density_counts.values()):
        plot_density_pie(density_counts)
    if class_totals:
        plot_class_distribution(dict(class_totals))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    ensure_dirs()

    # ── resolve source ───────────────────────────────────────
    source = args.source
    if source is None:
        print("[INFO] No source provided – downloading sample traffic video …")
        source = download_sample_video()

    # Numeric string → webcam index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    show = (not args.no_display) and SHOW_DISPLAY
    save = (not args.no_save)    and SAVE_OUTPUT
    log  = not args.no_log

    # ── init detector ────────────────────────────────────────
    detector = TrafficDetector(
        model_path = MODEL_PATH,
        conf       = args.conf,
        iou        = args.iou,
        device     = args.device,
    )

    # ── route to mode ────────────────────────────────────────
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS:
        process_image(detector, source, show, save)
    else:
        process_video(detector, source, show, save, log, args.max_frames)

    print("\n[INFO] Done. Check outputs/ for saved results.")


if __name__ == "__main__":
    main()
