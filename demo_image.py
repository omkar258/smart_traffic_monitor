"""
demo_image.py
=============
Quick-start demo: generates a synthetic traffic scene and runs
the detector on it – no external video file or webcam required.

Run:
    python demo_image.py

Produces:
    outputs/detections/demo_detected.jpg
    outputs/reports/class_distribution.png
"""

import os
import sys
import cv2
import numpy as np

# ── ensure src/ importable ──────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from detector   import TrafficDetector
from visualizer import plot_class_distribution
from utils      import ensure_dirs
from config     import DETECT_DIR, MODEL_PATH


def make_synthetic_scene(width=1280, height=720):
    """
    Draw a simple road scene with coloured rectangles
    representing vehicles so the demo runs without a real image.
    The detector will still run on this frame; real detections
    are unlikely on simple rectangles, but the full pipeline runs end-to-end.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 80   # grey road

    # Sky
    frame[:height//3, :] = [200, 180, 120]

    # Road markings
    for x in range(0, width, 80):
        cv2.line(frame, (x, height//2), (x + 40, height//2), (255, 255, 255), 3)

    # Fake car / person shapes
    cars = [
        ((100,  350), (260, 450), (0,  100, 200), "Car"),
        ((320,  360), (500, 460), (0,   50, 180), "Car"),
        ((560,  340), (760, 455), (30,  80, 220), "Truck"),
        ((820,  355), (970, 450), (0,   80, 200), "Car"),
        ((1050, 360), (1200,450), (0,   60, 190), "Bus"),
        ((200,  500), (340, 590), (0,  120, 220), "Car"),
        ((150,  250), (180, 340), (200, 200, 200), "Person"),
        ((700,  245), (730, 340), (200, 200, 200), "Person"),
    ]
    for (x1, y1), (x2, y2), color, _ in cars:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return frame


def main():
    ensure_dirs()

    print("=" * 50)
    print("  DEMO — Smart Traffic Monitor (YOLOv8)")
    print("=" * 50)
    print("[INFO] Generating synthetic traffic scene …")

    frame    = make_synthetic_scene()
    detector = TrafficDetector(model_path=MODEL_PATH)

    print("[INFO] Running YOLOv8 detection …")
    detections = detector.detect(frame)
    summary    = detector.get_summary(detections)
    annotated  = detector.annotate(frame, detections, summary)

    # ── Save annotated image ─────────────────────────────────
    out_path = os.path.join(DETECT_DIR, "demo_detected.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"\n  Vehicles   : {summary['vehicle_count']}")
    print(f"  Pedestrians: {summary['pedestrian_count']}")
    print(f"  Density    : {summary['density_label']}")
    print(f"  Classes    : {summary['class_counts']}")
    print(f"\n[INFO] Annotated image saved → {out_path}")

    if summary["class_counts"]:
        plot_class_distribution(summary["class_counts"])

    # ── Display window (graceful fallback if GUI not available) ──────────
    try:
        cv2.imshow("Demo – Smart Traffic Monitor", annotated)
        print("[INFO] Press any key to exit …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("\n[WARN] cv2.imshow not available.")
        print("       This happens when opencv-python-headless is installed.")
        print("       Fix it with:")
        print("         pip uninstall opencv-python-headless -y")
        print("         pip install opencv-python")
        print(f"\n[OK]  Output image already saved → {out_path}")

    print("\n[INFO] Demo complete! Check outputs/ folder for results.")


if __name__ == "__main__":
    main()
