"""
detector.py – Core YOLO-based object detector
=============================================
Wraps Ultralytics YOLOv8 and provides:
  • detect()        – run inference on a single frame / image
  • annotate()      – draw bounding boxes + labels on the frame
  • get_summary()   – return structured count & density info
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_PATH, MODEL_NAME, CONF_THRESH, IOU_THRESH, IMG_SIZE,
    VEHICLE_CLASSES, PEDESTRIAN_CLASSES, CLASS_NAMES,
    COLOR_MAP, DENSITY_COLORS,
    DENSITY_LOW, DENSITY_MEDIUM,
    FONT_SCALE, FONT_THICKNESS,
    MODEL_DIR,
)


class TrafficDetector:
    """
    Main detection engine.

    Parameters
    ----------
    model_path : str  – Path to a .pt weights file.
    conf       : float – Confidence threshold (0–1).
    iou        : float – NMS IoU threshold.
    device     : str  – 'cuda', 'cpu', or 'mps'.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf: float     = CONF_THRESH,
        iou: float      = IOU_THRESH,
        device: str     = None,
    ):
        # ── resolve device ──────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ── load model (downloads automatically if not found) ──
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"[INFO] Model not found locally – downloading {MODEL_NAME} …")
            os.makedirs(MODEL_DIR, exist_ok=True)
            # Ultralytics auto-downloads to cache; we then copy to models/
            self.model = YOLO(MODEL_NAME)
            # Save weight to our models/ directory
            self.model.save(str(model_file))
        else:
            print(f"[INFO] Loading model from {model_file}")
            self.model = YOLO(str(model_file))

        self.conf   = conf
        self.iou    = iou
        print(f"[INFO] Device : {self.device.upper()}")
        print(f"[INFO] Model  : {MODEL_NAME}  |  conf={conf}  iou={iou}")

    # ──────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on a single BGR frame (numpy array).

        Returns
        -------
        detections : list of dicts  – each dict has keys:
            class_id, class_name, confidence, bbox (x1,y1,x2,y2)
        """
        results = self.model.predict(
            source    = frame,
            conf      = self.conf,
            iou       = self.iou,
            imgsz     = IMG_SIZE,
            device    = self.device,
            verbose   = False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id  = int(box.cls[0])
            # Keep only classes we care about
            if cls_id not in (VEHICLE_CLASSES + PEDESTRIAN_CLASSES + [1, 9]):
                continue
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "class_id"   : cls_id,
                "class_name" : CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                "confidence" : round(conf_val, 3),
                "bbox"       : (x1, y1, x2, y2),
            })
        return detections

    # ──────────────────────────────────────────────────────────
    def get_summary(self, detections: list) -> dict:
        """
        Compute vehicle count, pedestrian count, and traffic density.

        Returns
        -------
        dict with keys: vehicle_count, pedestrian_count,
                        class_counts, density_label, density_color
        """
        vehicle_count    = 0
        pedestrian_count = 0
        class_counts     = {}

        for d in detections:
            name = d["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
            if d["class_id"] in VEHICLE_CLASSES:
                vehicle_count += 1
            elif d["class_id"] in PEDESTRIAN_CLASSES:
                pedestrian_count += 1

        # Density classification
        if vehicle_count <= DENSITY_LOW:
            density = "LOW"
        elif vehicle_count <= DENSITY_MEDIUM:
            density = "MEDIUM"
        else:
            density = "HIGH"

        return {
            "vehicle_count"    : vehicle_count,
            "pedestrian_count" : pedestrian_count,
            "class_counts"     : class_counts,
            "density_label"    : density,
            "density_color"    : DENSITY_COLORS[density],
        }

    # ──────────────────────────────────────────────────────────
    def annotate(self, frame: np.ndarray, detections: list, summary: dict) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and HUD overlay on the frame.

        Returns annotated frame (copy, original unchanged).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── 1. Draw bounding boxes ───────────────────────────
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            name  = d["class_name"]
            conf  = d["confidence"]
            color = COLOR_MAP.get(name, COLOR_MAP["default"])
            label = f"{name} {conf:.2f}"

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - lh - 6),
                (x1 + lw + 4, y1),
                color, -1,
            )
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
            )

        # ── 2. HUD overlay (top-left panel) ─────────────────
        density_color = summary["density_color"]
        hud_lines = [
            (f"Vehicles   : {summary['vehicle_count']}",    (220, 220, 220)),
            (f"Pedestrians: {summary['pedestrian_count']}", (220, 220, 220)),
            (f"Density    : {summary['density_label']}",    density_color),
        ]

        panel_x, panel_y = 10, 10
        panel_h = len(hud_lines) * 28 + 12
        panel_w = 280
        cv2.rectangle(
            annotated,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (30, 30, 30), -1,
        )
        cv2.rectangle(
            annotated,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (100, 100, 100), 1,
        )

        for i, (text, color) in enumerate(hud_lines):
            cv2.putText(
                annotated, text,
                (panel_x + 8, panel_y + 24 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
            )

        # ── 3. Class breakdown (bottom-left) ────────────────
        breakdown_y = h - 10
        for name, cnt in sorted(summary["class_counts"].items()):
            text  = f"{name}: {cnt}"
            color = COLOR_MAP.get(name, COLOR_MAP["default"])
            cv2.putText(
                annotated, text,
                (10, breakdown_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            breakdown_y -= 22

        return annotated
