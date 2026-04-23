"""
config.py – Central configuration file for Smart Traffic Monitoring System
All tunable parameters are declared here for easy modification.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
DETECT_DIR  = os.path.join(OUTPUT_DIR, "detections")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")

# ─────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────
MODEL_NAME  = "yolov8n.pt"          # nano (fast); swap to yolov8s/m/l for accuracy
MODEL_PATH  = os.path.join(MODEL_DIR, MODEL_NAME)
CONF_THRESH = 0.40                  # Minimum detection confidence
IOU_THRESH  = 0.45                  # NMS IoU threshold
IMG_SIZE    = 640                   # Inference resolution

# ─────────────────────────────────────────────
# DETECTION CLASSES  (COCO indices)
# ─────────────────────────────────────────────
# car=2, motorcycle=3, bus=5, truck=7, person=0, bicycle=1, traffic light=9
VEHICLE_CLASSES    = [2, 3, 5, 7]   # car, motorcycle, bus, truck
PEDESTRIAN_CLASSES = [0]            # person
ALL_CLASSES        = VEHICLE_CLASSES + PEDESTRIAN_CLASSES + [1, 9]

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    9: "Traffic Light",
}

# ─────────────────────────────────────────────
# TRAFFIC DENSITY THRESHOLDS (vehicle count)
# ─────────────────────────────────────────────
DENSITY_LOW    = 5    # 0–5  vehicles  → LOW
DENSITY_MEDIUM = 15   # 6–15 vehicles  → MEDIUM
                      # >15  vehicles  → HIGH

# ─────────────────────────────────────────────
# COLOURS  (BGR for OpenCV)
# ─────────────────────────────────────────────
COLOR_MAP = {
    "Person":        (0,   200, 255),   # amber
    "Bicycle":       (0,   255, 127),   # spring green
    "Car":           (0,   100, 255),   # orange
    "Motorcycle":    (255, 100,   0),   # azure
    "Bus":           (255,   0, 200),   # magenta-ish
    "Truck":         (60,  20, 220),    # crimson
    "Traffic Light": (0,   255, 255),   # yellow
    "default":       (200, 200, 200),   # grey
}

DENSITY_COLORS = {
    "LOW":    (0,   200,   0),   # green
    "MEDIUM": (0,   165, 255),   # orange
    "HIGH":   (0,     0, 220),   # red
}

# ─────────────────────────────────────────────
# VIDEO / WEBCAM SETTINGS
# ─────────────────────────────────────────────
DEFAULT_FPS   = 30
DISPLAY_SCALE = 1.0   # Resize display frame (1.0 = original)
SAVE_OUTPUT   = True  # Write annotated output video
SHOW_DISPLAY  = True  # Pop up OpenCV window

# ─────────────────────────────────────────────
# FONT SETTINGS
# ─────────────────────────────────────────────
FONT_SCALE     = 0.6
FONT_THICKNESS = 2
