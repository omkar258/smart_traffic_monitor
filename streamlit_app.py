"""
streamlit_app.py
================
Streamlit web interface for the Real-Time Smart Traffic Monitoring System.
Supports image upload and demo mode using a synthetic test frame.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st

# ── ensure src/ is importable ──────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from config import (
    CONF_THRESH, IOU_THRESH, MODEL_PATH,
    VEHICLE_CLASSES, PEDESTRIAN_CLASSES, CLASS_NAMES,
    DENSITY_LOW, DENSITY_MEDIUM,
)
from detector import TrafficDetector

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Traffic Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — dark, premium look
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1f2937, #111827);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: #8b949e;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Density badge */
.density-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    margin-top: 4px;
}
.density-LOW    { background: #14532d; color: #4ade80; border: 1px solid #16a34a; }
.density-MEDIUM { background: #78350f; color: #fb923c; border: 1px solid #ea580c; }
.density-HIGH   { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #374151 !important;
    border-radius: 12px;
    background: #161b22;
}

/* Divider */
hr { border-color: #21262d; }

/* Info box */
.info-box {
    background: #1f2937;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 12px 16px;
    color: #cbd5e1;
    font-size: 0.92rem;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🚦 Smart Traffic Monitor</h1>
    <p>Real-Time Vehicle Detection &amp; Traffic Density Analysis · Powered by YOLOv8</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — settings
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")

    conf_thresh = st.slider(
        "Confidence Threshold", 0.10, 0.90, CONF_THRESH, 0.05,
        help="Minimum confidence score to accept a detection"
    )
    iou_thresh = st.slider(
        "NMS IoU Threshold", 0.10, 0.90, IOU_THRESH, 0.05,
        help="Non-maximum suppression IoU threshold"
    )

    st.markdown("---")
    st.markdown("## 📊 Detection Classes")
    for cid, name in CLASS_NAMES.items():
        icon = "🚗" if cid in VEHICLE_CLASSES else ("🚶" if cid in PEDESTRIAN_CLASSES else "🚦")
        st.markdown(f"{icon} &nbsp; **{name}**", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    <div class="info-box">
    Built with <b>YOLOv8</b> + <b>OpenCV</b>.<br>
    Detects vehicles, pedestrians, and classifies traffic density in real-time.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# LOAD DETECTOR (cached)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Loading YOLOv8 model …")
def load_detector(conf, iou):
    return TrafficDetector(model_path=MODEL_PATH, conf=conf, iou=iou)


# ══════════════════════════════════════════════════════════════════════════
# HELPER: run detection on a BGR numpy frame, return annotated + summary
# ══════════════════════════════════════════════════════════════════════════
def run_detection(detector, frame_bgr):
    detections = detector.detect(frame_bgr)
    summary    = detector.get_summary(detections)
    annotated  = detector.annotate(frame_bgr, detections, summary)
    return annotated, summary, detections


# ══════════════════════════════════════════════════════════════════════════
# HELPER: render KPI metrics row
# ══════════════════════════════════════════════════════════════════════════
def render_metrics(summary, detections, elapsed_ms):
    density = summary["density_label"]
    badge_html = f'<span class="density-badge density-{density}">{density}</span>'

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🚗 Vehicles Detected", summary["vehicle_count"])
    with c2:
        st.metric("🚶 Pedestrians", summary["pedestrian_count"])
    with c3:
        st.metric("📦 Total Objects", len(detections))
    with c4:
        st.metric("⚡ Inference Time", f"{elapsed_ms:.0f} ms")

    st.markdown(f"**Traffic Density:** {badge_html}", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPER: class breakdown bar chart
# ══════════════════════════════════════════════════════════════════════════
def render_class_chart(class_counts):
    if not class_counts:
        return
    import pandas as pd
    df = pd.DataFrame(
        list(class_counts.items()), columns=["Class", "Count"]
    ).sort_values("Count", ascending=False)
    st.bar_chart(df.set_index("Class"), color="#38bdf8", height=220)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — input mode tabs
# ══════════════════════════════════════════════════════════════════════════
tab_image, tab_video, tab_demo = st.tabs(["📷 Image Upload", "🎥 Video Upload", "🧪 Demo Mode"])


# ──────────────────────────────────────────────────────────────────────────
# TAB 1: IMAGE UPLOAD
# ──────────────────────────────────────────────────────────────────────────
with tab_image:
    st.markdown("### Upload a Traffic Image")
    uploaded_img = st.file_uploader(
        "Drop a JPG / PNG image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
    )

    if uploaded_img:
        detector = load_detector(conf_thresh, iou_thresh)

        # Decode
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect
        t0 = time.perf_counter()
        annotated, summary, detections = run_detection(detector, frame_bgr)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Show images side by side
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_ann:
            st.markdown("**Annotated — YOLOv8 Detections**")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        render_metrics(summary, detections, elapsed_ms)

        st.markdown("#### 📊 Detected Class Breakdown")
        render_class_chart(summary["class_counts"])
    else:
        st.info("⬆️ Upload a traffic image to begin detection.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 2: VIDEO UPLOAD
# ──────────────────────────────────────────────────────────────────────────
with tab_video:
    st.markdown("### Upload a Traffic Video")
    st.warning(
        "⚠️ On Streamlit Cloud the free tier has limited memory. "
        "For best results, use a short clip (< 30 seconds)."
    )
    uploaded_vid = st.file_uploader(
        "Drop an MP4 / AVI video here",
        type=["mp4", "avi", "mov"],
        key="vid_uploader",
    )
    max_frames_ui = st.number_input(
        "Max frames to analyse (0 = all)", min_value=0, max_value=500, value=60, step=10
    )

    if uploaded_vid and st.button("▶️ Run Detection", key="run_vid"):
        detector = load_detector(conf_thresh, iou_thresh)

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = max_frames_ui if max_frames_ui > 0 else total_frames

        progress = st.progress(0, text="Analysing video …")
        frame_placeholder = st.empty()

        vehicle_timeline = []
        class_totals     = defaultdict(int)
        density_counts   = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        frame_no         = 0
        total_inf_ms     = 0

        while cap.isOpened() and frame_no < limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            t0 = time.perf_counter()
            annotated, summary, detections = run_detection(detector, frame)
            total_inf_ms += (time.perf_counter() - t0) * 1000

            vehicle_timeline.append(summary["vehicle_count"])
            density_counts[summary["density_label"]] += 1
            for cls, cnt in summary["class_counts"].items():
                class_totals[cls] += cnt

            # Show every 5th frame
            if frame_no % 5 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_no}/{limit}",
                    use_container_width=True,
                )
            progress.progress(min(frame_no / limit, 1.0), text=f"Frame {frame_no}/{limit}")

        cap.release()
        os.unlink(tmp_path)
        progress.empty()

        st.success(f"✅ Processed **{frame_no}** frames")
        st.markdown("---")

        # Summary metrics
        avg_vehicles = sum(vehicle_timeline) / max(len(vehicle_timeline), 1)
        peak_vehicles = max(vehicle_timeline, default=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎞️ Frames Analysed", frame_no)
        m2.metric("🚗 Avg Vehicles / Frame", f"{avg_vehicles:.1f}")
        m3.metric("📈 Peak Vehicle Count", peak_vehicles)
        m4.metric("⚡ Avg Inference", f"{total_inf_ms/max(frame_no,1):.0f} ms")

        st.markdown("#### 🚦 Traffic Density Distribution")
        import pandas as pd
        density_df = pd.DataFrame(
            list(density_counts.items()), columns=["Density", "Frames"]
        )
        st.bar_chart(density_df.set_index("Density"), color="#818cf8", height=220)

        st.markdown("#### 📊 Cumulative Class Breakdown")
        render_class_chart(dict(class_totals))

        st.markdown("#### 📉 Vehicle Count Timeline")
        timeline_df = pd.DataFrame({"Vehicles": vehicle_timeline})
        st.line_chart(timeline_df, color="#f472b6", height=220)

    elif not uploaded_vid:
        st.info("⬆️ Upload a traffic video to begin analysis.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 3: DEMO MODE (synthetic frame)
# ──────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("### 🧪 Demo — Synthetic Traffic Scene")
    st.markdown("""
    <div class="info-box">
    This mode generates a synthetic traffic-like image and runs YOLOv8 detection on it.
    Use it to verify the model is working correctly without uploading any file.
    </div>
    """, unsafe_allow_html=True)

    def make_synthetic_frame(w=960, h=540):
        """Draw a simple top-down road scene with coloured rectangles."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Sky gradient
        for y in range(h // 3):
            val = int(15 + y * 0.3)
            frame[y, :] = [val, val + 5, val + 20]
        # Road
        cv2.rectangle(frame, (0, h // 3), (w, h), (55, 55, 55), -1)
        # Lane markings
        for x in range(0, w, 80):
            cv2.rectangle(frame, (x, h // 2 - 4), (x + 40, h // 2 + 4), (200, 200, 0), -1)
        # "Cars" (orange boxes)
        cars = [(120, 200, 220, 320), (350, 220, 550, 340), (650, 190, 850, 310),
                (80, 380, 240, 460),  (420, 360, 600, 450), (720, 370, 900, 460)]
        for x1, y1, x2, y2 in cars:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 180), 2)
        # "Pedestrians" (amber boxes)
        peds = [(290, 300, 315, 360), (560, 290, 585, 355)]
        for x1, y1, x2, y2 in peds:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), -1)
        return frame

    if st.button("🚀 Run Demo Detection", key="run_demo"):
        detector = load_detector(conf_thresh, iou_thresh)
        demo_frame = make_synthetic_frame()

        t0 = time.perf_counter()
        annotated, summary, detections = run_detection(detector, demo_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Synthetic Scene**")
            st.image(cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_d2:
            st.markdown("**YOLOv8 Detections**")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        render_metrics(summary, detections, elapsed_ms)
        st.markdown("#### 📊 Class Breakdown")
        render_class_chart(summary["class_counts"])

        if not detections:
            st.info(
                "💡 No objects detected on the synthetic frame — "
                "this is expected since it uses simple rectangles. "
                "Try uploading a **real traffic image** in the Image tab!"
            )
    else:
        st.info("👆 Click **Run Demo Detection** to test the model instantly.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4b5563;font-size:0.85rem;'>"
    "🚦 Smart Traffic Monitor · YOLOv8 · OpenCV · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
