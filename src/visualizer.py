"""
visualizer.py – Plotting utilities for traffic analytics
=========================================================
Generates:
  • Bar chart of detected class counts
  • Density pie chart
  • Per-frame vehicle count timeline
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all platforms)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import REPORT_DIR, COLOR_MAP


# ── helper: BGR → matplotlib RGB ──────────────────────────────
def _bgr_to_rgb(bgr):
    b, g, r = bgr
    return (r / 255, g / 255, b / 255)


# ──────────────────────────────────────────────────────────────
def plot_class_distribution(class_counts: dict, save_path: str = None):
    """Bar chart: number of detections per class (single frame or session total)."""
    if not class_counts:
        print("[WARN] No class counts to plot.")
        return

    labels  = list(class_counts.keys())
    values  = list(class_counts.values())
    colors  = [_bgr_to_rgb(COLOR_MAP.get(l, COLOR_MAP["default"])) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(val), ha="center", fontsize=11, fontweight="bold"
        )

    ax.set_title("Detection Distribution by Class", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Object Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(REPORT_DIR, "class_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved class distribution chart → {save_path}")


# ──────────────────────────────────────────────────────────────
def plot_density_pie(density_counts: dict, save_path: str = None):
    """
    Pie chart showing proportion of frames in each density category.

    density_counts : {"LOW": n, "MEDIUM": n, "HIGH": n}
    """
    palette = {
        "LOW":    "#27ae60",
        "MEDIUM": "#e67e22",
        "HIGH":   "#c0392b",
    }
    labels  = [k for k, v in density_counts.items() if v > 0]
    sizes   = [density_counts[k] for k in labels]
    colors  = [palette[k] for k in labels]

    if not sizes:
        print("[WARN] No density data to plot.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title("Traffic Density Distribution", fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(REPORT_DIR, "density_pie.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved density pie chart → {save_path}")


# ──────────────────────────────────────────────────────────────
def plot_vehicle_timeline(vehicle_counts_per_frame: list, save_path: str = None):
    """
    Line chart: vehicle count across frames (video timeline).

    vehicle_counts_per_frame : list of ints
    """
    if not vehicle_counts_per_frame:
        print("[WARN] No timeline data to plot.")
        return

    x = list(range(len(vehicle_counts_per_frame)))
    y = vehicle_counts_per_frame

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="#2980b9", linewidth=1.8, label="Vehicle Count")
    ax.fill_between(x, y, alpha=0.15, color="#2980b9")

    # Rolling average (window = 15)
    if len(y) >= 15:
        kernel = np.ones(15) / 15
        smooth = np.convolve(y, kernel, mode="same")
        ax.plot(x, smooth, color="#e74c3c", linewidth=1.5, linestyle="--", label="15-frame avg")

    # Density zones
    ax.axhline(y=5,  color="#27ae60", linewidth=1, linestyle=":", alpha=0.7, label="Low limit")
    ax.axhline(y=15, color="#e67e22", linewidth=1, linestyle=":", alpha=0.7, label="Medium limit")

    ax.set_title("Vehicle Count Over Time", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Frame Number", fontsize=11)
    ax.set_ylabel("Vehicle Count", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(REPORT_DIR, "vehicle_timeline.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved vehicle timeline → {save_path}")


# ──────────────────────────────────────────────────────────────
def generate_summary_report(session_stats: dict, save_path: str = None):
    """
    Print and save a plain-text session summary.

    session_stats keys:
        total_frames, total_vehicles, total_pedestrians,
        avg_vehicle_count, peak_vehicle_count,
        density_counts, class_totals
    """
    lines = [
        "=" * 55,
        "   SMART TRAFFIC MONITORING – SESSION REPORT",
        "=" * 55,
        f"  Total Frames Processed : {session_stats.get('total_frames', 0)}",
        f"  Total Vehicles Detected : {session_stats.get('total_vehicles', 0)}",
        f"  Total Pedestrians       : {session_stats.get('total_pedestrians', 0)}",
        f"  Avg Vehicles / Frame    : {session_stats.get('avg_vehicle_count', 0):.2f}",
        f"  Peak Vehicle Count      : {session_stats.get('peak_vehicle_count', 0)}",
        "",
        "  Traffic Density Breakdown:",
    ]
    for label, cnt in session_stats.get("density_counts", {}).items():
        lines.append(f"    {label:<8}: {cnt} frames")
    lines.append("")
    lines.append("  Class Totals:")
    for cls, cnt in sorted(session_stats.get("class_totals", {}).items()):
        lines.append(f"    {cls:<16}: {cnt}")
    lines.append("=" * 55)

    report_text = "\n".join(lines)
    print(report_text)

    if save_path is None:
        save_path = os.path.join(REPORT_DIR, "session_report.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report_text)
    print(f"[INFO] Session report saved → {save_path}")
