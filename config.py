#!/usr/bin/env python3
"""
config.py
=========
Single source of truth shared by the two-stage pipeline:

  Stage 1 : author_steps.py   (interactive: step count + names + rough times)
  Stage 2 : autosegment.py    (single pose pass -> exact times, trajectory,
                               rules, camera distance, contact sheet, bundle)

Edit the three PATHS for your machine. Everything else has sane defaults.
"""

import os
import cv2
import numpy as np

# ============================================================
# PATHS  — edit these for your machine
# ============================================================
# macOS mounts external drives under /Volumes. Run `ls /Volumes` to confirm.
VIDEOS_DIR      = "/Volumes/Akoode Technologies/Lakshya/m2 method/All videos"
ANNOTATIONS_DIR = "/Users/lakshya/Desktop/OfficeProjects/m2 trainer/annotations"
OUTPUT_DIR      = "/Users/lakshya/Desktop/OfficeProjects/m2 trainer/output"
TRAJECTORY_DIR  = os.path.join(OUTPUT_DIR, "trajectories")  # internal .npz copies

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")

# ============================================================
# POSE / PROCESSING KNOBS
# ============================================================
MODEL_COMPLEXITY = 2     # 2 = most accurate (offline). 1 is faster.
PROCESS_SIZE     = 720   # frames letterboxed to this square before pose.
FRAME_STRIDE     = 1     # 1 = every frame. 2-3 = faster, minor accuracy loss.
SMOOTH_WINDOW    = 9     # frames; temporal smoothing (odd number).

# ============================================================
# SEGMENTATION KNOBS
# ============================================================
MIN_STEP_SECONDS    = 1.5  # two boundaries can't be closer than this.
SNAP_WINDOW_SECONDS = 1.5  # rough times snap to the quietest frame within +/- this.

# ============================================================
# RULE-BUILDING KNOBS  (mirrors trainer.py)
# ============================================================
NUM_AUGMENTATIONS = 6      # augmented copies per pose -> widens tolerance bands.
USE_FLIP_AUG      = False  # horizontal flip; keep False for L/R-specific moves.

# ============================================================
# FLUTTER EXPORT KNOBS
# ============================================================
SCHEMA_VERSION = 1
EXPORT_HZ      = 10        # downsample the reference trajectory to this rate.
# Landmark indices exported to the app (keep small for on-device DTW).
# Nose + shoulders/elbows/wrists + hips/knees/ankles.
EXPORT_JOINTS  = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# MediaPipe Pose landmark indices used throughout.
L_SH, R_SH = 11, 12
L_HP, R_HP = 23, 24
# Joints whose movement defines "motion energy" (settle-point detection).
KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# ============================================================
# SHARED HELPERS
# ============================================================
def letterbox(frame, size=PROCESS_SIZE):
    """Resize to a square WITHOUT distorting aspect ratio (pads with black)."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def normalize(frame_landmarks):
    """
    frame_landmarks: (33,4) array [x,y,z,visibility].
    Returns (33,3) normalized coords: hip-centered, torso-scaled, so the
    reference is body-type invariant.
    """
    pts = frame_landmarks[:, :3].copy()
    hip_mid = (pts[L_HP] + pts[R_HP]) / 2.0
    sho_mid = (pts[L_SH] + pts[R_SH]) / 2.0
    torso = np.linalg.norm(sho_mid - hip_mid) + 1e-6
    return (pts - hip_mid) / torso


def moving_average(x, w=SMOOTH_WINDOW):
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def list_videos(videos_dir=None):
    """Sorted absolute paths of every video in videos_dir."""
    videos_dir = videos_dir or VIDEOS_DIR
    return sorted(
        os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )


def stem_of(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]
