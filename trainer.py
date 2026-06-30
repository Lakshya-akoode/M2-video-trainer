#!/usr/bin/env python3
"""
build_exercise_rules.py
=======================
Local (Mac/Linux) batch version of the Colab "step-by-step training" notebook.

What it does, for every video in VIDEOS_DIR that has a matching annotation:
  1. Runs MediaPipe Pose ONCE per video (extracts 33 landmarks + camera-distance z).
  2. Augments the landmark data to widen the tolerance bands.
  3. Slices the video into steps using a per-video annotation JSON.
  4. Computes joint angles + body metrics per step.
  5. Writes per-exercise outputs:
        <OUTPUT_DIR>/<video_name>/annotation.json
        <OUTPUT_DIR>/<video_name>/augmented_step_features.csv
        <OUTPUT_DIR>/<video_name>/augmented_step_validation_rules.json
        <OUTPUT_DIR>/<video_name>/ideal_camera_distance.json

IMPORTANT: step counts/timings are NOT auto-detected. Each video needs its own
annotation file (see sample_annotation.json). A video with no annotation is skipped.

Run:   python3 build_exercise_rules.py
"""

import os
import cv2
import json
import random
import traceback
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ============================================================
# CONFIG  — edit these three paths
# ============================================================
# Your external drive shows up under /Volumes on macOS.
VIDEOS_DIR      = "/Volumes/Akoode Technologies/Lakshya/m2 method/All videos"      # folder holding the 133 videos
ANNOTATIONS_DIR = "/Users/lakshya/Desktop/OfficeProjects/m2 trainer/annotations" # one <video_basename>.json per video
OUTPUT_DIR      = "/Users/lakshya/Desktop/OfficeProjects/m2 trainer/output"

# Processing knobs
FRAME_STRIDE      = 1      # 1 = every frame. Set 2 or 3 to go ~2-3x faster with minor accuracy loss.
MODEL_COMPLEXITY  = 2      # 2 = most accurate (offline). Use 1 if you want it faster.
PROCESS_SIZE      = 720    # frames are letterboxed (aspect-ratio preserved) to this square.
NUM_AUGMENTATIONS = 6      # augmented copies per pose. Controls how wide the tolerance bands get.
USE_FLIP_AUG      = False  # horizontal flip; leave False for left/right-specific exercises.
VIDEO_EXTENSIONS  = (".mp4", ".mov", ".m4v", ".avi", ".mkv")

mp_pose = mp.solutions.pose


# ============================================================
# 1. POSE EXTRACTION  (single pass: landmarks + camera distance)
# ============================================================
def letterbox(frame, size):
    """Resize to a square WITHOUT distorting aspect ratio (pads with black)."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def process_video(video_path):
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    poses, z_values = [], []
    count = 0

    torso_idx = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    ]

    pbar = tqdm(total=total_frames, desc="  poses", leave=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_STRIDE == 0:
            frame = letterbox(frame, PROCESS_SIZE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                landmarks = [v for p in lm for v in (p.x, p.y, p.z, p.visibility)]
                poses.append({
                    "frame_number": count,
                    "timestamp": count / fps,
                    "landmarks": landmarks,
                })
                z_values.append(float(np.mean([lm[i].z for i in torso_idx])))

        count += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    pose.close()

    camera_distance = None
    if z_values:
        camera_distance = {
            "ideal_camera_distance": {
                "min_z": round(float(np.min(z_values)), 4),
                "max_z": round(float(np.max(z_values)), 4),
                "mean_z": round(float(np.mean(z_values)), 4),
            }
        }

    return {
        "poses": poses,
        "fps": fps,
        "total_frames": total_frames,
        "camera_distance": camera_distance,
    }


# ============================================================
# 2. AUGMENTATION
# ============================================================
def augment_landmarks(landmarks, scale=1.0, rotation_deg=0, flip=False, noise_std=0.01):
    lm = np.array(landmarks).reshape(33, 4)
    coords = lm[:, :3]
    coords = coords * scale

    theta = np.radians(rotation_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    coords = coords.dot(rot.T)

    if flip:
        coords[:, 0] = -coords[:, 0]

    coords = coords + np.random.normal(0, noise_std, coords.shape)
    lm[:, :3] = coords
    return lm.flatten().tolist()


def augment_dataset(training_data, num_augmentations=6):
    poses = training_data["poses"]
    augmented = []
    for _ in range(num_augmentations):
        scale = random.uniform(0.95, 1.05)
        rotation = random.uniform(-10, 10)
        flip = random.choice([True, False]) if USE_FLIP_AUG else False
        noise_std = random.uniform(0.005, 0.02)
        for p in poses:
            augmented.append({
                "frame_number": p["frame_number"],
                "timestamp": p["timestamp"],
                "landmarks": augment_landmarks(
                    p["landmarks"], scale=scale, rotation_deg=rotation,
                    flip=flip, noise_std=noise_std
                ),
            })
    return poses + augmented


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================
def calculate_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def extract_full_metrics(pts):
    L_SH, R_SH = 11, 12
    L_HP, R_HP = 23, 24
    L_KN, R_KN = 25, 26
    L_AN, R_AN = 27, 28
    NOSE = 0

    shoulder_height = 1 - ((pts[L_SH][1] + pts[R_SH][1]) / 2)
    hip_height      = 1 - ((pts[L_HP][1] + pts[R_HP][1]) / 2)
    knee_height     = 1 - ((pts[L_KN][1] + pts[R_KN][1]) / 2)
    ankle_height    = 1 - ((pts[L_AN][1] + pts[R_AN][1]) / 2)

    shoulder_width = abs(pts[L_SH][0] - pts[R_SH][0])
    hip_width      = abs(pts[L_HP][0] - pts[R_HP][0])

    torso_angle     = calculate_angle(pts[L_SH], pts[L_HP], pts[L_AN])
    spine_angle     = calculate_angle(pts[R_SH], pts[R_HP], pts[R_AN])
    head_tilt_angle = calculate_angle(pts[NOSE], pts[L_SH], pts[R_SH])

    return {
        "shoulder_height": shoulder_height,
        "hip_height": hip_height,
        "knee_height": knee_height,
        "ankle_height": ankle_height,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_angle": torso_angle,
        "spine_angle": spine_angle,
        "head_tilt_angle": head_tilt_angle,
    }


def extract_features(poses, annotation, fps):
    all_features = []
    for step in annotation["steps"]:
        start = int(step["start_time"] * fps)
        end = int(step["end_time"] * fps)
        step_data = [p for p in poses if start <= p["frame_number"] <= end]
        if not step_data:
            continue

        feats = []
        for p in step_data:
            lm = np.array(p["landmarks"]).reshape(33, 4)
            pts = lm[:, :3]

            L_SH, R_SH = 11, 12
            L_EL, R_EL = 13, 14
            L_WR, R_WR = 15, 16
            L_HP, R_HP = 23, 24
            L_KN, R_KN = 25, 26
            L_AN, R_AN = 27, 28

            angles = {
                "left_shoulder_angle":  calculate_angle(pts[L_EL], pts[L_SH], pts[L_HP]),
                "right_shoulder_angle": calculate_angle(pts[R_EL], pts[R_SH], pts[R_HP]),
                "left_elbow_angle":     calculate_angle(pts[L_SH], pts[L_EL], pts[L_WR]),
                "right_elbow_angle":    calculate_angle(pts[R_SH], pts[R_EL], pts[R_WR]),
                "left_hip_angle":       calculate_angle(pts[L_SH], pts[L_HP], pts[L_KN]),
                "right_hip_angle":      calculate_angle(pts[R_SH], pts[R_HP], pts[R_KN]),
                "left_knee_angle":      calculate_angle(pts[L_HP], pts[L_KN], pts[L_AN]),
                "right_knee_angle":     calculate_angle(pts[R_HP], pts[R_KN], pts[R_AN]),
                "left_ankle_angle":     calculate_angle(pts[L_KN], pts[L_AN], pts[L_AN] + np.array([0, 1, 0])),
                "right_ankle_angle":    calculate_angle(pts[R_KN], pts[R_AN], pts[R_AN] + np.array([0, 1, 0])),
            }
            feats.append({**angles, **extract_full_metrics(pts)})

        df = pd.DataFrame(feats)
        summary = {
            "step_number": step["step_number"],
            "step_name": step["step_name"],
            "num_frames": len(df),
        }
        summary.update(df.mean().to_dict())
        summary.update({k + "_std": df[k].std() for k in df.columns})
        all_features.append(summary)

    return all_features


# ============================================================
# 4. RULE BUILDING
# ============================================================
def make_rules(features, annotation):
    rules = {"exercise_name": annotation["exercise_name"], "steps": []}
    if not features:
        return rules

    metadata_keys = {"step_number", "step_name", "num_frames"}
    all_keys = set().union(*(f.keys() for f in features))
    metric_keys = sorted(k for k in all_keys if k not in metadata_keys)
    bases = sorted({k for k in metric_keys if not k.endswith("_std")})

    for f in features:
        step_meta = next(
            (s for s in annotation["steps"] if s["step_number"] == f.get("step_number")),
            {},
        )
        step_rules = {
            "step_number": f.get("step_number"),
            "step_name": f.get("step_name"),
            "start_time": step_meta.get("start_time"),
            "end_time": step_meta.get("end_time"),
            "criteria": {},
        }
        for base in bases:
            if base not in f:
                continue
            try:
                mean_val = float(f[base]) if f[base] is not None else None
            except (TypeError, ValueError):
                mean_val = None
            if mean_val is None:
                continue

            std_key = base + "_std"
            std_val = None
            if std_key in f and f[std_key] is not None:
                try:
                    std_val = float(f[std_key])
                except (TypeError, ValueError):
                    std_val = None

            if std_val is not None and not np.isnan(std_val):
                step_rules["criteria"][base] = {
                    "min": mean_val - 2 * std_val,
                    "max": mean_val + 2 * std_val,
                    "mean": mean_val,
                    "std": std_val,
                }
            else:
                step_rules["criteria"][base] = {"expected": mean_val}
        rules["steps"].append(step_rules)

    return rules


# ============================================================
# 5. BATCH DRIVER
# ============================================================
def find_annotation(video_basename):
    """Look for <basename>.json in ANNOTATIONS_DIR (basename without extension)."""
    stem = os.path.splitext(video_basename)[0]
    candidate = os.path.join(ANNOTATIONS_DIR, stem + ".json")
    return candidate if os.path.exists(candidate) else None


def process_one(video_path):
    name = os.path.basename(video_path)
    stem = os.path.splitext(name)[0]

    ann_path = find_annotation(name)
    if not ann_path:
        return ("skipped", name, "no annotation file")

    with open(ann_path, "r") as fh:
        annotation = json.load(fh)

    data = process_video(video_path)
    if not data["poses"]:
        return ("failed", name, "no pose detected in any frame")

    augmented = augment_dataset(data, num_augmentations=NUM_AUGMENTATIONS)
    features = extract_features(augmented, annotation, data["fps"])
    if not features:
        return ("failed", name, "no step matched any frames (check start/end times)")

    rules = make_rules(features, annotation)

    out_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "annotation.json"), "w") as fh:
        json.dump(annotation, fh, indent=2)
    pd.DataFrame(features).to_csv(
        os.path.join(out_dir, "augmented_step_features.csv"), index=False
    )
    with open(os.path.join(out_dir, "augmented_step_validation_rules.json"), "w") as fh:
        json.dump(rules, fh, indent=2)
    if data["camera_distance"]:
        with open(os.path.join(out_dir, "ideal_camera_distance.json"), "w") as fh:
            json.dump(data["camera_distance"], fh, indent=2)

    return ("ok", name, f"{len(rules['steps'])} steps, {len(augmented)} frames")


def main():
    if not os.path.isdir(VIDEOS_DIR):
        raise SystemExit(f"VIDEOS_DIR not found: {VIDEOS_DIR}\n"
                         f"On macOS your external drive is under /Volumes/. "
                         f"Run `ls /Volumes` to see its exact name.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    videos = sorted(
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    print(f"Found {len(videos)} video(s) in {VIDEOS_DIR}\n")

    results = []
    for vp in tqdm(videos, desc="Videos"):
        try:
            results.append(process_one(vp))
        except Exception as e:
            traceback.print_exc()
            results.append(("failed", os.path.basename(vp), str(e)))

    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    for status in ("ok", "skipped", "failed"):
        rows = [r for r in results if r[0] == status]
        print(f"\n{status.upper()} ({len(rows)}):")
        for _, name, info in rows:
            print(f"  - {name}: {info}")
    print(f"\nOutputs written under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()