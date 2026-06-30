#!/usr/bin/env python3
"""
annotate.py
===========
Interactive annotation helper.

For every video in VIDEOS_DIR that does NOT yet have an annotation:
  1. Opens the video in your default player so you can watch it.
  2. Asks you: how many steps? what are their names? what are the timestamps?
  3. Writes the annotation JSON to ANNOTATIONS_DIR.
  4. Immediately runs the rules builder (trainer.py logic) on that video.

Run:  python3 annotate.py

Tips:
  - Press Enter to skip a video and come back later.
  - Type 'quit' at any prompt to stop and save progress.
  - Already-annotated videos are skipped automatically.
  - To re-annotate, delete its .json from the annotations/ folder.
"""

import os
import sys
import json
import subprocess
import traceback
import random

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ============================================================
# CONFIG  — same paths as your other scripts
# ============================================================
VIDEOS_DIR      = "/Volumes/Akoode Technologies/Lakshya/m2 method/All videos"
ANNOTATIONS_DIR = "/Users/lakshya/Desktop/OfficeProjects/m2 trainer/annotations"
OUTPUT_DIR      = "/Users/lakshya/Desktop/OfficeProjects/m2 trainer/output"

VIDEO_EXTENSIONS  = (".mp4", ".mov", ".m4v", ".avi", ".mkv")

# Rules-builder settings (mirrors trainer.py)
FRAME_STRIDE      = 1
MODEL_COMPLEXITY  = 2
PROCESS_SIZE      = 720
NUM_AUGMENTATIONS = 6
USE_FLIP_AUG      = False

mp_pose = mp.solutions.pose


# ============================================================
# Rules builder helpers (self-contained copy from trainer.py)
# ============================================================
def letterbox(frame, size):
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
        static_image_mode=False, model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True, min_detection_confidence=0.5,
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
    pbar = tqdm(total=total_frames, desc="  extracting poses", leave=False)
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
                poses.append({"frame_number": count, "timestamp": count / fps, "landmarks": landmarks})
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
                "min_z":  round(float(np.min(z_values)), 4),
                "max_z":  round(float(np.max(z_values)), 4),
                "mean_z": round(float(np.mean(z_values)), 4),
            }
        }
    return {"poses": poses, "fps": fps, "total_frames": total_frames, "camera_distance": camera_distance}


def augment_landmarks(landmarks, scale=1.0, rotation_deg=0, flip=False, noise_std=0.01):
    lm = np.array(landmarks).reshape(33, 4)
    coords = lm[:, :3] * scale
    theta = np.radians(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0, 0, 1]])
    coords = coords.dot(rot.T)
    if flip:
        coords[:, 0] = -coords[:, 0]
    coords = coords + np.random.normal(0, noise_std, coords.shape)
    lm[:, :3] = coords
    return lm.flatten().tolist()


def augment_dataset(data):
    poses = data["poses"]
    augmented = []
    for _ in range(NUM_AUGMENTATIONS):
        scale     = random.uniform(0.95, 1.05)
        rotation  = random.uniform(-10, 10)
        flip      = random.choice([True, False]) if USE_FLIP_AUG else False
        noise_std = random.uniform(0.005, 0.02)
        for p in poses:
            augmented.append({
                "frame_number": p["frame_number"],
                "timestamp": p["timestamp"],
                "landmarks": augment_landmarks(p["landmarks"], scale=scale,
                                               rotation_deg=rotation, flip=flip, noise_std=noise_std),
            })
    return poses + augmented


def calculate_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def extract_full_metrics(pts):
    L_SH, R_SH, L_HP, R_HP = 11, 12, 23, 24
    L_KN, R_KN, L_AN, R_AN = 25, 26, 27, 28
    NOSE = 0
    return {
        "shoulder_height": 1 - ((pts[L_SH][1] + pts[R_SH][1]) / 2),
        "hip_height":      1 - ((pts[L_HP][1] + pts[R_HP][1]) / 2),
        "knee_height":     1 - ((pts[L_KN][1] + pts[R_KN][1]) / 2),
        "ankle_height":    1 - ((pts[L_AN][1] + pts[R_AN][1]) / 2),
        "shoulder_width":  abs(pts[L_SH][0] - pts[R_SH][0]),
        "hip_width":       abs(pts[L_HP][0] - pts[R_HP][0]),
        "torso_angle":     calculate_angle(pts[L_SH], pts[L_HP], pts[L_AN]),
        "spine_angle":     calculate_angle(pts[R_SH], pts[R_HP], pts[R_AN]),
        "head_tilt_angle": calculate_angle(pts[NOSE], pts[L_SH], pts[R_SH]),
    }


def extract_features(poses, annotation, fps):
    all_features = []
    for step in annotation["steps"]:
        start = int(step["start_time"] * fps)
        end   = int(step["end_time"]   * fps)
        step_data = [p for p in poses if start <= p["frame_number"] <= end]
        if not step_data:
            continue
        feats = []
        for p in step_data:
            lm  = np.array(p["landmarks"]).reshape(33, 4)
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
        summary = {"step_number": step["step_number"], "step_name": step["step_name"], "num_frames": len(df)}
        summary.update(df.mean().to_dict())
        summary.update({k + "_std": df[k].std() for k in df.columns})
        all_features.append(summary)
    return all_features


def make_rules(features, annotation):
    rules = {"exercise_name": annotation["exercise_name"], "steps": []}
    if not features:
        return rules
    metadata_keys = {"step_number", "step_name", "num_frames"}
    all_keys    = set().union(*(f.keys() for f in features))
    metric_keys = sorted(k for k in all_keys if k not in metadata_keys)
    bases = sorted({k for k in metric_keys if not k.endswith("_std")})
    for f in features:
        step_meta = next((s for s in annotation["steps"] if s["step_number"] == f.get("step_number")), {})
        step_rules = {
            "step_number": f.get("step_number"),
            "step_name":   f.get("step_name"),
            "start_time":  step_meta.get("start_time"),
            "end_time":    step_meta.get("end_time"),
            "criteria":    {},
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


def run_rules_builder(video_path, annotation):
    name = os.path.basename(video_path)
    stem = os.path.splitext(name)[0]
    print(f"\n  Extracting poses from '{name}' (this takes a minute)...")
    data = process_video(video_path)
    if not data["poses"]:
        print("  ERROR: No poses detected. Check the video.")
        return False
    augmented = augment_dataset(data)
    features  = extract_features(augmented, annotation, data["fps"])
    if not features:
        print("  ERROR: No frames matched the step time ranges. Double-check your timestamps.")
        return False
    rules = make_rules(features, annotation)
    out_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "annotation.json"), "w") as fh:
        json.dump(annotation, fh, indent=2)
    pd.DataFrame(features).to_csv(os.path.join(out_dir, "augmented_step_features.csv"), index=False)
    with open(os.path.join(out_dir, "augmented_step_validation_rules.json"), "w") as fh:
        json.dump(rules, fh, indent=2)
    if data["camera_distance"]:
        with open(os.path.join(out_dir, "ideal_camera_distance.json"), "w") as fh:
            json.dump(data["camera_distance"], fh, indent=2)
    print(f"  Done! Rules written to: output/{stem}/")
    return True


# ============================================================
# Interactive prompts
# ============================================================
def ask(prompt, allow_empty=False):
    try:
        val = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nStopped. Progress saved.")
        sys.exit(0)
    if val.lower() == "quit":
        print("\nStopped. Progress saved.")
        sys.exit(0)
    if not allow_empty and not val:
        return None
    return val


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return round(frames / fps, 2)


def open_video(video_path):
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", video_path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", video_path])
        else:
            os.startfile(video_path)
    except Exception as e:
        print(f"  (Could not auto-open video: {e}. Open it manually.)")


def prompt_annotation(video_path, duration):
    name = os.path.basename(video_path)
    stem = os.path.splitext(name)[0]

    print(f"\n{'='*60}")
    print(f"  VIDEO : {name}")
    print(f"  LENGTH: {duration}s")
    print(f"{'='*60}")
    print("  The video is now open in your player.")
    print("  Watch it, then come back and enter the step info.")
    print("  Tips:")
    print("    - Press Enter to SKIP this video for now.")
    print("    - Type 'quit' to stop the whole script.\n")

    # Number of steps
    while True:
        raw = ask("  How many steps/phases does this exercise have? [e.g. 4]: ")
        if raw is None:
            print("  Skipping this video.\n")
            return None
        try:
            n_steps = int(raw)
            if n_steps >= 1:
                break
        except ValueError:
            pass
        print("  Enter a whole number like 3 or 4.")

    steps = []
    prev_end = 0.0

    for i in range(n_steps):
        print(f"\n  --- Step {i+1} of {n_steps} ---")

        # Step name
        default_name = f"step_{i+1}"
        raw_name = ask(f"  Name for step {i+1} (e.g. 'lift leg up') [default: {default_name}]: ",
                       allow_empty=True)
        if raw_name is None or raw_name == "":
            raw_name = default_name
        step_name = raw_name.lower().replace(" ", "_")

        # Start time
        start_time = 0.0 if i == 0 else prev_end
        if i == 0:
            print(f"  Start time: 0.0s  (auto)")
        else:
            print(f"  Start time: {start_time}s  (auto)")

        # End time
        if i == n_steps - 1:
            end_time = duration
            print(f"  End time  : {duration}s  (end of video, auto)")
        else:
            while True:
                raw = ask(f"  End time for step {i+1} in seconds (> {start_time} and < {duration}): ")
                if raw is None:
                    print("  Skipping this video.\n")
                    return None
                try:
                    end_time = float(raw)
                    if start_time < end_time < duration:
                        break
                    print(f"  Must be between {start_time} and {duration}. Try again.")
                except ValueError:
                    print("  Enter a number like 12.5")

        steps.append({
            "step_number": i + 1,
            "step_name":   step_name,
            "start_time":  round(start_time, 2),
            "end_time":    round(end_time, 2),
            "key_checks":  [],
        })
        prev_end = end_time

    annotation = {
        "exercise_name":         stem,
        "total_duration_seconds": duration,
        "steps":                 steps,
        "_auto_segmented":       False,
    }

    # Preview + confirm
    print(f"\n  Preview for '{stem}':")
    for s in steps:
        print(f"    Step {s['step_number']}: {s['step_name']}  [{s['start_time']}s -> {s['end_time']}s]")

    confirm = ask("\n  Save and build rules? (y = yes / n = redo / Enter = skip): ", allow_empty=True)
    if confirm is None or confirm == "":
        print("  Skipping.\n")
        return None
    if confirm.lower() == "n":
        print("  Redoing...\n")
        return prompt_annotation(video_path, duration)
    return annotation


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(VIDEOS_DIR):
        raise SystemExit(f"VIDEOS_DIR not found: {VIDEOS_DIR}")

    videos = sorted(
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )

    if not videos:
        raise SystemExit(f"No videos found in {VIDEOS_DIR}")

    pending, already_done = [], []
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0]
        ann_path = os.path.join(ANNOTATIONS_DIR, stem + ".json")
        if os.path.exists(ann_path):
            already_done.append(os.path.basename(vp))
        else:
            pending.append(vp)

    print(f"\n{'='*60}")
    print(f"  Found {len(videos)} video(s) total")
    print(f"  Already annotated : {len(already_done)}")
    print(f"  Need annotation   : {len(pending)}")
    print(f"{'='*60}")

    if already_done:
        print(f"  Skipping (already done): {', '.join(already_done)}")

    if not pending:
        print("\n  All videos annotated! Nothing to do.")
        print("  To re-annotate a video, delete its .json from annotations/")
        return

    print(f"\n  Will annotate {len(pending)} video(s) interactively.")
    print("  For each video:")
    print("    1. The video opens automatically")
    print("    2. You watch it and note down the timestamps")
    print("    3. You enter the step names + timestamps here")
    print("    4. Rules are built immediately\n")

    input("  Press Enter to begin... ")

    success, skipped, failed = [], [], []

    for vp in pending:
        name = os.path.basename(vp)
        duration = get_video_duration(vp)
        open_video(vp)

        annotation = prompt_annotation(vp, duration)
        if annotation is None:
            skipped.append(name)
            continue

        stem = os.path.splitext(name)[0]
        ann_path = os.path.join(ANNOTATIONS_DIR, stem + ".json")
        with open(ann_path, "w") as fh:
            json.dump(annotation, fh, indent=2)
        print(f"\n  Annotation saved -> annotations/{stem}.json")

        try:
            ok = run_rules_builder(vp, annotation)
            (success if ok else failed).append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*60}")
    print("  FINISHED")
    print(f"{'='*60}")
    print(f"  Processed : {len(success)}  {success}")
    print(f"  Skipped   : {len(skipped)}  {skipped}")
    print(f"  Failed    : {len(failed)}  {failed}")
    print(f"\n  All outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
