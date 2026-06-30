#!/usr/bin/env python3
"""
autosegment.py  —  STAGE 2 of 2
===============================
One MediaPipe pass per video, producing EVERYTHING downstream needs:

  1. Exact step boundaries — snaps the ROUGH times you gave in Stage 1
     (author_steps.py) to the nearest motion "settle point", and auto-detects
     any you left blank.
  2. Final annotation        -> annotations/<stem>.json
  3. Validation rules        -> output/<stem>/augmented_step_validation_rules.json
                                output/<stem>/augmented_step_features.csv
  4. Camera-distance guide    -> output/<stem>/ideal_camera_distance.json
  5. Flutter JSON trajectory  -> output/<stem>/trajectory.json   (Dart-readable)
                                output/trajectories/<stem>.npz   (internal extra)
  6. Boundary contact sheet   -> output/<stem>/boundaries_contactsheet.jpg
  7. App bundle               -> output/<stem>/app_bundle.json
  8. Catalog (after batch)    -> output/exercises_index.json

Input per video (from Stage 1), either is accepted:
  - annotations/<stem>.steps.json   (object: names + rough times + cues)
  - annotations/<stem>.names.json   (legacy plain list of names)

A video with no such file is skipped.

Run:  venv/bin/python autosegment.py
"""

import os
import json
import traceback

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

import config
# Reuse the rule-building logic from trainer.py (no duplication, no 2nd pose pass).
from trainer import augment_dataset, extract_features, make_rules

mp_pose = mp.solutions.pose


# ============================================================
# Single pose pass -> raw + normalized trajectory + camera z
# ============================================================
def extract_trajectory(video_path):
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=config.MODEL_COMPLEXITY,
        smooth_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_seq, norm_seq, times, frames, z_values = [], [], [], [], []
    torso_idx = [config.L_HP, config.R_HP, config.L_SH, config.R_SH]

    count = 0
    pbar = tqdm(total=total, desc="  poses", leave=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % config.FRAME_STRIDE == 0:
            sq = config.letterbox(frame, config.PROCESS_SIZE)
            rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                arr = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
                )
                raw_seq.append(arr)                  # (33,4)
                norm_seq.append(config.normalize(arr))  # (33,3)
                times.append(count / fps)
                frames.append(count)
                z_values.append(float(np.mean([arr[i, 2] for i in torso_idx])))
        count += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    pose.close()

    if not norm_seq:
        return None
    return {
        "raw": np.array(raw_seq),        # (T,33,4)
        "norm": np.array(norm_seq),      # (T,33,3)
        "times": np.array(times),        # (T,)
        "frames": np.array(frames),      # (T,) original frame indices
        "z": np.array(z_values),         # (T,)
        "fps": fps,
        "total_frames": total,
        "duration": total / fps,
    }


# ============================================================
# Stage-1 spec loading (.steps.json or legacy .names.json)
# ============================================================
def load_spec(stem):
    """Returns (names, rough_end_times, cues) or None if no input file."""
    steps_path = os.path.join(config.ANNOTATIONS_DIR, stem + ".steps.json")
    if os.path.exists(steps_path):
        with open(steps_path) as fh:
            spec = json.load(fh)
        steps = spec.get("steps", [])
        names = [s["step_name"] for s in steps]
        rough = [s.get("rough_end_time") for s in steps]
        cues  = [s.get("cue", "") for s in steps]
        return names, rough, cues

    names_path = os.path.join(config.ANNOTATIONS_DIR, stem + ".names.json")
    if os.path.exists(names_path):
        with open(names_path) as fh:
            names = json.load(fh)
        if isinstance(names, list) and names:
            names = [str(n) for n in names]
            return names, [None] * len(names), [""] * len(names)
    return None


# ============================================================
# Motion energy + boundary pinning
# ============================================================
def motion_energy(norm):
    """Per-frame speed of key joints (normalized units), smoothed."""
    diffs = np.diff(norm[:, config.KEY_JOINTS, :], axis=0)   # (T-1, K, 3)
    speed = np.linalg.norm(diffs, axis=2).sum(axis=1)        # (T-1,)
    speed = np.concatenate([[speed[0]], speed]) if len(speed) else np.zeros(len(norm))
    return config.moving_average(speed, config.SMOOTH_WINDOW)


def pin_boundaries(energy, times, rough, n_steps, duration):
    """
    Returns n_steps-1 sorted internal boundary TIMES.
    Rough times snap to the quietest frame within +/- SNAP_WINDOW_SECONDS;
    blanks fall back to globally quietest, well-separated frames.
    """
    n_internal = max(0, n_steps - 1)
    if n_internal == 0:
        return []

    fps_like = 1.0 / np.mean(np.diff(times)) if len(times) > 1 else 30.0
    min_sep  = max(1, int(config.MIN_STEP_SECONDS * fps_like))
    snap_win = max(1, int(config.SNAP_WINDOW_SECONDS * fps_like))
    n = len(energy)

    chosen = {}  # boundary index -> frame index
    # 1) snap provided rough times
    for i in range(n_internal):
        rt = rough[i] if i < len(rough) else None
        if rt is None:
            continue
        center = int(np.argmin(np.abs(times - rt)))
        lo, hi = max(0, center - snap_win), min(n, center + snap_win + 1)
        chosen[i] = lo + int(np.argmin(energy[lo:hi]))

    # 2) fill blanks with globally quietest, separated frames
    taken = sorted(chosen.values())
    order = np.argsort(energy)
    for i in range(n_internal):
        if i in chosen:
            continue
        for idx in order:
            idx = int(idx)
            if idx < min_sep or idx > n - min_sep:
                continue
            if all(abs(idx - t) >= min_sep for t in taken):
                chosen[i] = idx
                taken.append(idx)
                taken.sort()
                break

    bnd_times = sorted(float(times[chosen[i]]) for i in range(n_internal) if i in chosen)

    # Fallback: short/degenerate video -> evenly spaced boundaries.
    if len(bnd_times) < n_internal:
        bnd_times = list(np.linspace(0, duration, n_steps + 1))[1:-1]
    return [round(t, 2) for t in bnd_times]


def build_annotation(stem, names, cues, boundaries, duration):
    times = [0.0] + list(boundaries) + [duration]
    steps = []
    for i, name in enumerate(names):
        steps.append({
            "step_number": i + 1,
            "step_name": name,
            "start_time": round(times[i], 2),
            "end_time": round(times[i + 1], 2),
            "cue": cues[i] if i < len(cues) else "",
            "key_checks": [],
        })
    return {
        "exercise_name": stem.replace("_", " "),
        "total_duration_seconds": round(duration, 2),
        "steps": steps,
        "_auto_segmented": True,
    }


# ============================================================
# Flutter exports
# ============================================================
def smooth_normalized(norm):
    """Temporally smooth each joint/coordinate of the (T,33,3) trajectory."""
    T = norm.shape[0]
    out = np.empty_like(norm)
    for j in range(33):
        for c in range(3):
            out[:, j, c] = config.moving_average(norm[:, j, c], config.SMOOTH_WINDOW)
    return out


def write_trajectory_json(path, exercise_name, smoothed, times, fps):
    """Downsampled, key-joint-only normalized trajectory (Dart-readable)."""
    stride = max(1, int(round(fps / config.EXPORT_HZ)))
    frames = []
    for i in range(0, len(times), stride):
        pts = smoothed[i][config.EXPORT_JOINTS]        # (J,3)
        frames.append({
            "t": round(float(times[i]), 3),
            "pts": [[round(float(v), 4) for v in p] for p in pts],
        })
    payload = {
        "schema_version": config.SCHEMA_VERSION,
        "exercise_name": exercise_name,
        "normalization": "hip_centered_torso_scaled",
        "sample_rate_hz": config.EXPORT_HZ,
        "joints": config.EXPORT_JOINTS,
        "frames": frames,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh)


def write_contact_sheet(path, video_path, annotation, fps, tile=320):
    """Grab the first frame of each step, label it, tile horizontally."""
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for s in annotation["steps"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s["start_time"] * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        thumb = config.letterbox(frame, tile)
        label = f"{s['step_number']}. {s['step_name']}  {s['start_time']}s"
        cv2.rectangle(thumb, (0, 0), (tile, 28), (0, 0, 0), -1)
        cv2.putText(thumb, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(thumb)
    cap.release()
    if tiles:
        cv2.imwrite(path, cv2.hconcat(tiles))


def write_app_bundle(path, annotation, rules, camera_distance):
    crit_by_step = {s["step_number"]: s.get("criteria", {}) for s in rules.get("steps", [])}
    bundle = {
        "schema_version": config.SCHEMA_VERSION,
        "exercise_name": annotation["exercise_name"],
        "total_duration_seconds": annotation["total_duration_seconds"],
        "camera_distance": (camera_distance or {}).get("ideal_camera_distance"),
        "trajectory_file": "trajectory.json",
        "steps": [
            {
                "step_number": s["step_number"],
                "step_name": s["step_name"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "cue": s.get("cue", ""),
                "criteria": crit_by_step.get(s["step_number"], {}),
            }
            for s in annotation["steps"]
        ],
    }
    with open(path, "w") as fh:
        json.dump(bundle, fh, indent=2)


# ============================================================
# Per-video driver
# ============================================================
def process_one(video_path):
    name = os.path.basename(video_path)
    stem = config.stem_of(video_path)

    spec = load_spec(stem)
    if spec is None:
        return ("skipped", name, "no .steps.json/.names.json (run author_steps.py)")
    names, rough, cues = spec

    traj = extract_trajectory(video_path)
    if traj is None:
        return ("failed", name, "no pose detected")

    # --- boundaries + final annotation ---
    energy = motion_energy(traj["norm"])
    boundaries = pin_boundaries(energy, traj["times"], rough, len(names), traj["duration"])
    annotation = build_annotation(stem, names, cues, boundaries, traj["duration"])

    os.makedirs(config.ANNOTATIONS_DIR, exist_ok=True)
    with open(os.path.join(config.ANNOTATIONS_DIR, stem + ".json"), "w") as fh:
        json.dump(annotation, fh, indent=2)

    out_dir = os.path.join(config.OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    # --- validation rules (reuse trainer.py, on already-extracted poses) ---
    poses = [
        {"frame_number": int(traj["frames"][i]),
         "timestamp": float(traj["times"][i]),
         "landmarks": traj["raw"][i].flatten().tolist()}
        for i in range(len(traj["frames"]))
    ]
    augmented = augment_dataset({"poses": poses}, num_augmentations=config.NUM_AUGMENTATIONS)
    features = extract_features(augmented, annotation, traj["fps"])
    rules = make_rules(features, annotation)
    with open(os.path.join(out_dir, "augmented_step_validation_rules.json"), "w") as fh:
        json.dump(rules, fh, indent=2)
    if features:
        pd.DataFrame(features).to_csv(
            os.path.join(out_dir, "augmented_step_features.csv"), index=False)

    # --- camera distance ---
    camera_distance = {
        "ideal_camera_distance": {
            "min_z":  round(float(np.min(traj["z"])), 4),
            "max_z":  round(float(np.max(traj["z"])), 4),
            "mean_z": round(float(np.mean(traj["z"])), 4),
        }
    }
    with open(os.path.join(out_dir, "ideal_camera_distance.json"), "w") as fh:
        json.dump(camera_distance, fh, indent=2)

    # --- trajectory (JSON for Flutter + npz internal) ---
    smoothed = smooth_normalized(traj["norm"])
    write_trajectory_json(
        os.path.join(out_dir, "trajectory.json"),
        annotation["exercise_name"], smoothed, traj["times"], traj["fps"])
    os.makedirs(config.TRAJECTORY_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(config.TRAJECTORY_DIR, stem + ".npz"),
        trajectory=smoothed, times=traj["times"],
        visibility=traj["raw"][:, :, 3], boundaries=np.array(boundaries))

    # --- contact sheet + app bundle ---
    write_contact_sheet(
        os.path.join(out_dir, "boundaries_contactsheet.jpg"),
        video_path, annotation, traj["fps"])
    write_app_bundle(
        os.path.join(out_dir, "app_bundle.json"), annotation, rules, camera_distance)

    return ("ok", name, f"{len(names)} steps, {len(traj['times'])} frames")


def write_index(results):
    """Catalog of every exercise that produced an app bundle."""
    entries = []
    for status, name, _ in results:
        if status != "ok":
            continue
        stem = os.path.splitext(name)[0]
        bundle_path = os.path.join(config.OUTPUT_DIR, stem, "app_bundle.json")
        try:
            with open(bundle_path) as fh:
                b = json.load(fh)
            entries.append({
                "exercise_name": b["exercise_name"],
                "folder": stem,
                "total_duration_seconds": b["total_duration_seconds"],
                "num_steps": len(b["steps"]),
            })
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    index = {"schema_version": config.SCHEMA_VERSION, "exercises": entries}
    with open(os.path.join(config.OUTPUT_DIR, "exercises_index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    return len(entries)


def main():
    if not os.path.isdir(config.VIDEOS_DIR):
        raise SystemExit(f"VIDEOS_DIR not found: {config.VIDEOS_DIR}  (edit config.py)")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    videos = config.list_videos()
    print(f"Found {len(videos)} video(s)\n")

    results = []
    for vp in tqdm(videos, desc="Videos"):
        try:
            results.append(process_one(vp))
        except Exception as e:
            traceback.print_exc()
            results.append(("failed", os.path.basename(vp), str(e)))

    n_indexed = write_index(results)

    print("\n" + "=" * 70)
    for status in ("ok", "skipped", "failed"):
        rows = [r for r in results if r[0] == status]
        print(f"\n{status.upper()} ({len(rows)}):")
        for _, n, info in rows:
            print(f"  - {n}: {info}")
    print(f"\nCatalog: output/exercises_index.json ({n_indexed} exercises)")
    print(f"Per-exercise outputs under: {config.OUTPUT_DIR}/<stem>/")


if __name__ == "__main__":
    main()
