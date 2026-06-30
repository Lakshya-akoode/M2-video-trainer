# Flutter Export Schema

Stage 2 (`autosegment.py`) writes, per exercise, a folder
`output/<stem>/` and one catalog `output/exercises_index.json`. These are the
files your Flutter app should ship and read. All landmark indices follow
[MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
(0 = nose, 11/12 = shoulders, 23/24 = hips, 25/26 = knees, 27/28 = ankles).

Every file carries `schema_version` so the app can evolve safely.

---

## `exercises_index.json` — the catalog (read first)

```json
{
  "schema_version": 1,
  "exercises": [
    { "exercise_name": "a4", "folder": "a4 (1)",
      "total_duration_seconds": 41.2, "num_steps": 4 }
  ]
}
```

`folder` is the directory under `output/` that holds the rest of this
exercise's files.

---

## `app_bundle.json` — everything for one exercise (except the trajectory)

```json
{
  "schema_version": 1,
  "exercise_name": "a4",
  "total_duration_seconds": 41.2,
  "camera_distance": { "min_z": -0.9, "max_z": -0.4, "mean_z": -0.65 },
  "trajectory_file": "trajectory.json",
  "steps": [
    {
      "step_number": 1,
      "step_name": "start_position",
      "start_time": 0.0,
      "end_time": 6.2,
      "cue": "Lie flat on your back",
      "criteria": {
        "left_knee_angle": { "min": 165.1, "max": 178.9, "mean": 172.0, "std": 3.4 },
        "hip_height":      { "expected": 0.51 }
      }
    }
  ]
}
```

- **camera_distance** — torso `z` range MediaPipe reported in the reference
  video. Before the user starts, run pose on their camera and nudge them
  closer/farther until their torso `z` sits inside `[min_z, max_z]`.
- **steps[].start_time / end_time** — seconds. Drive the on-screen step timeline
  and decide which step's `criteria` to score at the current moment.
- **steps[].cue** — text to display/speak when the step begins.
- **steps[].criteria** — per-metric pass bands for live form feedback:
  - `{min, max, mean, std}` → the user value is "good" inside `[min, max]`
    (mean ± 2·std). Show `mean` as the target.
  - `{expected}` → a single target value when no spread was available.
  - Angle metrics are in **degrees**; height/width metrics are normalized image
    fractions (0–1). Names: `left/right_{shoulder,elbow,hip,knee,ankle}_angle`,
    `torso_angle`, `spine_angle`, `head_tilt_angle`, `shoulder/hip_height`,
    `knee/ankle_height`, `shoulder/hip_width`.

---

## `trajectory.json` — reference motion for pose matching

```json
{
  "schema_version": 1,
  "exercise_name": "a4",
  "normalization": "hip_centered_torso_scaled",
  "sample_rate_hz": 10,
  "joints": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
  "frames": [
    { "t": 0.0, "pts": [[0.01, -0.98, 0.03], [ ... ]] }
  ]
}
```

- **joints** — which landmark indices each frame's `pts` array corresponds to,
  in order. (Downsampled to the joints the app needs; not all 33.)
- **pts** — normalized 3-D coords: origin at the hip midpoint, scaled by torso
  length, so they're **independent of the user's body size and distance**.
  Normalize the user's live landmarks the same way before comparing
  (`(pt - hip_mid) / ||shoulder_mid - hip_mid||`).
- **sample_rate_hz** — frames are ~10 Hz. For scoring a step, slice
  `frames` to the step's `[start_time, end_time]` and run **DTW** against the
  user's same-window normalized sequence; lower distance = closer to ideal.

### Mirror / front-camera note (important)
A selfie (front) camera **mirrors** the user, so their left/right is flipped vs.
the reference. Before comparing, either:
- negate the **x** of the user's normalized points, **or**
- compare against an x-negated copy of the reference `pts`.

Do this consistently or left/right-specific exercises will always score wrong.

---

## Files NOT for the app (internal / QA)
- `boundaries_contactsheet.jpg` — thumbnail of each step's first frame for you
  to eyeball that the auto-cuts are right. Fix `annotations/<stem>.json` and
  re-run Stage 2 if a cut is off.
- `augmented_step_features.csv`, `output/trajectories/<stem>.npz` — debug/raw.
