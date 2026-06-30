#!/usr/bin/env python3
"""
author_steps.py  —  STAGE 1 of 2
================================
Interactive step authoring (like the start of annotate.py), but you ONLY enter:

  - how many steps the exercise has,
  - each step's name,
  - (optionally) a ROUGH boundary time.

You do NOT pin exact timestamps here. Stage 2 (autosegment.py) snaps your rough
times to the exact motion "settle point" and detects any you left blank.

For every video in VIDEOS_DIR without a <stem>.steps.json it:
  1. Opens the video in your default player.
  2. Prompts for step count / names / optional rough end times.
  3. Writes annotations/<stem>.steps.json

Run:  venv/bin/python author_steps.py

Tips:
  - Press Enter at "how many steps" to SKIP a video for now.
  - Leave a rough time blank (just press Enter) to let Stage 2 decide it.
  - Type 'quit' at any prompt to stop; finished videos are already saved.
  - To redo a video, delete its .steps.json from annotations/.
"""

import os
import sys
import json
import subprocess

import cv2

import config


# ============================================================
# Interactive helpers (adapted from annotate.py)
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
            os.startfile(video_path)  # noqa: type-ignore (Windows only)
    except Exception as e:
        print(f"  (Could not auto-open video: {e}. Open it manually.)")


# ============================================================
# Prompt one video -> steps spec
# ============================================================
def prompt_steps(video_path, duration):
    stem = config.stem_of(video_path)
    name = os.path.basename(video_path)

    print(f"\n{'='*60}")
    print(f"  VIDEO : {name}")
    print(f"  LENGTH: {duration}s")
    print(f"{'='*60}")
    print("  The video is now open in your player.")
    print("  Enter the step names. Rough times are OPTIONAL — leave blank to let")
    print("  Stage 2 detect them automatically.")
    print("  Press Enter at the step-count prompt to SKIP this video.\n")

    # Number of steps
    while True:
        raw = ask("  How many steps/phases? [e.g. 4]: ")
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
    last_rough = 0.0
    for i in range(n_steps):
        print(f"\n  --- Step {i+1} of {n_steps} ---")

        default_name = f"step_{i+1}"
        raw_name = ask(
            f"  Name for step {i+1} (e.g. 'lift leg up') [default: {default_name}]: ",
            allow_empty=True,
        )
        if not raw_name:
            raw_name = default_name
        step_name = raw_name.lower().replace(" ", "_")

        # Rough end time — only meaningful for non-final steps. Optional.
        rough_end = None
        if i < n_steps - 1:
            while True:
                raw = ask(
                    f"  ROUGH end time for step {i+1} in seconds "
                    f"(> {last_rough}, < {duration}) [Enter = auto]: ",
                    allow_empty=True,
                )
                if not raw:
                    rough_end = None
                    break
                try:
                    t = float(raw)
                    if last_rough < t < duration:
                        rough_end = round(t, 2)
                        last_rough = t
                        break
                    print(f"  Must be between {last_rough} and {duration}. Try again.")
                except ValueError:
                    print("  Enter a number like 12.5, or press Enter to skip.")

        # Optional spoken/displayed cue for the app (blank = fill later).
        cue = ask(
            f"  Cue text shown/spoken for '{step_name}' [Enter = none]: ",
            allow_empty=True,
        ) or ""

        steps.append({
            "step_number": i + 1,
            "step_name": step_name,
            "rough_end_time": rough_end,
            "cue": cue,
        })

    spec = {
        "exercise_name": stem.replace("_", " "),
        "total_duration_seconds": duration,
        "steps": steps,
    }

    # Preview + confirm
    print(f"\n  Preview for '{stem}':")
    for s in steps:
        rt = "auto" if s["rough_end_time"] is None else f"~{s['rough_end_time']}s"
        print(f"    Step {s['step_number']}: {s['step_name']}  (rough end: {rt})")

    confirm = ask("\n  Save? (y = yes / n = redo / Enter = skip): ", allow_empty=True)
    if not confirm:
        print("  Skipping.\n")
        return None
    if confirm.lower() == "n":
        print("  Redoing...\n")
        return prompt_steps(video_path, duration)
    return spec


# ============================================================
# Main
# ============================================================
def steps_path(stem):
    return os.path.join(config.ANNOTATIONS_DIR, stem + ".steps.json")


def main():
    os.makedirs(config.ANNOTATIONS_DIR, exist_ok=True)

    if not os.path.isdir(config.VIDEOS_DIR):
        raise SystemExit(
            f"VIDEOS_DIR not found: {config.VIDEOS_DIR}\n"
            f"On macOS your external drive is under /Volumes/. "
            f"Run `ls /Volumes` to see its exact name, then edit config.py."
        )

    videos = config.list_videos()
    if not videos:
        raise SystemExit(f"No videos found in {config.VIDEOS_DIR}")

    pending, done = [], []
    for vp in videos:
        (done if os.path.exists(steps_path(config.stem_of(vp))) else pending).append(vp)

    print(f"\n{'='*60}")
    print(f"  Found {len(videos)} video(s)")
    print(f"  Already authored : {len(done)}")
    print(f"  Need authoring   : {len(pending)}")
    print(f"{'='*60}")

    if not pending:
        print("\n  All videos authored! Run Stage 2:  venv/bin/python autosegment.py")
        print("  To redo one, delete its .steps.json from annotations/.")
        return

    input("\n  Press Enter to begin... ")

    saved, skipped = [], []
    for vp in pending:
        name = os.path.basename(vp)
        duration = get_video_duration(vp)
        open_video(vp)

        spec = prompt_steps(vp, duration)
        if spec is None:
            skipped.append(name)
            continue

        out = steps_path(config.stem_of(vp))
        with open(out, "w") as fh:
            json.dump(spec, fh, indent=2)
        print(f"  Saved -> {os.path.basename(out)}")
        saved.append(name)

    print(f"\n{'='*60}")
    print("  STAGE 1 FINISHED")
    print(f"{'='*60}")
    print(f"  Saved   : {len(saved)}  {saved}")
    print(f"  Skipped : {len(skipped)}  {skipped}")
    print("\n  Next:  venv/bin/python autosegment.py")


if __name__ == "__main__":
    main()
