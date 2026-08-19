# KneeCheck

A fully **offline** webcam app for knee assessments (KKH × NYP
research prototype), suitable for patients of any age. MediaPipe Pose measures the knee joint angle
(hip–knee–ankle); MediaPipe Face Mesh captures a per-user neutral-face
**baseline**, and during exercises a **pain level** (estimated from facial expression) is
computed as the weighted deviation from that baseline (brow lowering, brow
knitting, eye squeezing, lip pressing, mouth-corner droop).

**Privacy by default** — sessions store only landmark coordinates and derived
scores. Saving video requires flipping an explicit switch on the setup page
(default OFF). The app makes no network calls.

## Run it

```powershell
cd kneecheck
pip install -r requirements.txt
python main.py
```

Windows, Python 3.10+, any standard webcam. If no camera is found the app
shows a retry page instead of crashing.

## The flow

1. **Home** — "New user" (name → profile → 5-second relaxed-face scan with
   live feedback + progress bar) or "Returning user" (loads the saved baseline —
   **the scan is skipped**; a *Redo my face scan* button is on the profile
   page if it's stale).
2. **Choose an activity** — Sit to Stand / Knee Bends (ROM) / Single Leg
   Stance — then its options: legs (Sit to Stand supports both-legs with a
   measured side, or single-leg), sets × reps (or seconds per hold), and the
   optional video switch. Unfinished sessions report the average per set.
3. **Live** — skeleton overlay, knee-angle tag at the joint, face inset,
   top status bar, star-per-rep row (or hold bar), colored pain-level meter,
   Finish / Stop buttons. The app tracks **only the chosen leg** — if it isn't
   visible it asks the user to step back rather than switching legs.
4. **Results** — reps / best hold, min/avg/max knee angle, avg/max pain level,
   difficulty rating, and where everything was saved.

## What each session saves (`sessions/<user>_<exercise>_<time>/`)

| File              | Contents                                                            |
| ----------------- | ------------------------------------------------------------------- |
| `frames.csv`      | per frame: time, knee angle, reps/hold, pain score + 5 sub-scores, tracking flags |
| `face_points.csv` | 468 face landmarks (x, y, z) per frame                              |
| `summary.csv`     | key–value session summary                                           |
| `capture.mp4`     | **only** if the video switch was on                                 |

Profiles (baseline JSON) live in `profiles/<name>/profile.json`.

## Tuning (clinicians)

Everything adjustable is a named constant at the top of
[`settings.py`](settings.py): rep thresholds, targets, pain scales /
weights / level cut-offs, baseline duration, visibility limits, smoothing.

**Left/right:** the preview is mirrored (most people find a mirror natural), but
"left leg" always means the *patient's anatomical left* in the UI, overlays,
and CSVs. The single place that reconciles mirroring with MediaPipe's
landmark naming is `vision.landmark_side()`. Set `MIRROR_VIEW = False` in
settings.py to disable mirroring — nothing else needs to change.

## Layout

```
main.py        entry point
settings.py    every tunable constant + colors
vision.py      the one camera/MediaPipe thread + overlay drawing
metrics.py     knee angle, face features, pain scoring
exercises.py   rep counters + single-leg hold timer, organised into sets
session.py     per-frame session logic + summary
records.py     profiles + session CSV/video files
gui/           customtkinter pages and widgets
```
