"""KneeCheck — every tunable value in one place.

Clinicians: the numbers you are most likely to adjust are grouped at the top.
Angles are degrees, times are seconds, distances are fractions of image size
(0..1) unless stated otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

# ===========================================================================
# EXERCISE TARGETS & THRESHOLDS  (clinician tunable)
# ===========================================================================
# NOTE ON KNEE ANGLES -------------------------------------------------------
# All knee angles below are CLINICAL FLEXION angles:
#     0 deg   = leg fully straight (full extension)
#     90 deg  = knee bent to a right angle (e.g. sitting)
#     ~130 deg = deep bend
# (Internally this is 180 deg minus the raw hip-knee-ankle angle; the
# conversion happens once, in vision.read_leg.)
# ---------------------------------------------------------------------------

REP_TARGET = 10                 # default reps per set for Knee ROM

# --- Sit to Stand (timed 5x sit-to-stand) ---------------------------------
# The standard test is 5 repetitions; the score is how many SECONDS the 5
# reps take.  Only the number of sets is adjustable in the UI.
STS_REPS_PER_SET = 5
STS_BENT_ANGLE = 70.0           # seated: knee flexed past this
STS_STRAIGHT_ANGLE = 30.0       # standing: knee straighter than this

# --- Knee Range of Motion --------------------------------------------------
# ROM score = deepest flexion - straightest extension over the session.
ROM_FLEX_ANGLE = 60.0           # bend past this to start a rep
ROM_EXTEND_ANGLE = 20.0         # straighten below this to complete a rep

# --- Single Leg Stance -----------------------------------------------------
# Both legs are measured, one after the other.  Timing starts when the free
# foot leaves the ground and stops when it touches down again (or at the cap).
SLS_MAX_HOLD_S = 60.0           # stop automatically at this time
SLS_TOUCHDOWN_GRACE_S = 0.4     # foot must stay down this long to end the try

REP_DEBOUNCE_S = 1.0            # minimum time between two counted reps
REST_BETWEEN_SETS_S = 8.0       # rest countdown between sets

# Limits for the sets/reps pickers on the options page
MAX_SETS = 10
MAX_REPS_PER_SET = 30
MAX_HOLD_PER_SET = 60           # seconds

# Single Leg Stance: the free ankle must be this much higher than the stance
# ankle (fraction of frame height), with the free knee bent at least this much.
LIFT_MIN_ANKLE_GAP = 0.05
LIFT_KNEE_BEND_MIN = 20.0       # free-knee flexion required to count as lifted

# Pain-level cut-offs (score is 0..1 deviation from the neutral baseline)
PAIN_MODERATE = 0.20
PAIN_HIGH = 0.45

# Per-feature scales: how much raw deviation counts as "fully tense" (1.0).
# Larger scale = less sensitive (more facial change needed to score).
PAIN_SCALES = {
    "brow_lower": 0.030,
    "brow_knit": 0.045,
    "eye_squeeze": 0.150,
    "lip_press": 0.075,
    "corner_droop": 0.023,
}
# Weights must sum to 1.0.
PAIN_WEIGHTS = {
    "brow_lower": 0.30,
    "brow_knit": 0.20,
    "eye_squeeze": 0.25,
    "lip_press": 0.10,
    "corner_droop": 0.15,
}

# Baseline capture
BASELINE_SECONDS = 5.0
BASELINE_MIN_SAMPLES = 25       # refuse to store a baseline thinner than this
BASELINE_TIMEOUT_S = 30.0       # give up (nothing saved) after this long

# Tracking quality
JOINT_VISIBILITY_MIN = 0.5      # each of hip/knee/ankle must beat this
LOST_LEG_GRACE_FRAMES = 12      # frames of poor visibility before warning
ANGLE_SMOOTHING = 7             # moving-average window (frames)
PAIN_SMOOTHING = 5

# ===========================================================================
# CAMERA / DISPLAY
# ===========================================================================
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 960, 540
VIDEO_FPS_NOMINAL = 20.0        # nominal FPS written into the opt-in video
UI_TICK_MS = 30

# Show the camera like a mirror (most users find this natural).  All left/right
# language in the app means the PATIENT'S anatomical side; the single point that
# reconciles this with the mirrored image is vision.landmark_side().
MIRROR_VIEW = True

# --- optional second camera for the face (e.g. a phone) --------------------
# Persisted in camera_config.json so it survives restarts.
#   body: device index of the body camera (laptop webcam)
#   face: None  -> single camera (face read from the body camera)
#         int   -> another device index (DroidCam / Iriun phone, USB cam …)
#         "http://…" -> network stream URL (Android "IP Webcam" app, local WiFi)
CAMERA_CONFIG_FILE = Path(__file__).resolve().parent / "camera_config.json"

# Phone cameras stream sideways, so the face feed is rotated by this many
# degrees clockwise before face detection (the pain features assume an upright
# face).  Change here if a different phone mount needs 180 or 270.
FACE_ROTATION = 90


def load_camera_config() -> Dict[str, Optional[Union[int, str]]]:
    """Stored camera choices (the face feed always uses FACE_ROTATION)."""
    try:
        data = json.loads(CAMERA_CONFIG_FILE.read_text(encoding="utf-8"))
        return {"body": data.get("body", CAMERA_INDEX), "face": data.get("face")}
    except (OSError, ValueError, TypeError):
        return {"body": CAMERA_INDEX, "face": None}


def save_camera_config(body: Union[int, str], face: Optional[Union[int, str]]) -> None:
    CAMERA_CONFIG_FILE.write_text(
        json.dumps({"body": body, "face": face}, indent=2), encoding="utf-8"
    )


# ===========================================================================
# FILES
# ===========================================================================
APP_NAME = "KneeCheck"
BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
SESSIONS_DIR = BASE_DIR / "sessions"

# ===========================================================================
# LOOK & FEEL
# ===========================================================================
CLR_BG = "#0b1220"
CLR_PANEL = "#141e30"
CLR_PANEL_2 = "#1d2a40"
CLR_TEXT = "#eef2f7"
CLR_MUTED = "#93a4bd"
CLR_ACCENT = "#2dd4bf"          # teal — main actions
CLR_ACCENT_DARK = "#14b8a6"
CLR_STAR = "#fbbf24"            # gold — rep stars
CLR_OK = "#4ade80"
CLR_WARN = "#fbbf24"
CLR_BAD = "#f87171"

FONT = "Segoe UI"
