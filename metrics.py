"""Derived measurements: knee angle, neutral-face baseline, pain score.

The pain score is a facial proxy: a weighted, per-user deviation from a
neutral-face baseline across five expression features (brow lowering, brow
knitting, eye squeezing, lip pressing, mouth-corner droop).  Everything is
normalized by eye-to-eye distance so head distance from the camera does not
matter.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import settings

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Angle in degrees at vertex ``b`` of triangle a-b-c."""
    a, b, c = (np.asarray(p, dtype=np.float64) for p in (a, b, c))
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def dist(p: Sequence[float], q: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))


class Smoother:
    """Fixed-window moving average."""

    def __init__(self, window: int) -> None:
        self._buf: deque = deque(maxlen=window)

    def push(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))


# ---------------------------------------------------------------------------
# FaceMesh landmark indices (MediaPipe canonical face model)
# ---------------------------------------------------------------------------
_L_EYE_OUT, _L_EYE_IN, _L_LID_TOP, _L_LID_BOT = 33, 133, 159, 145
_R_EYE_OUT, _R_EYE_IN, _R_LID_TOP, _R_LID_BOT = 263, 362, 386, 374
_L_BROW_IN, _L_BROW_MID = 55, 105
_R_BROW_IN, _R_BROW_MID = 285, 334
_MOUTH_L, _MOUTH_R = 61, 291
_LIP_TOP, _LIP_BOT = 13, 14

FaceFeatures = Dict[str, float]
SUBSCORE_NAMES: List[str] = list(settings.PAIN_WEIGHTS.keys())


def _pt(lms: Sequence, idx: int) -> np.ndarray:
    return np.array([lms[idx].x, lms[idx].y], dtype=np.float64)


def face_features(lms: Sequence) -> FaceFeatures:
    """Expression features normalized by eye-to-eye (outer corner) distance."""
    scale = max(dist(_pt(lms, _L_EYE_OUT), _pt(lms, _R_EYE_OUT)), 1e-6)

    def eye_open(top: int, bot: int, outc: int, inc: int) -> float:
        return dist(_pt(lms, top), _pt(lms, bot)) / max(dist(_pt(lms, outc), _pt(lms, inc)), 1e-6)

    eye_openness = (
        eye_open(_L_LID_TOP, _L_LID_BOT, _L_EYE_OUT, _L_EYE_IN)
        + eye_open(_R_LID_TOP, _R_LID_BOT, _R_EYE_OUT, _R_EYE_IN)
    ) / 2.0

    brow_height = (
        dist(_pt(lms, _L_BROW_MID), _pt(lms, _L_LID_TOP))
        + dist(_pt(lms, _R_BROW_MID), _pt(lms, _R_LID_TOP))
    ) / 2.0 / scale

    brow_gap = dist(_pt(lms, _L_BROW_IN), _pt(lms, _R_BROW_IN)) / scale

    mouth_openness = dist(_pt(lms, _LIP_TOP), _pt(lms, _LIP_BOT)) / max(
        dist(_pt(lms, _MOUTH_L), _pt(lms, _MOUTH_R)), 1e-6
    )

    lip_mid_y = (_pt(lms, _LIP_TOP)[1] + _pt(lms, _LIP_BOT)[1]) / 2.0
    corner_droop = (
        (_pt(lms, _MOUTH_L)[1] - lip_mid_y) + (_pt(lms, _MOUTH_R)[1] - lip_mid_y)
    ) / 2.0 / scale

    return {
        "eye_openness": eye_openness,
        "brow_height": brow_height,
        "brow_gap": brow_gap,
        "mouth_openness": mouth_openness,
        "corner_droop": corner_droop,
    }


def mean_features(samples: Sequence[FaceFeatures]) -> FaceFeatures:
    """Element-wise mean — this is what gets stored as the user's baseline."""
    return {k: float(np.mean([s[k] for s in samples])) for k in samples[0]}


def pain_score(current: FaceFeatures,
                  baseline: Optional[FaceFeatures]) -> Tuple[float, Dict[str, float]]:
    """Weighted deviation of ``current`` from ``baseline`` → (score, sub-scores).

    Each sub-score is the one-sided deviation in the pain direction, scaled
    by settings.PAIN_SCALES and clipped to 0..1.
    """
    if baseline is None:
        return 0.0, {name: 0.0 for name in SUBSCORE_NAMES}

    deviations = {
        "brow_lower": baseline["brow_height"] - current["brow_height"],
        "brow_knit": baseline["brow_gap"] - current["brow_gap"],
        "eye_squeeze": baseline["eye_openness"] - current["eye_openness"],
        "lip_press": baseline["mouth_openness"] - current["mouth_openness"],
        "corner_droop": current["corner_droop"] - baseline["corner_droop"],
    }
    subs = {
        name: float(np.clip(deviations[name] / settings.PAIN_SCALES[name], 0.0, 1.0))
        for name in SUBSCORE_NAMES
    }
    score = float(np.clip(
        sum(settings.PAIN_WEIGHTS[name] * subs[name] for name in SUBSCORE_NAMES),
        0.0, 1.0,
    ))
    return score, subs


def pain_level(score: float) -> str:
    if score >= settings.PAIN_HIGH:
        return "High"
    if score >= settings.PAIN_MODERATE:
        return "Moderate"
    return "Low"
