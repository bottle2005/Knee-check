"""Camera + MediaPipe pipeline (single- or dual-camera).

Default: one webcam feeds both Pose (body) and FaceMesh (face).

Dual mode: a second camera — e.g. a phone placed near the user's face —
feeds FaceMesh, while the laptop webcam keeps feeding Pose.  The face camera
can be another device index (DroidCam/Iriun make phones show up as normal
cameras) or an ``http://…/video`` URL (the Android "IP Webcam" app streaming
over local WiFi; no internet involved).

Threads: one for the body camera (+Pose, and FaceMesh in single mode) and,
in dual mode, one for the face camera (+FaceMesh).  The GUI only ever polls
:meth:`CameraWorker.latest`.

Left/right note: with a mirrored body view, MediaPipe labels the user's
anatomical LEFT joints as its "right" landmarks.  :func:`landmark_side` is
the only place in the app that knows this.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np

import settings
from metrics import joint_angle

_pose_mod = mp.solutions.pose
_face_mod = mp.solutions.face_mesh
_draw_mod = mp.solutions.drawing_utils

_PL = _pose_mod.PoseLandmark

CameraSource = Union[int, str]


class CameraUnavailable(RuntimeError):
    """The (body) webcam could not be opened."""


def landmark_side(anatomical: str) -> str:
    """Map the user's anatomical side to MediaPipe's landmark naming."""
    if settings.MIRROR_VIEW:
        return "LEFT" if anatomical == "right" else "RIGHT"
    return anatomical.upper()


@dataclass(frozen=True)
class LegReading:
    """One leg's knee flexion + landmark positions (normalized coords).

    ``knee_angle`` follows the clinical convention: **0 deg = fully straight
    (extension)** and the value grows as the knee bends (deep bend ~ 130 deg).
    It is the supplement of the raw hip-knee-ankle interior angle.
    """

    side: str                       # anatomical side
    hip: Tuple[float, float]
    knee: Tuple[float, float]
    ankle: Tuple[float, float]
    knee_angle: float               # flexion, degrees (0 = straight)
    worst_visibility: float

    @property
    def trackable(self) -> bool:
        return self.worst_visibility >= settings.JOINT_VISIBILITY_MIN


def read_leg(pose_lms: Sequence, anatomical: str) -> LegReading:
    """Extract hip/knee/ankle and the knee flexion for the given side."""
    tag = landmark_side(anatomical)
    hip = pose_lms[getattr(_PL, f"{tag}_HIP").value]
    knee = pose_lms[getattr(_PL, f"{tag}_KNEE").value]
    ankle = pose_lms[getattr(_PL, f"{tag}_ANKLE").value]
    interior = joint_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    return LegReading(
        side=anatomical,
        hip=(hip.x, hip.y),
        knee=(knee.x, knee.y),
        ankle=(ankle.x, ankle.y),
        knee_angle=180.0 - interior,          # clinical flexion angle
        worst_visibility=min(hip.visibility, knee.visibility, ankle.visibility),
    )


@dataclass(frozen=True)
class Snapshot:
    """The newest processed body frame (+ face data, possibly from cam #2)."""

    frame_id: int
    bgr: np.ndarray                            # body camera frame
    pose_raw: object = None
    pose_lms: Optional[Sequence] = None
    face_raw: object = None
    face_lms: Optional[Sequence] = None
    face_bgr: Optional[np.ndarray] = None      # face camera frame (dual mode)
    taken_at: float = field(default_factory=time.monotonic)

    @property
    def dual(self) -> bool:
        return self.face_bgr is not None


# ---------------------------------------------------------------------------
# capture helpers
# ---------------------------------------------------------------------------

def _open_capture(source: CameraSource) -> cv2.VideoCapture:
    if isinstance(source, str) and not source.isdigit():
        return cv2.VideoCapture(source)                 # network stream / file
    idx = int(source)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(idx)
    return cap


def probe_cameras(max_index: int = 5, skip: Optional[int] = None) -> List[int]:
    """Device indices that currently open and deliver a frame.

    ``skip`` is assumed available without probing (the camera we already own).
    """
    good: List[int] = []
    for i in range(max_index + 1):
        if i == skip:
            good.append(i)
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            good.append(i)
    return good


def _shrink(frame: np.ndarray, max_w: int = 640) -> np.ndarray:
    """Downscale big (phone) frames so FaceMesh stays fast."""
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    k = max_w / w
    return cv2.resize(frame, (max_w, int(h * k)))


_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class _FaceFeed(threading.Thread):
    """Dual mode: reads the face camera and runs FaceMesh on it.

    ``rotation`` (degrees clockwise) is applied *before* detection, so a phone
    mounted sideways still yields an upright face - both for the on-screen
    inset and for the pain features, which assume an upright face.
    """

    def __init__(self, source: CameraSource,
                 rotation: int = settings.FACE_ROTATION) -> None:
        super().__init__(name="face-camera", daemon=True)
        self._rotate_code = _ROTATIONS.get(int(rotation) % 360)
        self._cap = _open_capture(source)
        self.ok = self._cap.isOpened()
        self._mesh = _face_mod.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        self._lock = threading.Lock()
        self._latest: Tuple[Optional[np.ndarray], object, Optional[Sequence]] = (None, None, None)
        self._stop = threading.Event()
        if self.ok:
            self.start()

    def latest(self) -> Tuple[Optional[np.ndarray], object, Optional[Sequence]]:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout=3.0)
        self._cap.release()
        self._mesh.close()

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.25)
                continue
            if self._rotate_code is not None:
                frame = cv2.rotate(frame, self._rotate_code)
            frame = _shrink(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = self._mesh.process(rgb)
            lms = (res.multi_face_landmarks[0].landmark
                   if res.multi_face_landmarks else None)
            with self._lock:
                self._latest = (frame, res, lms)


class CameraWorker:
    """Owns the body camera thread and, in dual mode, the face camera thread."""

    def __init__(self) -> None:
        cfg = settings.load_camera_config()
        self.body_source: CameraSource = cfg.get("body", settings.CAMERA_INDEX)
        self.face_source: Optional[CameraSource] = cfg.get("face")
        self.face_rotation: int = settings.FACE_ROTATION

        self._cap = _open_capture(self.body_source)
        if not self._cap.isOpened():
            raise CameraUnavailable(
                "The body camera could not be opened. Plug in a webcam "
                "(or close the app using it) and retry."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAPTURE_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAPTURE_H)

        self._pose = _pose_mod.Pose(model_complexity=1, smooth_landmarks=True,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

        # Face pipeline: second camera if configured and reachable,
        # otherwise FaceMesh runs on the body camera (single mode).
        self._face_feed: Optional[_FaceFeed] = None
        self._own_mesh: Optional[_face_mod.FaceMesh] = None
        self.face_feed_ok = False
        if self.face_source is not None:
            feed = _FaceFeed(self.face_source, rotation=self.face_rotation)
            if feed.ok:
                self._face_feed = feed
                self.face_feed_ok = True
            else:
                feed.close()
        if self._face_feed is None:
            self._own_mesh = _face_mod.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._snapshot: Optional[Snapshot] = None
        self.healthy = True

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="body-camera", daemon=True)
        self._thread.start()

    @property
    def dual(self) -> bool:
        return self._face_feed is not None

    def latest(self) -> Optional[Snapshot]:
        with self._lock:
            return self._snapshot

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._cap.release()
        self._pose.close()
        if self._own_mesh is not None:
            self._own_mesh.close()
        if self._face_feed is not None:
            self._face_feed.close()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self.healthy = False
                time.sleep(0.25)
                continue
            self.healthy = True

            if settings.MIRROR_VIEW:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] != settings.CAPTURE_W or frame.shape[0] != settings.CAPTURE_H:
                frame = cv2.resize(frame, (settings.CAPTURE_W, settings.CAPTURE_H))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = self._pose.process(rgb)

            if self._face_feed is not None:
                face_bgr, face_res, face_lms = self._face_feed.latest()
            else:
                face_res = self._own_mesh.process(rgb)
                face_lms = (face_res.multi_face_landmarks[0].landmark
                            if face_res.multi_face_landmarks else None)
                face_bgr = None

            snap = Snapshot(
                frame_id=next(self._ids),
                bgr=frame,
                pose_raw=pose_res,
                pose_lms=pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None,
                face_raw=face_res,
                face_lms=face_lms,
                face_bgr=face_bgr,
            )
            with self._lock:
                self._snapshot = snap


# ---------------------------------------------------------------------------
# drawing helpers (BGR, OpenCV)
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_skeleton(frame: np.ndarray, pose_raw) -> None:
    if pose_raw is None or not pose_raw.pose_landmarks:
        return
    _draw_mod.draw_landmarks(
        frame, pose_raw.pose_landmarks, _pose_mod.POSE_CONNECTIONS,
        _draw_mod.DrawingSpec(color=(191, 219, 45), thickness=2, circle_radius=3),
        _draw_mod.DrawingSpec(color=(230, 230, 230), thickness=2, circle_radius=2),
    )


def draw_angle_tag(frame: np.ndarray, knee_xy: Tuple[float, float], angle: float) -> None:
    """A rounded dark chip with the knee angle, anchored at the knee."""
    h, w = frame.shape[:2]
    x, y = int(knee_xy[0] * w), int(knee_xy[1] * h)
    text = f"{angle:.0f} deg"
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.7, 1)
    x1, y1 = x + 14, y - th - 22
    x1 = min(max(4, x1), w - tw - 20)
    y1 = min(max(4, y1), h - th - 16)
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 16, y1 + th + 14), (30, 30, 30), -1)
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 16, y1 + th + 14), (191, 219, 45), 1)
    cv2.putText(frame, text, (x1 + 8, y1 + th + 5), _FONT, 0.7, (191, 219, 45), 1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 8, (191, 219, 45), 2)


def draw_face_inset(frame: np.ndarray, face_lms: Sequence,
                    face_source: Optional[np.ndarray] = None, size: int = 150) -> None:
    """Zoomed face view in the top-right corner.

    Crops around the detected face — from ``face_source`` (the phone camera's
    frame, in dual mode) or from ``frame`` itself (single mode).
    """
    src = face_source if face_source is not None else frame
    sh, sw = src.shape[:2]
    xs = [p.x for p in face_lms]
    ys = [p.y for p in face_lms]
    pad = 25
    x1, y1 = max(0, int(min(xs) * sw) - pad), max(0, int(min(ys) * sh) - pad)
    x2, y2 = min(sw, int(max(xs) * sw) + pad), min(sh, int(max(ys) * sh) + pad)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return
    crop = cv2.resize(src[y1:y2, x1:x2], (size, size))
    h, w = frame.shape[:2]
    ox, oy = w - size - 16, 16
    frame[oy:oy + size, ox:ox + size] = crop
    cv2.rectangle(frame, (ox, oy), (ox + size, oy + size), (230, 230, 230), 2)
    if face_source is not None:
        cv2.putText(frame, "phone", (ox + 4, oy + size - 6), _FONT, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)


def draw_topbar(frame: np.ndarray, left_text: str, right_text: str = "") -> None:
    """Semi-transparent strip along the top with session info."""
    h, w = frame.shape[:2]
    strip = frame[0:44, 0:w].copy()
    cv2.rectangle(strip, (0, 0), (w, 44), (12, 18, 32), -1)
    cv2.addWeighted(strip, 0.65, frame[0:44, 0:w], 0.35, 0, frame[0:44, 0:w])
    cv2.putText(frame, left_text, (16, 29), _FONT, 0.65, (238, 242, 247), 1, cv2.LINE_AA)
    if right_text:
        (tw, _), _ = cv2.getTextSize(right_text, _FONT, 0.65, 1)
        cv2.putText(frame, right_text, (w - tw - 180, 29), _FONT, 0.65,
                    (147, 164, 189), 1, cv2.LINE_AA)


def draw_notice(frame: np.ndarray, text: str, good: bool = False) -> None:
    """Big centred message near the bottom (warnings / cheers)."""
    h, w = frame.shape[:2]
    color = (128, 222, 74) if good else (113, 113, 248)   # BGR of OK / BAD
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.85, 1)
    x = (w - tw) // 2
    y = h - 34
    box = frame[y - th - 12:y + 12, x - 16:x + tw + 16]
    if box.size:
        dark = box.copy()
        dark[:] = (12, 18, 32)
        cv2.addWeighted(dark, 0.6, box, 0.4, 0, box)
    cv2.putText(frame, text, (x, y), _FONT, 0.85, color, 1, cv2.LINE_AA)
