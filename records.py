"""Disk records: user profiles and per-session output files.

Privacy: by default a session stores only landmark coordinates and derived
numbers.  ``video=True`` (the explicit opt-in) is the only path that writes
pixels to disk.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

import settings
from metrics import SUBSCORE_NAMES

# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def safe_folder_name(name: str) -> str:
    """Display name → filesystem-safe folder name (no separators or dots)."""
    kept = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " _-")
    return re.sub(r"\s+", "_", kept.strip())[:48]


@dataclass
class Profile:
    """A user's stored identity + neutral-face baseline."""

    name: str
    folder: Path
    created: str = ""
    baseline_taken: str = ""
    baseline: Optional[Dict[str, float]] = field(default=None)

    @property
    def has_baseline(self) -> bool:
        return bool(self.baseline)


class ProfileBook:
    """CRUD for profiles under ``profiles/<name>/profile.json``."""

    def __init__(self, root: Path = settings.PROFILES_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _json_path(self, folder_name: str) -> Path:
        return self.root / folder_name / "profile.json"

    def names(self) -> List[str]:
        return sorted(p.parent.name for p in self.root.glob("*/profile.json"))

    def create(self, display_name: str) -> Profile:
        """New profile; raises ValueError for empty or duplicate names."""
        folder_name = safe_folder_name(display_name)
        if not folder_name:
            raise ValueError("Please type a name with letters or numbers in it.")
        if self._json_path(folder_name).exists():
            raise ValueError(
                f'"{folder_name}" already exists - open it from Welcome Back, '
                "or pick a different name."
            )
        profile = Profile(
            name=folder_name,
            folder=self.root / folder_name,
            created=datetime.now().isoformat(timespec="seconds"),
        )
        profile.folder.mkdir(parents=True, exist_ok=True)
        self._flush(profile)
        return profile

    def open(self, display_name: str) -> Profile:
        folder_name = safe_folder_name(display_name)
        path = self._json_path(folder_name)
        if not path.exists():
            raise FileNotFoundError(f'No profile called "{folder_name}" yet.')
        data = json.loads(path.read_text(encoding="utf-8"))
        return Profile(
            name=data.get("name", folder_name),
            folder=path.parent,
            created=data.get("created", ""),
            baseline_taken=data.get("baseline_taken", ""),
            baseline=data.get("baseline"),
        )

    def store_baseline(self, profile: Profile, baseline: Dict[str, float]) -> None:
        if not baseline:
            raise ValueError("Empty baseline - nothing was saved.")
        profile.baseline = dict(baseline)
        profile.baseline_taken = datetime.now().isoformat(timespec="seconds")
        self._flush(profile)

    def _flush(self, profile: Profile) -> None:
        self._json_path(profile.name).write_text(json.dumps({
            "name": profile.name,
            "created": profile.created,
            "baseline_taken": profile.baseline_taken,
            "baseline": profile.baseline,
        }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# session history
# ---------------------------------------------------------------------------

def list_past_sessions(user: Optional[str] = None) -> List[Dict[str, str]]:
    """All saved sessions (newest first) as summary dicts.

    Each dict is the session's ``summary.csv`` plus ``_when`` (a datetime)
    and ``_folder`` (absolute path).  Pass ``user`` to filter to one person.
    """
    found: List[Dict[str, str]] = []
    for summary_path in settings.SESSIONS_DIR.glob("*/summary.csv"):
        try:
            with open(summary_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except OSError:
            continue
        data: Dict[str, str] = {r[0]: r[1] for r in rows[1:] if len(r) >= 2}
        if user is not None and data.get("user") != user:
            continue
        tag_parts = summary_path.parent.name.split("_")
        try:
            when = datetime.strptime("_".join(tag_parts[-2:]), "%Y%m%d_%H%M%S")
        except ValueError:
            when = datetime.fromtimestamp(summary_path.stat().st_mtime)
        data["_when"] = when                       # type: ignore[assignment]
        data["_folder"] = str(summary_path.parent.resolve())
        found.append(data)
    found.sort(key=lambda d: d["_when"], reverse=True)
    return found


def delete_session(folder: str) -> None:
    """Permanently remove one session folder.

    Refuses anything that is not a direct child of the sessions directory,
    so a corrupted path can never delete something else.  Read-only files
    (common under OneDrive sync) are made writable and retried; a file still
    held open by another program raises a clear error instead.
    """
    import os
    import shutil
    import stat
    import time

    target = Path(folder).resolve()
    sessions_root = settings.SESSIONS_DIR.resolve()
    if target.parent != sessions_root or not target.is_dir():
        raise ValueError(f"Not a session folder: {target}")

    def _make_writable_and_retry(func, path, _exc_info) -> None:
        os.chmod(path, stat.S_IWRITE)
        func(path)

    last_error: Optional[BaseException] = None
    for attempt in range(3):                     # ride out brief OneDrive locks
        try:
            shutil.rmtree(target, onerror=_make_writable_and_retry)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise OSError(
        "Windows refused to delete some files in this session "
        f"({last_error}).\n\nIf any of its files are open (Excel, a video "
        "player, an Explorer window inside the folder), close them and try "
        "again. OneDrive sync can also hold files briefly - pausing sync helps."
    )


# ---------------------------------------------------------------------------
# session outputs
# ---------------------------------------------------------------------------
FRAME_COLUMNS = [
    "frame", "t_sec", "user", "exercise", "measured_leg",
    "knee_flexion_deg", "set", "reps_total", "hold_s",
    "pain", "pain_level", *SUBSCORE_NAMES,
    "leg_tracked", "face_tracked",
]


class SessionFiles:
    """Owns one session folder and its writers; call :meth:`close` once."""

    def __init__(self, user: str, exercise_key: str, video: bool) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tag = f"{user}_{exercise_key}_{stamp}"
        self.folder = settings.SESSIONS_DIR / self.tag
        self.folder.mkdir(parents=True, exist_ok=True)
        self.video_enabled = video

        self._frames_f = open(self.folder / "frames.csv", "w", newline="", encoding="utf-8")
        self._frames = csv.writer(self._frames_f)
        self._frames.writerow(FRAME_COLUMNS)

        self._face_f = open(self.folder / "face_points.csv", "w", newline="", encoding="utf-8")
        self._face = csv.writer(self._face_f)
        head = ["frame", "t_sec"]
        for i in range(468):
            head += [f"x{i}", f"y{i}", f"z{i}"]
        self._face.writerow(head)

        self._video: Optional[cv2.VideoWriter] = None
        if video:
            self._video = cv2.VideoWriter(
                str(self.folder / "capture.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                settings.VIDEO_FPS_NOMINAL,
                (settings.CAPTURE_W, settings.CAPTURE_H),
            )
        self._open = True

    def frame_row(self, values: Sequence) -> None:
        if self._open:
            self._frames.writerow(values)

    def face_row(self, frame_no: int, t_sec: float, lms: Sequence) -> None:
        if not self._open:
            return
        row: list = [frame_no, round(t_sec, 3)]
        for p in lms:
            row += [p.x, p.y, p.z]
        self._face.writerow(row)

    def video_frame(self, bgr: np.ndarray) -> None:
        if self._video is not None and self._open:
            self._video.write(bgr)

    def summary(self, data: Dict[str, object]) -> None:
        with open(self.folder / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for key, value in data.items():
                w.writerow([key, value])

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._frames_f.close()
        self._face_f.close()
        if self._video is not None:
            self._video.release()
            self._video = None
