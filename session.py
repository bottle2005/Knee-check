"""LiveSession: turns camera snapshots into scores, overlays, and CSV rows.

The GUI feeds every *new* snapshot (deduplicated by ``frame_id``) into
:meth:`ingest`, which returns the annotated frame plus a state dict for the
widgets.  MediaPipe inference already happened on the camera thread; the work
here (arithmetic, drawing, CSV rows) is cheap enough for the UI timer.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import exercises
import metrics
import settings
import vision
from records import Profile, SessionFiles


_CHEERS = [
    (0.75, "Nearly there - keep it up!"),
    (0.50, "Halfway! Keep going!"),
    (0.25, "Great start!"),
    (0.00, "Ready when you are!"),
]


class LiveSession:
    """One assessment run for one user."""

    def __init__(self, profile: Profile, leg: str, exercise_key: str, video: bool,
                 sets: int = 1, quota: Optional[int] = None,
                 variant: str = "") -> None:
        self.profile = profile
        self.exercise = exercises.make(exercise_key, sets=sets, quota=quota,
                                       first_leg=leg)
        self.variant = variant      # STS: "both" / "left" / "right"; else ""
        self.files = SessionFiles(profile.name, exercise_key, video)
        # Two-legged tests move the measured side themselves.
        self.leg = self.exercise.current_leg if self.exercise.two_legged else leg
        # Per-leg flexion ranges, so two-legged tests can report each side.
        self._angles_by_leg: Dict[str, List[float]] = {}

        self._angle_smooth = metrics.Smoother(settings.ANGLE_SMOOTHING)
        self._pain_smooth = metrics.Smoother(settings.PAIN_SMOOTHING)
        self._started = time.monotonic()
        self._frames = 0
        self._leg_missing_streak = 0

        self._angles: List[float] = []
        self._pains: List[float] = []
        self._face_frames = 0
        self._leg_frames = 0

        self._closed = False

    # ------------------------------------------------------------------
    def ingest(self, snap: vision.Snapshot) -> Tuple[np.ndarray, Dict]:
        """Process one new snapshot; returns (annotated frame, ui state)."""
        frame = snap.bgr.copy()
        self._frames += 1
        t = time.monotonic() - self._started

        # -- facial pain score ------------------------------------------------
        pain = 0.0
        subs = {name: 0.0 for name in metrics.SUBSCORE_NAMES}
        face_ok = snap.face_lms is not None
        if face_ok:
            self._face_frames += 1
            raw, subs = metrics.pain_score(
                metrics.face_features(snap.face_lms), self.profile.baseline
            )
            pain = self._pain_smooth.push(raw)
            self._pains.append(pain)
            self.files.face_row(self._frames, t, snap.face_lms)
        level = metrics.pain_level(pain)

        # two-legged tests move the measured side themselves
        if self.exercise.two_legged and self.exercise.current_leg != self.leg:
            self.leg = self.exercise.current_leg
            self._angle_smooth = metrics.Smoother(settings.ANGLE_SMOOTHING)

        # -- knee angle on the chosen leg only -------------------------------
        angle: Optional[float] = None
        leg_ok = False
        if snap.pose_lms is not None:
            reading = vision.read_leg(snap.pose_lms, self.leg)
            if reading.trackable:
                leg_ok = True
                self._leg_frames += 1
                self._leg_missing_streak = 0
                angle = self._angle_smooth.push(reading.knee_angle)
                self._angles.append(angle)
                self._angles_by_leg.setdefault(self.leg, []).append(angle)
                self._advance(snap, angle, t)
            else:
                self._leg_missing_streak += 1
        else:
            self._leg_missing_streak += 1

        warning = ""
        if self._leg_missing_streak > settings.LOST_LEG_GRACE_FRAMES:
            warning = f"I can't see your {self.leg} leg - step back a little!"

        # -- draw -------------------------------------------------------------
        vision.draw_skeleton(frame, snap.pose_raw)
        if leg_ok and angle is not None and snap.pose_lms is not None:
            vision.draw_angle_tag(frame, vision.read_leg(snap.pose_lms, self.leg).knee, angle)
        if face_ok:
            vision.draw_face_inset(frame, snap.face_lms, face_source=snap.face_bgr)
        if self.variant == "both":
            leg_text = f"both legs (measuring {self.leg})"
        else:
            leg_text = f"{self.leg} leg"
        vision.draw_topbar(
            frame,
            f"{self.profile.name}  |  {self.exercise.title}  |  {leg_text}",
            f"{t:5.1f}s",
        )
        if self.exercise.done:
            vision.draw_notice(frame, "All done - great work!", good=True)
        elif getattr(self.exercise, "awaiting_next_leg", False):
            vision.draw_notice(
                frame,
                f"{self.leg.capitalize()} leg: "
                f"{self.exercise.results.get(self.leg, 0):.1f}s  -  "
                "press 'Start next leg'",
                good=True,
            )
        elif warning:
            vision.draw_notice(frame, warning, good=False)
        elif self.exercise.resting:
            vision.draw_notice(
                frame,
                f"Rest time!  Set {self.exercise.set_index + 1} starts in "
                f"{self.exercise.rest_remaining:.0f}s",
                good=True,
            )
        else:
            vision.draw_notice(frame, self._cheer(), good=True)

        # -- record ------------------------------------------------------------
        is_hold = isinstance(self.exercise, exercises.SingleLegStance)
        reps = "" if is_hold else self.exercise.total_reps
        reps_in_set = "" if is_hold else self.exercise.reps_in_set
        hold = self.exercise.hold_now_s if is_hold else ""
        self.files.frame_row([
            self._frames, round(t, 3), self.profile.name, self.exercise.key, self.leg,
            round(angle, 2) if angle is not None else "",
            self.exercise.set_index + 1,
            reps, round(hold, 2) if hold != "" else "",
            round(pain, 3), level,
            *(round(subs[n], 3) for n in metrics.SUBSCORE_NAMES),
            leg_ok, face_ok,
        ])
        self.files.video_frame(frame)

        return frame, {
            "t": t,
            "angle": angle,
            "reps": reps,
            "reps_in_set": reps_in_set,
            "hold": hold,
            "set_index": self.exercise.set_index,
            "sets_planned": self.exercise.sets_planned,
            "quota": self.exercise.quota,
            "resting": self.exercise.resting,
            "rest_remaining": self.exercise.rest_remaining,
            "pain": pain,
            "pain_level": level,
            "leg_ok": leg_ok,
            "face_ok": face_ok,
            "warning": warning,
            "progress": self.exercise.progress,
            "caption": self.exercise.progress_caption,
            "done": self.exercise.done,
            "measured_leg": self.leg,
            "two_legged": self.exercise.two_legged,
            "awaiting_next_leg": getattr(self.exercise, "awaiting_next_leg", False),
            "leg_phase": getattr(self.exercise, "phase", ""),
            "leg_results": dict(getattr(self.exercise, "results", {})),
            "set_elapsed": getattr(self.exercise, "set_elapsed_s", 0.0),
            "timed": getattr(self.exercise, "timed", False),
        }

    @property
    def free_leg(self) -> str:
        """The non-measured leg (the one that lifts in a single-leg stance)."""
        return "right" if self.leg == "left" else "left"

    def start_next_leg(self) -> None:
        """UI hook: arm the second leg of a two-legged test."""
        if isinstance(self.exercise, exercises.SingleLegStance):
            self.exercise.start_next_leg()

    def stop_current_leg(self) -> None:
        """UI hook: end the leg currently being measured."""
        if isinstance(self.exercise, exercises.SingleLegStance):
            self.exercise.stop_current_leg(time.monotonic() - self._started)

    def _advance(self, snap: vision.Snapshot, angle: float, t: float) -> None:
        if isinstance(self.exercise, exercises.SingleLegStance):
            free = vision.read_leg(snap.pose_lms, self.free_leg)
            stance = vision.read_leg(snap.pose_lms, self.leg)
            lifted = (
                stance.ankle[1] - free.ankle[1] > settings.LIFT_MIN_ANKLE_GAP
                and free.knee_angle > settings.LIFT_KNEE_BEND_MIN
            )
            self.exercise.feed(lifted, t)
        else:
            self.exercise.feed(angle, t)

    def _cheer(self) -> str:
        for threshold, text in _CHEERS:
            if self.exercise.progress >= threshold:
                return text
        return ""

    # ------------------------------------------------------------------
    def finish(self, aborted: bool = False) -> Dict[str, object]:
        """Write the summary, close files, and return the summary dict."""
        if self._closed:
            return {}
        self._closed = True

        avg_t = float(np.mean(self._pains)) if self._pains else 0.0
        max_t = float(np.max(self._pains)) if self._pains else 0.0
        if max_t >= settings.PAIN_HIGH or avg_t >= settings.PAIN_MODERATE:
            difficulty = "High"
        elif max_t >= settings.PAIN_MODERATE:
            difficulty = "Moderate"
        else:
            difficulty = "Low"

        is_hold = isinstance(self.exercise, exercises.SingleLegStance)
        set_results = list(self.exercise.set_results)
        partial = self.exercise.current_partial if not self.exercise.done else 0.0

        summary: Dict[str, object] = {
            "user": self.profile.name,
            "exercise": self.exercise.key,
            "exercise_title": self.exercise.title,
            "measured_leg": self.leg,
            "leg_variant": self.variant,
            "session": self.files.tag,
            "aborted": aborted,
            "target_reached": self.exercise.done,
            "finished_at_s": (round(self.exercise.finished_at_s, 2)
                              if self.exercise.finished_at_s is not None else ""),
            "sets_planned": self.exercise.sets_planned,
            "quota_per_set": self.exercise.quota,
            "sets_completed": len(set_results),
            "set_results": " | ".join(f"{r:g}" for r in set_results) or "",
            "partial_set": round(partial, 2) if partial else "",
            "average_per_set": round(self.exercise.average_per_set, 2),
            "reps": ("" if is_hold else self.exercise.total_reps),
            "best_hold_s": (round(self.exercise.best_hold_s, 2) if is_hold else ""),
            # Flexion angles: 0 deg = straight.  min = best extension reached,
            # max = deepest bend reached.
            "knee_flexion_min": round(float(np.min(self._angles)), 2) if self._angles else "",
            "knee_flexion_avg": round(float(np.mean(self._angles)), 2) if self._angles else "",
            "knee_flexion_max": round(float(np.max(self._angles)), 2) if self._angles else "",
            "pain_avg": round(avg_t, 3),
            "pain_max": round(max_t, 3),
            "pain_peak_level": metrics.pain_level(max_t),
            "difficulty_rating": difficulty,
            "frames_total": self._frames,
            "frames_face": self._face_frames,
            "frames_leg_tracked": self._leg_frames,
            "video_saved": self.files.video_enabled,
            "saved_to": str(self.files.folder.resolve()),
        }

        # --- timed 5x sit-to-stand: seconds per set ------------------------
        if getattr(self.exercise, "timed", False):
            times = self.exercise.set_durations
            summary["sts_set_times_s"] = " | ".join(f"{d:.1f}" for d in times)
            summary["sts_avg_time_s"] = round(self.exercise.average_time_s, 2) if times else ""
            summary["sts_best_time_s"] = round(self.exercise.best_time_s, 2) if times else ""
            summary["sts_avg_rep_time_s"] = (round(self.exercise.average_rep_time_s, 2)
                                             if times else "")
            summary["reps_per_set"] = self.exercise.quota

        # --- two-legged tests: per-leg results ------------------------------
        if self.exercise.two_legged:
            results = self.exercise.results
            summary["leg_order"] = " → ".join(self.exercise.leg_order)
            for side in ("right", "left"):
                summary[f"hold_{side}_s"] = (round(results[side], 2)
                                             if side in results else "")
            if len(results) == 2:
                summary["hold_difference_s"] = round(
                    abs(results["right"] - results["left"]), 2)

        # --- range of motion: only meaningful for the ROM assessment --------
        if self.exercise.key == "knee_rom":
            summary["rom_deg"] = (
                round(float(np.max(self._angles) - np.min(self._angles)), 1)
                if self._angles else "")
            for side, values in self._angles_by_leg.items():
                if values:
                    summary[f"rom_{side}_deg"] = round(
                        float(np.max(values) - np.min(values)), 1)
            if len(self._angles_by_leg) == 2:
                roms = [summary.get(f"rom_{s}_deg") for s in ("right", "left")]
                if all(isinstance(v, float) for v in roms):
                    summary["rom_difference_deg"] = round(abs(roms[0] - roms[1]), 1)

        self.files.summary(summary)
        self.files.close()
        return summary
