"""The three assessments as small state machines.

* **Sit to Stand** - the timed 5x sit-to-stand: each set is 5 repetitions and
  the score is how many SECONDS the set takes.  The number of sets is
  configurable; a short rest separates them.
* **Knee ROM** - repetition counting over ``sets`` x ``quota`` reps.  The
  clinical score (deepest flexion minus straightest extension) is computed by
  the session from the logged angles.
* **Single Leg Stance** - both legs, one after the other.  Timing starts when
  the free foot lifts and stops when it touches down again (or at the 60 s
  cap).  Between legs the operator presses "Start next leg".

All knee angles are clinical FLEXION angles (0 deg = leg straight), and every
threshold lives in :mod:`settings`.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import settings


class Exercise:
    """Shared interface: feed measurements in, read progress out."""

    key: str = ""
    title: str = ""
    uses_reps: bool = True
    two_legged: bool = False         # measures both legs in one session
    default_quota: int = 10

    def __init__(self, sets: int = 1, quota: Optional[int] = None) -> None:
        self.sets_planned = max(1, int(sets))
        self.quota = int(quota) if quota else self.default_quota
        self.set_index = 0                       # 0-based current set
        self.set_results: List[float] = []       # achieved amount per set
        self.done = False
        self.finished_at_s: Optional[float] = None
        self._rest_until: Optional[float] = None

    # ---- rest handling ------------------------------------------------
    @property
    def resting(self) -> bool:
        return self._rest_until is not None and time.monotonic() < self._rest_until

    @property
    def rest_remaining(self) -> float:
        if self._rest_until is None:
            return 0.0
        return max(0.0, self._rest_until - time.monotonic())

    def _complete_set(self, result: float, elapsed_s: float) -> None:
        self.set_results.append(result)
        self.set_index += 1
        if self.set_index >= self.sets_planned:
            self.done = True
            self.finished_at_s = elapsed_s
        else:
            self._rest_until = time.monotonic() + settings.REST_BETWEEN_SETS_S

    # ---- reporting -----------------------------------------------------
    @property
    def current_partial(self) -> float:
        """Progress inside the not-yet-finished set."""
        raise NotImplementedError

    @property
    def average_per_set(self) -> float:
        """Mean result across attempted sets, counting a partial final set."""
        results = list(self.set_results)
        if not self.done:
            results.append(self.current_partial)
        return sum(results) / len(results) if results else 0.0

    @property
    def progress(self) -> float:            # 0..1 across the whole session
        total = self.sets_planned * self.quota
        achieved = sum(self.set_results) + (0 if self.done else self.current_partial)
        return min(1.0, achieved / total) if total else 0.0

    @property
    def progress_caption(self) -> str:
        raise NotImplementedError


# ===========================================================================
class _RepExercise(Exercise):
    """Counts reps from the knee FLEXION angle (0 deg = straight).

    A rep = flex past ``bent_angle``, then straighten below ``straight_angle``.
    The gap between the two provides hysteresis.
    """

    bent_angle: float = 0.0
    straight_angle: float = 0.0

    def __init__(self, sets: int = 1, quota: Optional[int] = None) -> None:
        super().__init__(sets, quota)
        self.reps_in_set = 0
        self._phase = "waiting"             # waiting | bent
        self._last_rep_at = 0.0
        self._set_started_at: Optional[float] = None   # first movement of the set
        self.set_durations: List[float] = []           # seconds taken per set

    @property
    def total_reps(self) -> int:
        return int(sum(self.set_results)) + self.reps_in_set

    @property
    def current_partial(self) -> float:
        return float(self.reps_in_set)

    @property
    def set_elapsed_s(self) -> float:
        """Seconds since the current set's first movement (0 before it)."""
        if self._set_started_at is None:
            return 0.0
        return time.monotonic() - self._set_started_at

    def feed(self, knee_angle: float, elapsed_s: float) -> bool:
        """Advance with the smoothed knee flexion; True when a rep lands."""
        if self.done:
            return False
        if self.resting:
            self._phase = "waiting"
            return False
        if self._rest_until is not None:     # rest just ended
            self._rest_until = None

        if self._phase == "waiting" and knee_angle > self.bent_angle:
            self._phase = "bent"
            if self._set_started_at is None:          # first movement of the set
                self._set_started_at = time.monotonic()
        elif self._phase == "bent" and knee_angle < self.straight_angle:
            now = time.monotonic()
            if now - self._last_rep_at >= settings.REP_DEBOUNCE_S:
                self.reps_in_set += 1
                self._last_rep_at = now
                self._phase = "waiting"
                if self.reps_in_set >= self.quota:
                    self.set_durations.append(self.set_elapsed_s)
                    self._complete_set(float(self.reps_in_set), elapsed_s)
                    self.reps_in_set = 0
                    self._set_started_at = None
                return True
        return False

    @property
    def progress_caption(self) -> str:
        if self.done:
            return f"All {self.sets_planned} sets done!"
        return (f"Set {self.set_index + 1} of {self.sets_planned}  ·  "
                f"{self.reps_in_set} / {self.quota} reps")


class SitToStand(_RepExercise):
    """Timed 5x sit-to-stand: fixed 5 reps per set, scored in seconds."""

    key = "sit_to_stand"
    title = "Sit to Stand (timed 5x)"
    bent_angle = settings.STS_BENT_ANGLE
    straight_angle = settings.STS_STRAIGHT_ANGLE
    default_quota = settings.STS_REPS_PER_SET
    timed = True

    def __init__(self, sets: int = 1, quota: Optional[int] = None) -> None:
        # the 5xSTS protocol fixes the repetitions per set
        super().__init__(sets, settings.STS_REPS_PER_SET)

    @property
    def average_time_s(self) -> float:
        """Mean seconds per completed set (0 if none finished)."""
        return (sum(self.set_durations) / len(self.set_durations)
                if self.set_durations else 0.0)

    @property
    def best_time_s(self) -> float:
        return min(self.set_durations) if self.set_durations else 0.0

    @property
    def average_rep_time_s(self) -> float:
        """Mean seconds for a single repetition (set time / reps per set)."""
        return self.average_time_s / self.quota if self.quota else 0.0

    @property
    def progress_caption(self) -> str:
        if self.done:
            return f"Done!  Average {self.average_time_s:.1f}s per 5 reps"
        return (f"Set {self.set_index + 1} of {self.sets_planned}  ·  "
                f"{self.reps_in_set} / {self.quota} reps  ·  {self.set_elapsed_s:.1f}s")


class KneeROM(_RepExercise):
    key = "knee_rom"
    title = "Knee Bends (ROM)"
    bent_angle = settings.ROM_FLEX_ANGLE
    straight_angle = settings.ROM_EXTEND_ANGLE
    default_quota = settings.REP_TARGET


# ===========================================================================
class SingleLegStance(Exercise):
    """Both legs, one after the other, each timed until the foot touches down.

    Flow per leg:  waiting for lift -> holding -> finished (foot down or the
    60 s cap).  After the first leg the operator presses "Start next leg"
    (:meth:`start_next_leg`), which arms the second leg.
    """

    key = "single_leg_stance"
    title = "Single Leg Stance"
    uses_reps = False
    two_legged = True
    default_quota = int(settings.SLS_MAX_HOLD_S)

    def __init__(self, first_leg: str = "right", sets: int = 1,
                 quota: Optional[int] = None) -> None:
        super().__init__(sets=1, quota=quota or int(settings.SLS_MAX_HOLD_S))
        second = "left" if first_leg == "right" else "right"
        self.leg_order: List[str] = [first_leg, second]
        self.leg_index = 0
        self.results: Dict[str, float] = {}        # leg -> seconds held
        self.phase = "waiting"                     # waiting | holding | leg_done
        self._lift_started: Optional[float] = None
        self._down_since: Optional[float] = None

    # ---- state ---------------------------------------------------------
    @property
    def current_leg(self) -> str:
        return self.leg_order[min(self.leg_index, len(self.leg_order) - 1)]

    @property
    def awaiting_next_leg(self) -> bool:
        """True when a leg is finished and the next one is not armed yet."""
        return self.phase == "leg_done" and not self.done

    @property
    def hold_now_s(self) -> float:
        """Live timer for the leg being measured."""
        if self.phase == "holding" and self._lift_started is not None:
            return min(time.monotonic() - self._lift_started, settings.SLS_MAX_HOLD_S)
        return self.results.get(self.current_leg, 0.0)

    @property
    def current_partial(self) -> float:
        return self.hold_now_s

    @property
    def best_hold_s(self) -> float:
        return max(self.results.values()) if self.results else 0.0

    # ---- driving -------------------------------------------------------
    def feed(self, free_leg_lifted: bool, elapsed_s: float) -> bool:
        """Advance with whether the free foot is currently off the ground."""
        if self.done or self.phase == "leg_done":
            return False
        now = time.monotonic()

        if self.phase == "waiting":
            if free_leg_lifted:
                self.phase = "holding"
                self._lift_started = now
                self._down_since = None
            return False

        # phase == "holding"
        held = now - self._lift_started if self._lift_started else 0.0
        if free_leg_lifted:
            self._down_since = None
        else:
            # foot must stay down briefly so a wobble doesn't end the attempt
            if self._down_since is None:
                self._down_since = now
            elif now - self._down_since >= settings.SLS_TOUCHDOWN_GRACE_S:
                self._finish_leg(held - settings.SLS_TOUCHDOWN_GRACE_S, elapsed_s)
                return True

        if held >= settings.SLS_MAX_HOLD_S:
            self._finish_leg(settings.SLS_MAX_HOLD_S, elapsed_s)
            return True
        return False

    def _finish_leg(self, seconds: float, elapsed_s: float) -> None:
        self.results[self.current_leg] = max(0.0, round(seconds, 2))
        self.set_results.append(self.results[self.current_leg])
        self._lift_started = None
        self._down_since = None
        self.phase = "leg_done"
        if self.leg_index >= len(self.leg_order) - 1:
            self.done = True
            self.finished_at_s = elapsed_s

    def start_next_leg(self) -> None:
        """Arm the second leg (called from the UI button)."""
        if self.done or self.phase != "leg_done":
            return
        self.leg_index += 1
        self.phase = "waiting"

    def stop_current_leg(self, elapsed_s: float) -> None:
        """Manually end the leg being measured (UI 'Stop this leg')."""
        if self.phase == "holding" and self._lift_started is not None:
            self._finish_leg(time.monotonic() - self._lift_started, elapsed_s)
        elif self.phase == "waiting":
            self._finish_leg(0.0, elapsed_s)

    # ---- reporting -----------------------------------------------------
    @property
    def progress(self) -> float:
        legs_done = len(self.results)
        live = 0.0
        if self.phase == "holding":
            live = min(1.0, self.hold_now_s / settings.SLS_MAX_HOLD_S)
        return min(1.0, (legs_done + live) / len(self.leg_order))

    @property
    def progress_caption(self) -> str:
        if self.done:
            parts = [f"{leg.capitalize()} {self.results.get(leg, 0):.1f}s"
                     for leg in self.leg_order]
            return "  ·  ".join(parts)
        if self.phase == "leg_done":
            return (f"{self.current_leg.capitalize()} leg: "
                    f"{self.results.get(self.current_leg, 0):.1f}s - ready for the next leg")
        if self.phase == "waiting":
            return f"{self.current_leg.capitalize()} leg - lift the other foot to start"
        return f"{self.current_leg.capitalize()} leg  ·  {self.hold_now_s:.1f}s"


EXERCISES = {cls.key: cls for cls in (SitToStand, KneeROM, SingleLegStance)}

MENU = [
    (SitToStand.key, SitToStand.title,
     "Stand up and sit down 5 times - we time how long it takes."),
    (KneeROM.key, KneeROM.title,
     "Bend and straighten the knee; we measure your range of motion."),
    (SingleLegStance.key, SingleLegStance.title,
     "Balance on one leg, then the other - we time each side."),
]


def make(key: str, sets: int = 1, quota: Optional[int] = None,
         first_leg: str = "right") -> Exercise:
    cls = EXERCISES[key]
    if cls is SingleLegStance:
        return cls(first_leg=first_leg, quota=quota)
    return cls(sets=sets, quota=quota)
