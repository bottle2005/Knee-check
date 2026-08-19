"""All pages of KneeCheck.

Every page is a CTkFrame.  Pages that show live video implement
``on_snapshot(snap)``, which the app calls once per *new* camera frame.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Optional

import customtkinter as ctk

import exercises
import metrics
import settings
import vision
from session import LiveSession

from .widgets import StarRow, StatCard, Stepper, PainMeter, VideoView, big_button


class Page(ctk.CTkFrame):
    def __init__(self, parent, app, **kwargs) -> None:
        super().__init__(parent, fg_color=settings.CLR_BG)
        self.app = app
        self.build(**kwargs)

    def build(self, **kwargs) -> None: ...
    def on_snapshot(self, snap: vision.Snapshot) -> None: ...
    def on_leave(self) -> None:
        """Called right before the page is destroyed."""


def _pain_values(s: Dict) -> tuple:
    """(average, maximum, peak level) pain score from a session summary.

    Sessions recorded before the rename stored these as ``tension_*``.
    """
    avg = s.get("pain_avg", s.get("tension_avg", "?"))
    mx = s.get("pain_max", s.get("tension_max", "?"))
    peak = str(s.get("pain_peak_level", s.get("tension_peak_level", "")))
    return avg, mx, peak


def _flexion_values(s: Dict) -> Optional[tuple]:
    """(straightest, average, deepest) knee flexion from a session summary.

    Sessions recorded before the switch to clinical flexion stored the raw
    hip-knee-ankle angle instead; those are converted (180 - angle) so old and
    new sessions can be compared directly.
    """
    def num(key: str) -> Optional[float]:
        try:
            return float(s.get(key, ""))
        except (TypeError, ValueError):
            return None

    lo, avg, hi = num("knee_flexion_min"), num("knee_flexion_avg"), num("knee_flexion_max")
    if None not in (lo, avg, hi):
        return round(lo, 1), round(avg, 1), round(hi, 1)

    old_lo, old_avg, old_hi = num("knee_angle_min"), num("knee_angle_avg"), num("knee_angle_max")
    if None not in (old_lo, old_avg, old_hi):
        # interior angle -> flexion (and min/max swap)
        return round(180 - old_hi, 1), round(180 - old_avg, 1), round(180 - old_lo, 1)
    return None


def _hero(parent, title: str, subtitle: str = "") -> None:
    ctk.CTkLabel(parent, text=title, font=(settings.FONT, 30, "bold"),
                 text_color=settings.CLR_TEXT).pack(anchor="w")
    if subtitle:
        ctk.CTkLabel(parent, text=subtitle, font=(settings.FONT, 14),
                     text_color=settings.CLR_MUTED, justify="left",
                     wraplength=330).pack(anchor="w", pady=(6, 0))


class _SplitPage(Page):
    """Left column of controls + right live-video card."""

    def split(self) -> ctk.CTkFrame:
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=22, pady=22)
        left = ctk.CTkFrame(wrap, fg_color="transparent", width=360)
        left.pack(side="left", fill="y", padx=(0, 18))
        left.pack_propagate(False)
        self.video = VideoView(wrap)
        self.video.pack(side="right", fill="both", expand=True)
        return left

    def on_snapshot(self, snap: vision.Snapshot) -> None:
        frame = snap.bgr.copy()
        vision.draw_skeleton(frame, snap.pose_raw)
        self.video.push(frame)


# ===========================================================================
class HomePage(_SplitPage):
    def build(self, **_) -> None:
        left = self.split()
        _hero(left, "KneeCheck 🦵",
              "A camera-based movement and comfort check.\nNothing is sent anywhere - "
              "everything stays on this computer.")
        pad = ctk.CTkFrame(left, fg_color="transparent")
        pad.pack(fill="x", pady=(30, 0))
        big_button(pad, "🆕  New user", lambda: self.app.goto(NewUserPage)).pack(fill="x", pady=7)
        big_button(pad, "👋  Returning user", lambda: self.app.goto(WelcomeBackPage),
                   primary=False).pack(fill="x", pady=7)
        big_button(pad, "📖  Past sessions", lambda: self.app.goto(HistoryPage),
                   primary=False).pack(fill="x", pady=7)
        big_button(pad, "🎥  Camera setup", lambda: self.app.goto(CamerasPage),
                   primary=False).pack(fill="x", pady=7)
        big_button(pad, "Close", self.app.shutdown, primary=False).pack(fill="x", pady=(30, 0))


# ===========================================================================
class NewUserPage(_SplitPage):
    def build(self, **_) -> None:
        left = self.split()
        _hero(left, "What's your name?",
              "We'll create your profile, then take a quick 5-second "
              "scan of your relaxed face.")
        self.entry = ctk.CTkEntry(left, height=48, corner_radius=12,
                                  font=(settings.FONT, 17),
                                  placeholder_text="Type your name…")
        self.entry.pack(fill="x", pady=(24, 10))
        self.entry.bind("<Return>", lambda _e: self._go())
        self.entry.focus_set()
        big_button(left, "Let's go!  →", self._go).pack(fill="x", pady=7)
        big_button(left, "Back", lambda: self.app.goto(HomePage), primary=False).pack(fill="x", pady=7)

    def _go(self) -> None:
        try:
            self.app.profile = self.app.profiles.create(self.entry.get())
        except ValueError as exc:
            messagebox.showerror("Hmm…", str(exc))
            return
        self.app.goto(BaselinePage)


# ===========================================================================
class WelcomeBackPage(_SplitPage):
    def build(self, **_) -> None:
        left = self.split()
        names = self.app.profiles.names()
        _hero(left, "Welcome back!",
              "Pick your name - your saved face scan loads automatically, "
              "so there's nothing to redo." if names else
              "No profiles saved yet - go back and choose \"New user\".")
        if names:
            self.pick = ctk.CTkComboBox(left, values=names, height=44,
                                        corner_radius=12, font=(settings.FONT, 16),
                                        dropdown_font=(settings.FONT, 15))
            self.pick.set(names[0])
            self.pick.pack(fill="x", pady=(24, 10))
            big_button(left, "That's me!  →", self._open).pack(fill="x", pady=7)
        big_button(left, "Back", lambda: self.app.goto(HomePage), primary=False).pack(fill="x", pady=7)

    def _open(self) -> None:
        try:
            self.app.profile = self.app.profiles.open(self.pick.get())
        except FileNotFoundError as exc:
            messagebox.showerror("Hmm…", str(exc))
            return
        self.app.goto(ProfilePage)


# ===========================================================================
class ProfilePage(_SplitPage):
    def build(self, **_) -> None:
        left = self.split()
        p = self.app.profile
        if p.has_baseline:
            when = p.baseline_taken.replace("T", " at ")
            _hero(left, f"Hi, {p.name}!",
                  f"Your relaxed-face scan from {when} is loaded and ready - "
                  "no need to redo it.")
            big_button(left, "Choose an exercise  →",
                       lambda: self.app.goto(ActivityPage)).pack(fill="x", pady=(26, 7))
            big_button(left, "Redo my face scan", lambda: self.app.goto(BaselinePage),
                       primary=False).pack(fill="x", pady=7)
        else:
            _hero(left, f"Hi, {p.name}!",
                  "We still need your 5-second relaxed-face scan before "
                  "the first exercise.")
            big_button(left, "Do my face scan  →",
                       lambda: self.app.goto(BaselinePage)).pack(fill="x", pady=(26, 7))
        big_button(left, "📖  My past sessions",
                   lambda: self.app.goto(HistoryPage, user=p.name, back=ProfilePage),
                   primary=False).pack(fill="x", pady=7)
        big_button(left, "Not me - switch user", lambda: self.app.goto(HomePage),
                   primary=False).pack(fill="x", pady=(30, 0))


# ===========================================================================
class BaselinePage(_SplitPage):
    """5-second neutral-face capture with live feedback + progress."""

    def build(self, **_) -> None:
        left = self.split()
        _hero(left, "Relaxed face scan",
              "1. Look at the camera\n2. Relax your face\n3. Hold still for a few seconds…")
        self.bar = ctk.CTkProgressBar(left, height=18, corner_radius=9,
                                      progress_color=settings.CLR_ACCENT,
                                      fg_color=settings.CLR_PANEL_2)
        self.bar.pack(fill="x", pady=(26, 8))
        self.bar.set(0)
        self.status = ctk.CTkLabel(left, text="Looking for your face…",
                                   font=(settings.FONT, 15, "bold"),
                                   text_color=settings.CLR_WARN)
        self.status.pack(anchor="w", pady=(0, 16))
        self.buttons = ctk.CTkFrame(left, fg_color="transparent")
        self.buttons.pack(fill="x")
        big_button(self.buttons, "Cancel", lambda: self.app.goto(ProfilePage),
                   primary=False).pack(fill="x")

        self._samples: list = []
        self._captured = 0.0
        self._last_face_t: Optional[float] = None
        self._wall_start: Optional[float] = None
        self._state = "running"          # running | saved | failed

    def on_snapshot(self, snap: vision.Snapshot) -> None:
        frame = snap.bgr.copy()
        face_ok = snap.face_lms is not None
        if self._wall_start is None:
            self._wall_start = snap.taken_at

        if self._state == "running":
            if face_ok:
                self._samples.append(metrics.face_features(snap.face_lms))
                if self._last_face_t is not None:
                    self._captured += min(snap.taken_at - self._last_face_t, 0.2)
                self._last_face_t = snap.taken_at
                self.status.configure(text="Perfect - hold still…", text_color=settings.CLR_OK)
            else:
                self._last_face_t = None
                self.status.configure(text="Face not visible - look at the camera.",
                                      text_color=settings.CLR_BAD)
            self.bar.set(min(1.0, self._captured / settings.BASELINE_SECONDS))

            if (self._captured >= settings.BASELINE_SECONDS
                    and len(self._samples) >= settings.BASELINE_MIN_SAMPLES):
                self._finish_ok()
            elif snap.taken_at - self._wall_start > settings.BASELINE_TIMEOUT_S:
                self._finish_fail()

        if snap.dual:
            # dual mode: the phone's face view IS the main picture here
            frame = snap.face_bgr.copy()
        elif face_ok:
            vision.draw_face_inset(frame, snap.face_lms)
        vision.draw_topbar(frame, "Relaxed face scan",
                           f"{max(0.0, settings.BASELINE_SECONDS - self._captured):.1f}s left")
        self.video.push(frame)

    def _finish_ok(self) -> None:
        self._state = "saved"
        try:
            self.app.profiles.store_baseline(self.app.profile, metrics.mean_features(self._samples))
        except (ValueError, OSError) as exc:
            messagebox.showerror("Save problem", str(exc))
            self.app.goto(ProfilePage)
            return
        self.status.configure(text="Baseline saved - all set! ✅",
                              text_color=settings.CLR_OK)
        self.after(900, lambda: self.app.goto(ProfilePage))

    def _finish_fail(self) -> None:
        self._state = "failed"
        self.status.configure(text="We couldn't see a face for long enough - nothing saved.",
                              text_color=settings.CLR_BAD)
        for w in self.buttons.winfo_children():
            w.destroy()
        big_button(self.buttons, "Try again", lambda: self.app.goto(BaselinePage)).pack(fill="x", pady=4)
        big_button(self.buttons, "Back", lambda: self.app.goto(ProfilePage),
                   primary=False).pack(fill="x", pady=4)


# ===========================================================================
class ActivityPage(_SplitPage):
    """Step 1: pick the activity. Options for it come on the next page."""

    def build(self, **_) -> None:
        left = self.split()
        _hero(left, "Choose an activity", "What are we doing today?")
        for key, title, blurb in exercises.MENU:
            card = ctk.CTkFrame(left, corner_radius=14, fg_color=settings.CLR_PANEL)
            card.pack(fill="x", pady=6)
            btn = ctk.CTkButton(
                card, text=title, height=44, corner_radius=10, anchor="w",
                fg_color="transparent", hover_color=settings.CLR_PANEL_2,
                text_color=settings.CLR_ACCENT, font=(settings.FONT, 17, "bold"),
                command=lambda k=key: self._pick(k),
            )
            btn.pack(fill="x", padx=6, pady=(6, 0))
            ctk.CTkLabel(card, text=blurb, text_color=settings.CLR_MUTED,
                         font=(settings.FONT, 12), wraplength=300,
                         justify="left").pack(anchor="w", padx=18, pady=(0, 10))
        big_button(left, "Back", lambda: self.app.goto(ProfilePage),
                   primary=False).pack(fill="x", pady=(14, 0))

    def _pick(self, key: str) -> None:
        self.app.exercise_key = key
        self.app.goto(OptionsPage)


# ===========================================================================
class OptionsPage(_SplitPage):
    """Step 2: activity-specific options - legs, sets, reps, recording."""

    def build(self, **_) -> None:
        left = self.split()
        key = self.app.exercise_key
        ex_cls = exercises.EXERCISES[key]
        _hero(left, ex_cls.title)

        def label(text: str, top: int = 12) -> None:
            ctk.CTkLabel(left, text=text, text_color=settings.CLR_MUTED,
                         font=(settings.FONT, 11, "bold")).pack(anchor="w", pady=(top, 4))

        seg_style = dict(
            height=40, corner_radius=10, font=(settings.FONT, 15, "bold"),
            selected_color=settings.CLR_ACCENT, selected_hover_color=settings.CLR_ACCENT_DARK,
            unselected_color=settings.CLR_PANEL_2, text_color=settings.CLR_TEXT,
        )

        self.variant_var = None
        self.side_seg = None
        self.leg_seg = None

        if key == "single_leg_stance":
            label("WHICH LEG FIRST?")
            self.leg_seg = ctk.CTkSegmentedButton(left, values=["Left", "Right"], **seg_style)
            self.leg_seg.set(self.app.leg.capitalize())
            self.leg_seg.pack(fill="x")
            ctk.CTkLabel(
                left,
                text=(f"Both legs are measured, one after the other (up to "
                      f"{settings.SLS_MAX_HOLD_S:.0f}s each).\nBalance on the chosen "
                      "leg; timing stops when the other foot touches down.\n"
                      "Then press \"Start next leg\" for the other side."),
                text_color=settings.CLR_MUTED, font=(settings.FONT, 12),
                wraplength=330, justify="left",
            ).pack(anchor="w", pady=(8, 0))
        elif key == "sit_to_stand":
            label("HOW WILL YOU DO IT?")
            self.variant_var = tk.StringVar(value=self.app.sts_variant)
            for value, text in (("both", "Both legs (normal)"),
                                ("left", "Left leg only"),
                                ("right", "Right leg only")):
                ctk.CTkRadioButton(
                    left, text=text, variable=self.variant_var, value=value,
                    font=(settings.FONT, 14), fg_color=settings.CLR_ACCENT,
                    text_color=settings.CLR_TEXT, command=self._variant_changed,
                ).pack(anchor="w", pady=3)
            label("MEASURE FROM WHICH SIDE?")
            self.side_seg = ctk.CTkSegmentedButton(left, values=["Left", "Right"], **seg_style)
            self.side_seg.set(self.app.leg.capitalize())
            self.side_seg.pack(fill="x")
            self._variant_changed()
        else:
            label("WHICH LEG?" if key == "knee_rom" else "STAND ON WHICH LEG?")
            self.leg_seg = ctk.CTkSegmentedButton(left, values=["Left", "Right"], **seg_style)
            self.leg_seg.set(self.app.leg.capitalize())
            self.leg_seg.pack(fill="x")

        self.sets_pick = None
        self.quota_pick = None

        if key == "single_leg_stance":
            pass                       # one attempt per leg; nothing to configure
        elif key == "sit_to_stand":
            label(f"HOW MANY SETS OF {settings.STS_REPS_PER_SET}?", top=14)
            self.sets_pick = Stepper(left, value=self.app.sets, minimum=1,
                                     maximum=settings.MAX_SETS)
            self.sets_pick.pack(fill="x")
            ctk.CTkLabel(
                left,
                text=(f"Each set is {settings.STS_REPS_PER_SET} repetitions and is "
                      "timed - the score is how many seconds the set takes."),
                text_color=settings.CLR_MUTED, font=(settings.FONT, 12),
                wraplength=330, justify="left",
            ).pack(anchor="w", pady=(6, 0))
        else:
            label("SETS", top=14)
            self.sets_pick = Stepper(left, value=self.app.sets, minimum=1,
                                     maximum=settings.MAX_SETS)
            self.sets_pick.pack(fill="x")
            label("REPS PER SET")
            self.quota_pick = Stepper(left, value=self.app.quota or ex_cls.default_quota,
                                      minimum=1, maximum=settings.MAX_REPS_PER_SET)
            self.quota_pick.pack(fill="x")

        self.record = ctk.CTkSwitch(
            left, text="Also save a video (normally OFF)",
            font=(settings.FONT, 13), progress_color=settings.CLR_WARN,
            text_color=settings.CLR_TEXT,
        )
        self.record.pack(anchor="w", pady=(14, 2))
        ctk.CTkLabel(left, text="Privacy: only dot-positions and scores are saved "
                                "unless you switch this on.",
                     text_color=settings.CLR_MUTED, font=(settings.FONT, 11),
                     wraplength=330, justify="left").pack(anchor="w")

        big_button(left, "Start!  🚀", self._start).pack(fill="x", pady=(16, 5))
        big_button(left, "Back", lambda: self.app.goto(ActivityPage), primary=False).pack(fill="x")

    def _variant_changed(self) -> None:
        """Single-leg sit-to-stand fixes the measured side to that leg."""
        variant = self.variant_var.get()
        if variant in ("left", "right"):
            self.side_seg.set(variant.capitalize())
            self.side_seg.configure(state="disabled")
        else:
            self.side_seg.configure(state="normal")

    def _start(self) -> None:
        key = self.app.exercise_key
        if key == "sit_to_stand":
            self.app.sts_variant = self.variant_var.get()
            if self.app.sts_variant in ("left", "right"):
                self.app.leg = self.app.sts_variant
            else:
                self.app.leg = self.side_seg.get().lower()
        else:
            self.app.sts_variant = ""
            self.app.leg = self.leg_seg.get().lower()
        self.app.sets = self.sets_pick.get() if self.sets_pick else 1
        self.app.quota = self.quota_pick.get() if self.quota_pick else None
        self.app.record_video = bool(self.record.get())
        self.app.goto(LivePage)


# ===========================================================================
class LivePage(_SplitPage):
    """The live assessment."""

    def build(self, **_) -> None:
        left = self.split()
        self._closing = False
        self._done_timer_started = False
        variant = self.app.sts_variant if self.app.exercise_key == "sit_to_stand" else ""
        self.session = LiveSession(
            self.app.profile, self.app.leg, self.app.exercise_key, self.app.record_video,
            sets=self.app.sets, quota=self.app.quota, variant=variant,
        )

        ex = self.session.exercise
        if ex.two_legged:
            leg_text = f"{ex.leg_order[0]} leg first, then {ex.leg_order[1]}"
            plan_text = f"up to {settings.SLS_MAX_HOLD_S:.0f}s per leg"
        elif variant == "both":
            leg_text = f"both legs, measuring {self.app.leg}"
            plan_text = f"{ex.sets_planned} set(s) of {ex.quota} reps (timed)"
        else:
            leg_text = f"{self.app.leg} leg"
            plan_text = (f"{ex.sets_planned} set(s) of {ex.quota} reps"
                         + (" (timed)" if getattr(ex, "timed", False) else ""))
        _hero(left, ex.title, f"{self.app.profile.name} - {leg_text}\n{plan_text}")

        self.card_angle = StatCard(left, "Knee flexion (0° = straight)")
        self.card_angle.pack(fill="x", pady=(16, 5))
        self.card_progress = StatCard(
            left, "Reps" if ex.uses_reps else "Balance time")
        self.card_progress.pack(fill="x", pady=5)

        # timed 5xSTS gets a live set stopwatch
        self.card_timer = None
        if getattr(ex, "timed", False):
            self.card_timer = StatCard(left, "Set stopwatch")
            self.card_timer.pack(fill="x", pady=5)

        if ex.uses_reps:
            self.stars = StarRow(left, ex.quota)
            self.stars.pack(anchor="w", pady=(2, 5))
        else:
            self.stars = None
            self.hold_bar = ctk.CTkProgressBar(left, height=16, corner_radius=8,
                                               progress_color=settings.CLR_STAR,
                                               fg_color=settings.CLR_PANEL_2)
            self.hold_bar.pack(fill="x", pady=(2, 5))
            self.hold_bar.set(0)

        self.meter = PainMeter(left)
        self.meter.pack(fill="x", pady=5)

        self.hint = ctk.CTkLabel(left, text="", font=(settings.FONT, 13, "bold"),
                                 text_color=settings.CLR_BAD, wraplength=330,
                                 justify="left")
        self.hint.pack(anchor="w", pady=(6, 8))

        # Two-legged tests: gate between legs, plus a manual stop for the leg.
        self.next_leg_btn = None
        self.stop_leg_btn = None
        if ex.two_legged:
            self.next_leg_btn = big_button(left, "▶  Start next leg", self._next_leg)
            self.stop_leg_btn = big_button(left, "⏹  Stop this leg", self._stop_leg,
                                           primary=False)
            self.stop_leg_btn.pack(fill="x", pady=4)

        big_button(left, "✅  I'm finished", self._finish).pack(fill="x", pady=4)
        big_button(left, "Stop - don't keep this", self._abort, danger=True).pack(fill="x", pady=4)

    def _next_leg(self) -> None:
        self.session.start_next_leg()

    def _stop_leg(self) -> None:
        self.session.stop_current_leg()

    def on_snapshot(self, snap: vision.Snapshot) -> None:
        if self._closing:
            return
        frame, state = self.session.ingest(snap)
        self.video.push(frame)

        angle = state["angle"]
        self.card_angle.set(f"{angle:.0f}°" if angle is not None else "--",
                            settings.CLR_TEXT if state["leg_ok"] else settings.CLR_MUTED)

        if self.stars is not None:
            # rep-based: big "3 / 10", set number underneath
            self.card_progress.set(
                f"{state.get('reps_in_set', 0)} / {state.get('quota', 0)}",
                sub=(f"Set {state.get('set_index', 0) + 1} of "
                     f"{state.get('sets_planned', 1)}"
                     if state.get("sets_planned", 1) > 1 else ""),
            )
            self.stars.set(state.get("reps_in_set") or 0)
        else:
            # balance test: big stopwatch, short status underneath
            hold = state.get("hold") or 0.0
            leg = str(state.get("measured_leg", "")).capitalize()
            phase = state.get("leg_phase", "")
            if state.get("done"):
                status = "Both legs measured"
                color = settings.CLR_OK
            elif state.get("awaiting_next_leg"):
                status = f"{leg} leg finished - press \"Start next leg\""
                color = settings.CLR_ACCENT
            elif phase == "waiting":
                status = f"{leg} leg - lift the other foot to start"
                color = settings.CLR_MUTED
            else:
                status = f"Balancing on the {leg.lower()} leg"
                color = settings.CLR_TEXT
            done_legs = state.get("leg_results", {})
            if done_legs and not state.get("done"):
                status += "   (" + ", ".join(
                    f"{k[0].upper()} {v:.1f}s" for k, v in done_legs.items()) + ")"
            self.card_progress.set(f"{hold:.1f} s", color, sub=status)
            quota = max(1, state.get("quota", 1))
            self.hold_bar.set(min(1.0, hold / quota))
        self.meter.set(state["pain"], state["pain_level"])
        if self.card_timer is not None:
            self.card_timer.set(f"{state.get('set_elapsed', 0.0):.1f} s")

        # show/hide the between-legs gate
        if self.next_leg_btn is not None:
            waiting = state.get("awaiting_next_leg") and not state.get("done")
            if waiting and not self.next_leg_btn.winfo_ismapped():
                self.next_leg_btn.pack(fill="x", pady=4, before=self.stop_leg_btn)
            elif not waiting and self.next_leg_btn.winfo_ismapped():
                self.next_leg_btn.pack_forget()
            if self.stop_leg_btn is not None:
                self.stop_leg_btn.configure(state="disabled" if (waiting or state.get("done"))
                                            else "normal")

        if state.get("resting"):
            self.hint.configure(
                text=f"😮‍💨 Rest! Next set in {state['rest_remaining']:.0f}s",
                text_color=settings.CLR_ACCENT)
        elif state.get("awaiting_next_leg"):
            res = state.get("leg_results", {})
            done_txt = "   ".join(f"{k.capitalize()}: {v:.1f}s" for k, v in res.items())
            self.hint.configure(text=f"✅ {done_txt}", text_color=settings.CLR_ACCENT)
        else:
            self.hint.configure(text=state["warning"], text_color=settings.CLR_BAD)

        if state["done"] and not self._done_timer_started:
            self._done_timer_started = True
            self.after(1600, self._finish)          # let the cheer show briefly

    def _finish(self) -> None:
        if self._closing:
            return
        self._closing = True
        summary = self.session.finish(aborted=False)
        self.app.goto(ResultsPage, summary=summary)

    def _abort(self) -> None:
        if self._closing:
            return
        if not messagebox.askyesno("Stop?", "Stop now and throw away this attempt?"):
            return
        self._closing = True
        self.session.finish(aborted=True)
        self.app.goto(ActivityPage)

    def on_leave(self) -> None:
        self._closing = True
        self.session.finish(aborted=True)           # no-op if already finished


# ===========================================================================
class ResultsPage(Page):
    def build(self, summary: Dict = None, **_) -> None:
        s = summary or {}
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=26, pady=26)

        left = ctk.CTkFrame(wrap, fg_color="transparent", width=340)
        left.pack(side="left", fill="y", padx=(0, 20))
        left.pack_propagate(False)
        _hero(left, "Session complete 🎉" if s.get("target_reached") else "Session ended 💪",
              "Here's how it went.")
        big_button(left, "Another exercise  →",
                   lambda: self.app.goto(ActivityPage)).pack(fill="x", pady=(24, 6))
        big_button(left, "🏠  Main page", lambda: self.app.goto(HomePage),
                   primary=False).pack(fill="x", pady=6)
        big_button(left, "Open results folder", self._open_folder,
                   primary=False).pack(fill="x", pady=6)
        big_button(left, "Close", self.app.shutdown, primary=False).pack(fill="x", pady=(28, 0))

        card = ctk.CTkFrame(wrap, corner_radius=18, fg_color=settings.CLR_PANEL)
        card.pack(side="right", fill="both", expand=True)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=30, pady=26)

        colors = {"Low": settings.CLR_OK, "Moderate": settings.CLR_WARN,
                  "High": settings.CLR_BAD}
        is_two_legged = s.get("hold_right_s", "") != "" or s.get("hold_left_s", "") != ""
        heading = str(s.get("exercise_title", ""))
        if not is_two_legged:
            heading += f" — {str(s.get('measured_leg', '')).capitalize()} leg"
        ctk.CTkLabel(inner, text=heading,
                     font=(settings.FONT, 22, "bold"),
                     text_color=settings.CLR_TEXT).pack(anchor="w")
        diff = str(s.get("difficulty_rating", "Low"))
        ctk.CTkLabel(inner, text=f"Difficulty rating: {diff}",
                     font=(settings.FONT, 16, "bold"),
                     text_color=colors.get(diff, settings.CLR_TEXT)).pack(anchor="w", pady=(2, 18))

        def row(label: str, value: str) -> None:
            r = ctk.CTkFrame(inner, fg_color="transparent")
            r.pack(fill="x", pady=4)
            ctk.CTkLabel(r, text=label, width=250, anchor="w",
                         font=(settings.FONT, 13), text_color=settings.CLR_MUTED).pack(side="left")
            ctk.CTkLabel(r, text=value, anchor="w", font=(settings.FONT, 14, "bold"),
                         text_color=settings.CLR_TEXT, wraplength=460,
                         justify="left").pack(side="left", fill="x")

        def headline(text: str, color: str = settings.CLR_STAR) -> None:
            ctk.CTkLabel(inner, text=text, font=(settings.FONT, 18, "bold"),
                         text_color=color, justify="left").pack(anchor="w", pady=(0, 6))

        sets_planned = int(s.get("sets_planned") or 1)
        quota = s.get("quota_per_set") or ""

        # ---- Single Leg Stance: per-leg times -----------------------------
        if is_two_legged:
            for side in ("right", "left"):
                val = s.get(f"hold_{side}_s", "")
                headline(f"{side.capitalize()}:  {val} s" if val != ""
                         else f"{side.capitalize()}:  not measured")
            if s.get("hold_difference_s", "") != "":
                row("Difference between legs", f"{s.get('hold_difference_s')} s")
            row("Order", str(s.get("leg_order", "")).replace("→", "then"))

        # ---- timed 5x sit-to-stand ----------------------------------------
        if s.get("sts_avg_time_s", "") != "":
            reps_per_set = s.get("reps_per_set", settings.STS_REPS_PER_SET)
            headline(f"Total time for {reps_per_set} repetitions:  "
                     f"{s.get('sts_avg_time_s')} s"
                     + (" (average per set)" if sets_planned > 1 else ""))
            if s.get("sts_avg_rep_time_s", "") != "":
                headline(f"Average per repetition:  {s.get('sts_avg_rep_time_s')} s",
                         settings.CLR_ACCENT)
            if sets_planned > 1:
                row("Each set", f"{s.get('sts_set_times_s', '')}  seconds")
                row("Fastest set", f"{s.get('sts_best_time_s', '')} s")
            row("Sets completed", f"{s.get('sets_completed', 0)} of {sets_planned}"
                                  f"  ({reps_per_set} reps each)")
        elif sets_planned > 1 and not is_two_legged:
            row("Sets", f"{s.get('sets_completed', 0)} of {sets_planned} completed "
                        f"(target {quota} reps each)")
            row("Average per set", f"{s.get('average_per_set', '')} reps")

        if s.get("reps") not in ("", None) and s.get("sts_avg_time_s", "") == "":
            target = sets_planned * int(quota or settings.REP_TARGET)
            row("Reps completed (total)", f"{s.get('reps')} / {target}")

        # ---- range of motion (Knee ROM sessions only) ----------------------
        if s.get("exercise") == "knee_rom":
            rom_r, rom_l = s.get("rom_right_deg", ""), s.get("rom_left_deg", "")
            if rom_r != "" and rom_l != "":
                headline(f"Right knee ROM:  {rom_r}°", settings.CLR_ACCENT)
                headline(f"Left knee ROM:  {rom_l}°", settings.CLR_ACCENT)
                if s.get("rom_difference_deg", "") != "":
                    row("Difference", f"{s.get('rom_difference_deg')}°")
            elif s.get("rom_deg", "") != "":
                headline(f"Range of motion:  {s.get('rom_deg')}°", settings.CLR_ACCENT)
            flex = _flexion_values(s)
            if flex:
                row("Knee flexion  (straightest / avg / deepest bend)",
                    f"{flex[0]}° / {flex[1]}° / {flex[2]}°")
        avg, mx, peak = _pain_values(s)
        row("Pain level  (avg / max)", f"{avg} / {mx}")
        row("Peak pain level", peak)
        row("Video saved", "Yes" if s.get("video_saved") else "No (privacy default)")
        row("Saved to", str(s.get("saved_to", "")))

        self._folder = str(s.get("saved_to", ""))

    def _open_folder(self) -> None:
        try:
            os.startfile(self._folder)
        except OSError as exc:
            messagebox.showerror("Open folder", str(exc))


# ===========================================================================
class HistoryPage(Page):
    """Browsable log of all past sessions, read from summary.csv files."""

    def build(self, user: Optional[str] = None, back=None, **_) -> None:
        import records

        self._back = back or HomePage
        self._user_filter = user

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=26, pady=(24, 8))
        ctk.CTkButton(head, text="←  Back", width=100, height=38, corner_radius=10,
                      fg_color=settings.CLR_PANEL_2, hover_color="#28374f",
                      text_color=settings.CLR_TEXT, font=(settings.FONT, 14, "bold"),
                      command=lambda: self.app.goto(self._back)).pack(side="left", padx=(0, 14))
        ctk.CTkLabel(head, text="Past sessions 📖", font=(settings.FONT, 28, "bold"),
                     text_color=settings.CLR_TEXT).pack(side="left")

        names = ["All users"] + self.app.profiles.names()
        self._filter_box = ctk.CTkComboBox(
            head, values=names, width=170, height=38, corner_radius=10,
            font=(settings.FONT, 14), dropdown_font=(settings.FONT, 13),
            command=lambda _v: self._refill(),
        )
        self._filter_box.set(user if user in names else "All users")
        self._filter_box.pack(side="right")

        activities = ["All exercises"] + [title for _k, title, _b in exercises.MENU]
        self._activity_box = ctk.CTkComboBox(
            head, values=activities, width=190, height=38, corner_radius=10,
            font=(settings.FONT, 14), dropdown_font=(settings.FONT, 13),
            command=lambda _v: self._refill(),
        )
        self._activity_box.set("All exercises")
        self._activity_box.pack(side="right", padx=(0, 8))

        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(fill="both", expand=True, padx=26, pady=(0, 20))

        self._refill()

    def _refill(self) -> None:
        import records

        for w in self._list.winfo_children():
            w.destroy()
        pick = self._filter_box.get()
        sessions = records.list_past_sessions(None if pick == "All users" else pick)

        activity = self._activity_box.get()
        if activity != "All exercises":
            wanted = {k for k, title, _b in exercises.MENU if title == activity}
            sessions = [s for s in sessions if s.get("exercise") in wanted]

        if not sessions:
            ctk.CTkLabel(self._list, text="No sessions match these filters yet.",
                         font=(settings.FONT, 16), text_color=settings.CLR_MUTED
                         ).pack(pady=40)
            return

        for i, s in enumerate(sessions):
            self._make_card(s, self._previous_attempt(sessions, i))

    # -------------------------------------------------- card helpers
    @staticmethod
    def _previous_attempt(sessions, index: int) -> Optional[Dict]:
        """The next-older session by the same user doing the same exercise."""
        me = sessions[index]
        for older in sessions[index + 1:]:
            if (older.get("user") == me.get("user")
                    and older.get("exercise") == me.get("exercise")):
                return older
        return None

    @staticmethod
    def _friendly_when(when) -> str:
        from datetime import date
        d = when.date()
        if d == date.today():
            day = "Today"
        elif (date.today() - d).days == 1:
            day = "Yesterday"
        else:
            day = when.strftime("%d %b %Y")
        return f"{day}, {when.strftime('%H:%M')}"

    @staticmethod
    def _comfort_line(s: Dict) -> tuple:
        """Difficulty → (emoji sentence, color) in plain language."""
        diff = str(s.get("difficulty_rating", ""))
        return {
            "Low": ("😊  Looked comfortable", settings.CLR_OK),
            "Moderate": ("😐  Looked a little tough", settings.CLR_WARN),
            "High": ("😣  Looked like hard work", settings.CLR_BAD),
        }.get(diff, ("", settings.CLR_MUTED))

    @staticmethod
    def _progress_note(s: Dict, prev: Optional[Dict]) -> str:
        """Plain-language comparison with the previous try of this exercise."""
        if prev is None:
            return ""
        try:
            if s.get("reps") not in ("", None) and prev.get("reps") not in ("", None):
                delta = int(s["reps"]) - int(prev["reps"])
                unit = "rep" if abs(delta) == 1 else "reps"
                if delta > 0:
                    return f"⬆  {delta} more {unit} than last time - improving!"
                if delta == 0:
                    return "➡  Same as last time - steady"
                return f"⬇  {abs(delta)} {unit} fewer than last time"
            if s.get("best_hold_s") not in ("", None) and prev.get("best_hold_s") not in ("", None):
                delta = float(s["best_hold_s"]) - float(prev["best_hold_s"])
                if delta > 0.05:
                    return f"⬆  Held {delta:.1f}s longer than last time - improving!"
                if delta > -0.05:
                    return "➡  Same as last time - steady"
                return f"⬇  Held {abs(delta):.1f}s less than last time"
        except (TypeError, ValueError):
            pass
        return ""

    def _make_card(self, s: Dict, prev: Optional[Dict]) -> None:
        card = ctk.CTkFrame(self._list, corner_radius=14, fg_color=settings.CLR_PANEL)
        card.pack(fill="x", pady=5)

        # line 1: who + what + when
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(10, 0))
        title = (f"{s.get('user', '?')}  ·  "
                 f"{s.get('exercise_title', s.get('exercise', '?'))}  ·  "
                 f"{str(s.get('measured_leg', '')).capitalize()} leg")
        ctk.CTkLabel(top, text=title, font=(settings.FONT, 15, "bold"),
                     text_color=settings.CLR_TEXT).pack(side="left")
        ctk.CTkLabel(top, text=self._friendly_when(s["_when"]),
                     font=(settings.FONT, 13),
                     text_color=settings.CLR_MUTED).pack(side="right")

        # line 2: the big at-a-glance result
        try:
            sets_planned = int(s.get("sets_planned") or 1)
        except ValueError:
            sets_planned = 1
        try:
            quota = int(float(s.get("quota_per_set") or settings.REP_TARGET))
        except ValueError:
            quota = settings.REP_TARGET
        result_color = settings.CLR_STAR
        if s.get("hold_right_s", "") != "" or s.get("hold_left_s", "") != "":
            r = s.get("hold_right_s", "") or "-"
            l = s.get("hold_left_s", "") or "-"
            result_text = f"⏱  Right: {r} s      Left: {l} s"
        elif s.get("sts_avg_time_s", "") != "":
            reps_each = s.get("reps_per_set", settings.STS_REPS_PER_SET)
            result_text = f"⏱  {s['sts_avg_time_s']} s for {reps_each} reps"
            if s.get("sts_avg_rep_time_s", "") != "":
                result_text += f"   ·   {s['sts_avg_rep_time_s']} s per rep"
            if sets_planned > 1:
                result_text += f"   ·   {s.get('sets_completed', 0)} of {sets_planned} sets"
        elif s.get("reps") not in ("", None):
            try:
                reps = int(s["reps"])
            except ValueError:
                reps = 0
            rom = s.get("rom_deg", "")
            if rom != "":
                result_text = f"📐  ROM {rom}°   ·   {reps} reps"
            elif sets_planned > 1:
                result_text = (f"⭐ {reps} reps total  ·  "
                               f"{s.get('sets_completed', 0)} of {sets_planned} sets  ·  "
                               f"avg {s.get('average_per_set', '?')} per set")
            else:
                stars = "★" * min(reps, quota) + "☆" * max(0, quota - reps)
                result_text = f"{stars}   {reps} of {quota}"
        elif s.get("best_hold_s") not in ("", None):
            result_text = f"⏱  Held one-leg balance for {s['best_hold_s']} seconds"
        else:
            result_text = "No result recorded"
            result_color = settings.CLR_MUTED
        if s.get("aborted") == "True":
            result_text += "   (stopped early)"
        ctk.CTkLabel(card, text=result_text, font=(settings.FONT, 18, "bold"),
                     text_color=result_color).pack(anchor="w", padx=16, pady=(4, 0))

        # line 3: comfort in plain words (+ progress vs last time)
        comfort, comfort_color = self._comfort_line(s)
        if comfort:
            ctk.CTkLabel(card, text=comfort, font=(settings.FONT, 14),
                         text_color=comfort_color).pack(anchor="w", padx=16, pady=(2, 0))
        note = self._progress_note(s, prev)
        if note:
            ctk.CTkLabel(card, text=note, font=(settings.FONT, 13, "bold"),
                         text_color=settings.CLR_ACCENT).pack(anchor="w", padx=16, pady=(2, 0))

        # line 4: small technical detail for clinicians + Open files
        bottom = ctk.CTkFrame(card, fg_color="transparent")
        bottom.pack(fill="x", padx=16, pady=(4, 10))
        avg, mx, _peak = _pain_values(s)
        tech = f"pain {avg} avg / {mx} max"
        if s.get("exercise") == "knee_rom":
            flex = _flexion_values(s)
            if flex:
                tech += f"   ·   flexion {flex[0]}°–{flex[2]}°"
        ctk.CTkLabel(bottom, text=tech, font=(settings.FONT, 11),
                     text_color=settings.CLR_MUTED).pack(side="left")
        ctk.CTkButton(bottom, text="🗑", width=34, height=28,
                      corner_radius=8, fg_color="transparent",
                      hover_color="#3b1c26", text_color=settings.CLR_BAD,
                      border_width=1, border_color=settings.CLR_BAD,
                      font=(settings.FONT, 13),
                      command=lambda: self._delete(s)).pack(side="right", padx=(8, 0))
        ctk.CTkButton(bottom, text="Open files", width=90, height=28,
                      corner_radius=8, fg_color=settings.CLR_PANEL_2,
                      hover_color="#28374f", text_color=settings.CLR_TEXT,
                      font=(settings.FONT, 12),
                      command=lambda p=s["_folder"]: self._open(p)).pack(side="right")

    def _delete(self, s: Dict) -> None:
        import records

        what = (f"{s.get('user', '?')} - {s.get('exercise_title', '?')} "
                f"({self._friendly_when(s['_when'])})")
        if not messagebox.askyesno(
                "Delete session",
                f"Permanently delete this session and all its files?\n\n{what}\n\n"
                "This cannot be undone."):
            return
        try:
            records.delete_session(s["_folder"])
        except (ValueError, OSError) as exc:
            messagebox.showerror("Delete session", str(exc))
            return
        self._refill()

    @staticmethod
    def _open(path: str) -> None:
        try:
            os.startfile(path)
        except OSError as exc:
            messagebox.showerror("Open folder", str(exc))


# ===========================================================================
class CamerasPage(_SplitPage):
    """Choose the body camera and an optional face camera (e.g. a phone).

    Phones become cameras via DroidCam / Iriun (they appear as a normal
    camera device) or the Android "IP Webcam" app (a local-WiFi stream URL).
    """

    def build(self, **_) -> None:
        import threading

        left = self.split()
        _hero(left, "Camera setup 🎥",
              "Use a second camera (like a phone on a stand near the face) so "
              "the face scan works even when standing far from the laptop.")

        cfg = settings.load_camera_config()
        cur_face = cfg.get("face")

        ctk.CTkLabel(left, text="BODY CAMERA", text_color=settings.CLR_MUTED,
                     font=(settings.FONT, 11, "bold")).pack(anchor="w", pady=(16, 4))
        self.body_box = ctk.CTkComboBox(left, values=["scanning…"], height=40,
                                        corner_radius=10, font=(settings.FONT, 14))
        self.body_box.pack(fill="x")

        ctk.CTkLabel(left, text="FACE CAMERA", text_color=settings.CLR_MUTED,
                     font=(settings.FONT, 11, "bold")).pack(anchor="w", pady=(14, 4))
        self.face_box = ctk.CTkComboBox(left, values=["scanning…"], height=40,
                                        corner_radius=10, font=(settings.FONT, 14))
        self.face_box.pack(fill="x")

        self.url_entry = ctk.CTkEntry(
            left, height=38, corner_radius=10, font=(settings.FONT, 13),
            placeholder_text="http://192.168.x.x:8080/video",
        )
        self.url_entry.pack(fill="x", pady=(8, 0))
        if isinstance(cur_face, str):
            self.url_entry.insert(0, cur_face)
        ctk.CTkLabel(
            left,
            text=f"The phone view is rotated {settings.FACE_ROTATION}° automatically "
                 "so the face is upright (this also keeps the pain score accurate).",
            font=(settings.FONT, 11), text_color=settings.CLR_MUTED,
            wraplength=330, justify="left",
        ).pack(anchor="w", pady=(10, 0))

        ctk.CTkLabel(
            left,
            text="Phone options (no internet needed):\n"
                 "• DroidCam or Iriun app + their Windows client → the phone "
                 "shows up as a normal camera in the list above.\n"
                 "• Android \"IP Webcam\" app → start its server and paste the "
                 "URL here, then pick \"Phone stream (URL)\".",
            font=(settings.FONT, 11), text_color=settings.CLR_MUTED,
            wraplength=330, justify="left",
        ).pack(anchor="w", pady=(10, 10))

        self.status = ctk.CTkLabel(
            left, font=(settings.FONT, 12, "bold"), wraplength=330, justify="left",
            text=("Currently: two cameras (face camera active) ✅"
                  if getattr(self.app.camera, "dual", False)
                  else "Currently: one camera for everything"),
            text_color=(settings.CLR_OK if getattr(self.app.camera, "dual", False)
                        else settings.CLR_MUTED),
        )
        self.status.pack(anchor="w", pady=(0, 8))

        big_button(left, "Save & restart cameras", self._apply).pack(fill="x", pady=4)
        big_button(left, "Back", lambda: self.app.goto(HomePage), primary=False).pack(fill="x", pady=4)

        # probe devices off the UI thread (it takes a few seconds)
        body_now = cfg.get("body", 0)
        threading.Thread(
            target=lambda: self._scanned(vision.probe_cameras(
                skip=body_now if isinstance(body_now, int) else None)),
            daemon=True,
        ).start()

    def _scanned(self, found) -> None:
        def fill() -> None:
            if not self.winfo_exists():
                return
            cfg = settings.load_camera_config()
            body_vals = [f"Camera {i}" for i in found] or ["Camera 0"]
            self.body_box.configure(values=body_vals)
            body_now = cfg.get("body", 0)
            want = f"Camera {body_now}"
            self.body_box.set(want if want in body_vals else body_vals[0])

            face_vals = ["Same camera (single)"] + [f"Camera {i}" for i in found] \
                        + ["Phone stream (URL)"]
            self.face_box.configure(values=face_vals)
            cur = cfg.get("face")
            if isinstance(cur, int):
                self.face_box.set(f"Camera {cur}")
            elif isinstance(cur, str):
                self.face_box.set("Phone stream (URL)")
            else:
                self.face_box.set("Same camera (single)")
        self.after(0, fill)

    def _apply(self) -> None:
        body_pick = self.body_box.get()
        if not body_pick.startswith("Camera "):
            messagebox.showerror("Camera setup", "Still scanning - try again in a moment.")
            return
        body = int(body_pick.split()[-1])

        face_pick = self.face_box.get()
        face: object
        if face_pick == "Same camera (single)":
            face = None
        elif face_pick == "Phone stream (URL)":
            url = self.url_entry.get().strip()
            if not url.lower().startswith(("http://", "https://", "rtsp://")):
                messagebox.showerror("Camera setup",
                                     "Paste the phone's stream URL first, e.g.\n"
                                     "http://192.168.1.23:8080/video")
                return
            face = url
        else:
            face = int(face_pick.split()[-1])
            if face == body:
                messagebox.showerror("Camera setup",
                                     "Face and body can't use the same device - "
                                     "choose \"Same camera (single)\" for that.")
                return

        settings.save_camera_config(body, face)
        self.app.reboot_camera()


# ===========================================================================
class CameraFailPage(Page):
    def build(self, message: str = "", **_) -> None:
        box = ctk.CTkFrame(self, fg_color="transparent")
        box.pack(expand=True)
        _hero(box, "No camera found", message or "The webcam could not be opened.")
        big_button(box, "Try again", self.app.retry_camera).pack(fill="x", pady=(24, 6))
        big_button(box, "Close", self.app.shutdown, primary=False).pack(fill="x", pady=6)
