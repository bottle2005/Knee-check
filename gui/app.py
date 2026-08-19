"""Main window: navigation + the single UI timer that feeds pages frames."""
from __future__ import annotations

from typing import Optional, Type

import customtkinter as ctk

import settings
from records import Profile, ProfileBook
from vision import CameraUnavailable, CameraWorker

from . import pages


class KneeCheckApp(ctk.CTk):
    """Top-level window and app state."""

    def __init__(self) -> None:
        super().__init__(fg_color=settings.CLR_BG)
        ctk.set_appearance_mode("dark")
        self.title(settings.APP_NAME)
        self.geometry("1280x820")
        self.minsize(1024, 700)

        self.profiles = ProfileBook()
        self.profile: Optional[Profile] = None
        self.leg = "left"
        self.exercise_key = "sit_to_stand"
        self.sts_variant = "both"        # STS: "both" / "left" / "right"
        self.sets = 1
        self.quota: Optional[int] = None  # reps or seconds per set; None = default
        self.record_video = False

        self.camera: Optional[CameraWorker] = None
        self._page: Optional[pages.Page] = None
        self._last_frame_id = 0
        self._alive = True

        # Show the window immediately with a splash; the camera (which can
        # take several seconds to open) boots right after the first paint.
        self._splash = ctk.CTkLabel(self, text="🎥  Warming up the camera…",
                                    font=(settings.FONT, 22, "bold"),
                                    text_color=settings.CLR_MUTED)
        self._splash.pack(expand=True)
        self.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.after(150, self._boot_camera)
        self.after(settings.UI_TICK_MS, self._tick)

    # ------------------------------------------------------------------
    def _boot_camera(self) -> None:
        self.update_idletasks()          # make sure the splash is painted
        try:
            self.camera = CameraWorker()
            target = pages.HomePage
            kwargs = {}
        except CameraUnavailable as exc:
            self.camera = None
            target, kwargs = pages.CameraFailPage, {"message": str(exc)}
        if self._splash is not None:
            self._splash.destroy()
            self._splash = None
        self.goto(target, **kwargs)

    def retry_camera(self) -> None:
        self._boot_camera()

    def reboot_camera(self) -> None:
        """Close and reopen the cameras (after a camera-config change)."""
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self._page is not None:
            self._page.on_leave()
            self._page.destroy()
            self._page = None
        self._splash = ctk.CTkLabel(self, text="🎥  Restarting cameras…",
                                    font=(settings.FONT, 22, "bold"),
                                    text_color=settings.CLR_MUTED)
        self._splash.pack(expand=True)
        self.after(150, self._boot_camera)

    # ------------------------------------------------------------------
    def goto(self, page_cls: Type[pages.Page], **kwargs) -> None:
        if self._page is not None:
            self._page.on_leave()
            self._page.destroy()
        self._page = page_cls(self, self, **kwargs)
        self._page.pack(fill="both", expand=True)

    def _tick(self) -> None:
        if not self._alive:
            return
        if self.camera is not None and self._page is not None:
            snap = self.camera.latest()
            if snap is not None and snap.frame_id != self._last_frame_id:
                self._last_frame_id = snap.frame_id
                self._page.on_snapshot(snap)
        self.after(settings.UI_TICK_MS, self._tick)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if not self._alive:
            return
        self._alive = False
        if self._page is not None:
            self._page.on_leave()          # closes any open session files
        if self.camera is not None:
            self.camera.close()
        self.destroy()
