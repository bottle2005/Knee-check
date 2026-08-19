"""Reusable themed widgets."""
from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import settings


class VideoView(ctk.CTkFrame):
    """Rounded card that displays a live BGR frame stream efficiently.

    The PhotoImage is created once per size and updated with ``paste()``, so
    no per-frame widget re-layout happens.
    """

    def __init__(self, parent) -> None:
        super().__init__(parent, corner_radius=18, fg_color=settings.CLR_PANEL)
        self._label = tk.Label(self, bg="#000000", bd=0, highlightthickness=0)
        self._label.pack(fill="both", expand=True, padx=6, pady=6)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._size: Tuple[int, int] = (0, 0)

    def push(self, bgr: np.ndarray) -> None:
        w, h = self._label.winfo_width(), self._label.winfo_height()
        if w < 40 or h < 40:
            return
        fh, fw = bgr.shape[:2]
        k = min(w / fw, h / fh)
        nw, nh = max(1, int(fw * k)), max(1, int(fh * k))
        scaled = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        board = np.zeros((h, w, 3), np.uint8)
        x, y = (w - nw) // 2, (h - nh) // 2
        board[y:y + nh, x:x + nw] = scaled
        img = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        if self._photo is None or self._size != (w, h):
            self._photo = ImageTk.PhotoImage(img)
            self._size = (w, h)
            self._label.configure(image=self._photo)
        else:
            self._photo.paste(img)


class StatCard(ctk.CTkFrame):
    """Small card: muted caption, big value, and an optional status line.

    The big value is meant for short text (a number); longer explanations go
    in the wrapped status line so they never blow the card's layout apart.
    """

    def __init__(self, parent, caption: str) -> None:
        super().__init__(parent, corner_radius=14, fg_color=settings.CLR_PANEL)
        ctk.CTkLabel(self, text=caption.upper(), text_color=settings.CLR_MUTED,
                     font=(settings.FONT, 11, "bold")).pack(anchor="w", padx=14, pady=(10, 0))
        self._value = ctk.CTkLabel(self, text="--", text_color=settings.CLR_TEXT,
                                   font=(settings.FONT, 26, "bold"), anchor="w")
        self._value.pack(anchor="w", padx=14, pady=(0, 2))
        self._sub = ctk.CTkLabel(self, text="", text_color=settings.CLR_MUTED,
                                 font=(settings.FONT, 12), justify="left",
                                 wraplength=280, anchor="w")
        self._sub_shown = False

    def set(self, text: str, color: str = settings.CLR_TEXT,
            sub: str = "") -> None:
        self._value.configure(text=text, text_color=color)
        if sub and not self._sub_shown:
            self._sub.pack(anchor="w", padx=14, pady=(0, 10))
            self._sub_shown = True
        elif not sub and self._sub_shown:
            self._sub.pack_forget()
            self._sub_shown = False
        if sub:
            self._sub.configure(text=sub)


class PainMeter(ctk.CTkFrame):
    """Pain-level meter: colored bar + Low/Moderate/High label."""

    def __init__(self, parent) -> None:
        super().__init__(parent, corner_radius=14, fg_color=settings.CLR_PANEL)
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(top, text="PAIN LEVEL", text_color=settings.CLR_MUTED,
                     font=(settings.FONT, 11, "bold")).pack(side="left")
        self._level = ctk.CTkLabel(top, text="Low", text_color=settings.CLR_OK,
                                   font=(settings.FONT, 14, "bold"))
        self._level.pack(side="right")
        self._bar = ctk.CTkProgressBar(self, height=14, corner_radius=7,
                                       progress_color=settings.CLR_OK,
                                       fg_color=settings.CLR_PANEL_2)
        self._bar.pack(fill="x", padx=14, pady=(0, 12))
        self._bar.set(0.0)

    def set(self, score: float, level: str) -> None:
        color = {"Low": settings.CLR_OK, "Moderate": settings.CLR_WARN,
                 "High": settings.CLR_BAD}.get(level, settings.CLR_OK)
        self._bar.set(min(1.0, score))
        self._bar.configure(progress_color=color)
        self._level.configure(text=level, text_color=color)


class StarRow(ctk.CTkFrame):
    """One gold star per completed rep in the current set."""

    def __init__(self, parent, total: int) -> None:
        super().__init__(parent, fg_color="transparent")
        self._stars = []
        for _ in range(total):
            lbl = ctk.CTkLabel(self, text="☆", text_color=settings.CLR_MUTED,
                               font=(settings.FONT, 22))
            lbl.pack(side="left", padx=1)
            self._stars.append(lbl)
        self._lit = -1

    def set(self, count: int) -> None:
        if count == self._lit:
            return
        self._lit = count
        for i, lbl in enumerate(self._stars):
            if i < count:
                lbl.configure(text="★", text_color=settings.CLR_STAR)
            else:
                lbl.configure(text="☆", text_color=settings.CLR_MUTED)


class Stepper(ctk.CTkFrame):
    """A large, easy number picker:  [ - ]  value  [ + ]."""

    def __init__(self, parent, value: int, minimum: int, maximum: int,
                 step: int = 1, suffix: str = "") -> None:
        super().__init__(parent, fg_color=settings.CLR_PANEL, corner_radius=12)
        self._value = value
        self._min, self._max, self._step = minimum, maximum, step
        self._suffix = suffix

        btn = dict(width=52, height=40, corner_radius=10,
                   fg_color=settings.CLR_PANEL_2, hover_color="#28374f",
                   text_color=settings.CLR_TEXT, font=(settings.FONT, 20, "bold"))
        # + is packed BEFORE the flexible label so it can never be squeezed
        # out of a narrow panel.
        ctk.CTkButton(self, text="−", command=lambda: self._bump(-1), **btn
                      ).pack(side="left", padx=(8, 0), pady=6)
        ctk.CTkButton(self, text="+", command=lambda: self._bump(+1), **btn
                      ).pack(side="right", padx=(0, 8), pady=6)
        self._label = ctk.CTkLabel(self, text="", font=(settings.FONT, 20, "bold"),
                                   text_color=settings.CLR_TEXT, width=40)
        self._label.pack(side="left", expand=True, fill="x")
        self._refresh()

    def _bump(self, direction: int) -> None:
        self._value = max(self._min, min(self._max, self._value + direction * self._step))
        self._refresh()

    def _refresh(self) -> None:
        self._label.configure(text=f"{self._value}{self._suffix}")

    def get(self) -> int:
        return self._value


def big_button(parent, text: str, command, primary: bool = True,
               danger: bool = False) -> ctk.CTkButton:
    """Large rounded action button."""
    if danger:
        fg, hover = "transparent", "#3b1c26"
        return ctk.CTkButton(parent, text=text, command=command, height=46,
                             corner_radius=12, fg_color=fg, hover_color=hover,
                             border_width=2, border_color=settings.CLR_BAD,
                             text_color=settings.CLR_BAD,
                             font=(settings.FONT, 15, "bold"))
    fg = settings.CLR_ACCENT if primary else settings.CLR_PANEL_2
    hover = settings.CLR_ACCENT_DARK if primary else "#28374f"
    text_color = "#06281f" if primary else settings.CLR_TEXT
    return ctk.CTkButton(parent, text=text, command=command, height=52,
                         corner_radius=14, fg_color=fg, hover_color=hover,
                         text_color=text_color, font=(settings.FONT, 16, "bold"))
