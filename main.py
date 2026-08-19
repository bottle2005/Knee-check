"""KneeCheck — run with ``python main.py``.  Fully offline.

Importing MediaPipe takes several seconds (it loads a large native library),
so a small splash window is shown immediately and the heavy import runs on a
worker thread.  Without this the app looks frozen while Python loads.
"""
from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings  # noqa: E402  (light: no third-party imports)


def _splash() -> tk.Tk:
    """Small borderless 'starting…' window, centred on screen."""
    win = tk.Tk()
    win.overrideredirect(True)
    win.configure(bg=settings.CLR_PANEL)
    w, h = 360, 130
    x = (win.winfo_screenwidth() - w) // 2
    y = (win.winfo_screenheight() - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")
    tk.Label(win, text="KneeCheck", font=(settings.FONT, 20, "bold"),
             fg=settings.CLR_TEXT, bg=settings.CLR_PANEL).pack(pady=(26, 4))
    win.status = tk.Label(win, text="Starting…", font=(settings.FONT, 11),
                          fg=settings.CLR_MUTED, bg=settings.CLR_PANEL)
    win.status.pack()
    win.update()
    return win


def main() -> None:
    splash = _splash()

    loaded: dict = {}

    def load_modules() -> None:
        from gui.app import KneeCheckApp        # pulls in MediaPipe/OpenCV
        loaded["app_cls"] = KneeCheckApp

    worker = threading.Thread(target=load_modules, daemon=True)
    worker.start()

    # keep the splash painted (and Windows happy) while the import runs
    dots = 0
    while worker.is_alive():
        dots = (dots + 1) % 4
        splash.status.configure(text="Loading motion tracking" + "." * dots)
        splash.update()
        time.sleep(0.12)
    worker.join()

    splash.destroy()
    loaded["app_cls"]().mainloop()


if __name__ == "__main__":
    main()
