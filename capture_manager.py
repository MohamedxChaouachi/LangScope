"""
capture_manager.py — Continuous screen capture at 60 FPS.

Backend priority:
  1. DXCam  — Windows DXGI desktop duplication, lowest latency
  2. MSS    — cross-platform fallback

CaptureManager runs a single daemon thread that writes `latest_frame`
continuously. Readers (OCRWorker, render timer) simply read that attribute.

Thread safety: Python's GIL makes a single-reference attribute assignment
atomic for readers. A Lock guards the mutable region dict.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("langscope.capture")

TARGET_FPS = 60.0
FRAME_INTERVAL = 1.0 / TARGET_FPS


def _primary_monitor_rect() -> tuple[int, int, int, int]:
    """(left, top, width, height) of the primary monitor in desktop pixels."""
    from mss import mss as _mss

    with _mss() as sct:
        monitors = sct.monitors[1:] or sct.monitors[:1]
    for m in monitors:
        if m.get("is_primary"):
            return m["left"], m["top"], m["width"], m["height"]
    m = monitors[0]
    return m["left"], m["top"], m["width"], m["height"]


def _try_dxcam():
    try:
        import dxcam

        return dxcam
    except ImportError:
        return None


class CaptureManager:
    """
    Manages a background daemon thread that continuously grabs screen frames.

        cm = CaptureManager()
        cm.start({"top":…, "left":…, "width":…, "height":…})
        frame = cm.latest_frame   # None before first grab
        cm.stop()
    """

    def __init__(self):
        self.latest_frame: np.ndarray | None = None
        self._region: dict = {"top": 0, "left": 0, "width": 800, "height": 600}
        self._region_lock = threading.Lock()
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._dxcam = _try_dxcam()
        self._backend: str = "dxcam" if self._dxcam else "mss"
        log.info(f"[Capture] Backend: {self._backend}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self, region: dict) -> None:
        self.set_region(region)
        if self._running:
            return
        # Don't let OCR / the live feed see the previous region's last frame.
        self.latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="CaptureThread", daemon=True)
        self._thread.start()
        log.debug("CaptureManager started (backend=%s)", self._backend)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        log.debug("CaptureManager stopped.")

    def set_region(self, region: dict) -> None:
        with self._region_lock:
            self._region = dict(region)

    # ── Loops ─────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        if self._backend == "dxcam":
            self._loop_dxcam()
        else:
            self._loop_mss()

    def _loop_dxcam(self) -> None:
        try:
            camera = self._dxcam.create(output_color="BGR")
        except Exception as exc:
            log.warning(f"[Capture] DXCam init failed ({exc}) — switching to MSS.")
            log.warning("DXCam init failed: %s", exc)
            self._backend = "mss"
            self._loop_mss()
            return
        # dxcam.create() with no arguments captures the primary output only,
        # and grab() regions are relative to that output. The primary monitor
        # always sits at the desktop origin, so global == local coordinates for
        # regions inside it. Regions on any other monitor are grabbed with MSS,
        # which works in global virtual-desktop coordinates.
        sct = None
        try:
            from mss import mss as _mss

            px, py, pw, ph = _primary_monitor_rect()
            while self._running:
                t0 = time.perf_counter()
                with self._region_lock:
                    r = dict(self._region)
                inside_primary = (
                    r["left"] >= px
                    and r["top"] >= py
                    and r["left"] + r["width"] <= px + pw
                    and r["top"] + r["height"] <= py + ph
                )
                if inside_primary:
                    frame = camera.grab(
                        region=(
                            r["left"] - px,
                            r["top"] - py,
                            r["left"] - px + r["width"],
                            r["top"] - py + r["height"],
                        )
                    )
                else:
                    if sct is None:
                        sct = _mss()
                    frame = np.array(sct.grab(r))[..., :3]  # BGRA → BGR
                if frame is not None:
                    self.latest_frame = frame
                time.sleep(max(0.0, FRAME_INTERVAL - (time.perf_counter() - t0)))
        except Exception as exc:
            log.error(f"[Capture] DXCam error ({exc}) — switching to MSS.")
            log.exception("DXCam error: %s", exc)
            self._backend = "mss"
            self._loop_mss()
        finally:
            if sct is not None:
                try:
                    sct.close()
                except Exception:
                    pass
            try:
                del camera
            except Exception:
                pass

    def _loop_mss(self) -> None:
        from mss import mss as _mss

        with _mss() as sct:
            while self._running:
                t0 = time.perf_counter()
                with self._region_lock:
                    r = dict(self._region)
                screenshot = sct.grab(r)
                self.latest_frame = np.array(screenshot)[..., :3]  # BGRA → BGR
                time.sleep(max(0.0, FRAME_INTERVAL - (time.perf_counter() - t0)))
