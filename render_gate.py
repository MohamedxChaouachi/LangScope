"""
render_gate.py — decides what the 60 FPS render timer needs to do on each tick.

The capture thread replaces CaptureManager.latest_frame, and the translation worker replaces
TranslationWorker.latest_regions, as whole objects. So "has anything changed since the last
tick?" is an identity check on those two references, which costs nothing, whereas converting
and copying a full frame to repaint an identical picture 60 times a second does.

The gate also counts the frames that were really drawn, so the FPS label shows the drawing
rate rather than the timer rate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

FPS_WINDOW = 1.0  # seconds per FPS reading


class Tick(NamedTuple):
    render: bool  # True if the frame or the regions changed, so the UI needs updating
    fps: float | None  # frames drawn per second, once per FPS_WINDOW; otherwise None


class RenderGate:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._last_frame: object = None
        self._last_regions: object = None
        self._window_start: float | None = None
        self._drawn = 0

    def invalidate(self) -> None:
        """Make the next tick draw even if nothing changed, for example after the desktop
        overlay is switched on while the screen is static."""
        self._last_frame = None
        self._last_regions = None

    def tick(self, frame: object, regions: object) -> Tick:
        now = self._clock()
        if self._window_start is None:
            # The window starts at the first tick. It used to start at 0.0, which made the
            # first reading divide by the time since boot instead of by one second.
            self._window_start = now

        # Close the window first: the tick that ends one window is the first of the next, so
        # each reading counts only the frames drawn inside its own span.
        fps = None
        elapsed = now - self._window_start
        if elapsed >= FPS_WINDOW:
            fps = self._drawn / elapsed
            self._drawn = 0
            self._window_start = now

        render = frame is not None and (
            frame is not self._last_frame or regions is not self._last_regions
        )
        if render:
            self._last_frame = frame
            self._last_regions = regions
            self._drawn += 1
        return Tick(render, fps)
