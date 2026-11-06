"""
ocr_worker.py — OCR worker thread (8 FPS default).

Reads CaptureManager.latest_frame, applies frame-hash + motion-threshold
gating so OCR only fires when the screen actually changes, then emits
raw_segments_ready(list[RawSegment]).
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from capture_manager import CaptureManager
from ocr_providers import BaseOCR, EasyOCRProvider

log = logging.getLogger("langscope.ocr_worker")

OCR_FPS = 8.0
OCR_INTERVAL = 1.0 / OCR_FPS
PIXEL_DELTA = 40  # per-pixel intensity change that counts as "changed"
MIN_CHANGED_PX = 12  # changed pixels (at half resolution) needed to re-run OCR


def _hash_frame(frame: np.ndarray) -> str:
    ds = np.ascontiguousarray(frame[::4, ::4])
    return hashlib.md5(ds.tobytes()).hexdigest()


def _content_changed(frame: np.ndarray, prev: np.ndarray) -> bool:
    """
    True if enough pixels changed noticeably to warrant a new OCR pass.

    Counts changed pixels instead of averaging the difference over the whole
    frame: replacing one line of text in a large region barely moves the mean
    (≈0.6 on 1920x1080) but changes thousands of pixels, and a stale
    translation would otherwise stay on screen. The per-pixel delta ignores
    compression noise; the pixel minimum ignores a blinking caret.
    """
    a = frame[::2, ::2].astype(np.int16)
    b = prev[::2, ::2].astype(np.int16)
    changed = np.abs(a - b).max(axis=2) > PIXEL_DELTA
    return int(np.count_nonzero(changed)) >= MIN_CHANGED_PX


class OCRWorker(QThread):
    """
    QThread: polls CaptureManager.latest_frame, runs OCR on changed frames.

    Signals:
        raw_segments_ready(list[RawSegment])
    """

    raw_segments_ready = pyqtSignal(object)

    def __init__(self, capture: CaptureManager, ocr: BaseOCR | None = None, parent=None):
        super().__init__(parent)
        self._capture = capture
        self._ocr = ocr or EasyOCRProvider()
        self._running = True
        self._last_hash = ""
        self._last_frame: np.ndarray | None = None

        # Exposed metrics
        self.ocr_ms = 0.0
        self.preprocess_ms = 0.0  # split out of ocr_ms — see ocr_providers.read()
        self.inference_ms = 0.0
        self.frames_skipped = 0
        self.ocr_calls = 0

        # Rolling average over the last _AVG_WINDOW calls — smooths out
        # single-frame noise so the number on screen / in logs is actually
        # usable for a before/after comparison instead of jumping around.
        self._AVG_WINDOW = 20
        self._recent_ms: list[float] = []
        self.avg_ocr_ms = 0.0

    # ── Public setters (main thread) ──────────────────────────────────────────

    def set_languages(self, from_code: str, to_code: str) -> None:
        self._ocr.set_language(from_code)
        self._last_hash = ""  # force re-OCR

    def stop(self) -> None:
        self._running = False
        self.wait()

    # ── Worker loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.debug("OCRWorker started (%.0f FPS)", OCR_FPS)
        # stop() leaves _running False; without this reset a restarted thread
        # would exit immediately and OCR would silently never run again.
        self._running = True
        # Forget the previous region's frame so the first frame after a restart
        # is always OCR'd, even if the screen content hasn't changed.
        self._last_hash = ""
        self._last_frame = None
        while self._running:
            t0 = time.perf_counter()
            frame = self._capture.latest_frame

            if frame is None:
                time.sleep(OCR_INTERVAL)
                continue

            fhash = _hash_frame(frame)

            # Gate 1: identical hash → skip
            if fhash == self._last_hash:
                self.frames_skipped += 1
                self._sleep(t0)
                continue

            # Gate 2: nothing meaningful changed → skip
            if self._last_frame is not None and self._last_frame.shape == frame.shape:
                if not _content_changed(frame, self._last_frame):
                    self.frames_skipped += 1
                    self._sleep(t0)
                    continue

            self._last_hash = fhash
            self._last_frame = frame.copy()

            try:
                t_ocr = time.perf_counter()
                segs = self._ocr.read(frame)
                self.ocr_ms = (time.perf_counter() - t_ocr) * 1000
                self.ocr_calls += 1

                # Pull the preprocess/inference split if the provider tracks it
                self.preprocess_ms = getattr(self._ocr, "preprocess_ms", 0.0)
                self.inference_ms = getattr(self._ocr, "inference_ms", 0.0)

                self._recent_ms.append(self.ocr_ms)
                if len(self._recent_ms) > self._AVG_WINDOW:
                    self._recent_ms.pop(0)
                self.avg_ocr_ms = sum(self._recent_ms) / len(self._recent_ms)

                log.debug(
                    "OCR %.1f ms (prep=%.1f inf=%.1f, avg=%.1f) — %d segs",
                    self.ocr_ms,
                    self.preprocess_ms,
                    self.inference_ms,
                    self.avg_ocr_ms,
                    len(segs),
                )
                self.raw_segments_ready.emit(segs)
            except Exception as exc:
                log.exception("OCR error: %s", exc)

            self._sleep(t0)

    def _sleep(self, t0: float) -> None:
        time.sleep(max(0.0, OCR_INTERVAL - (time.perf_counter() - t0)))
