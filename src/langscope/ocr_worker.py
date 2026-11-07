"""
ocr_worker.py — OCR worker thread (8 FPS default).

Reads CaptureManager.latest_frame and asks a FrameChangeGate (a frame hash, then a count of
noticeably changed pixels) whether the screen changed enough to be worth OCR. If so, it runs
OCR and emits raw_segments_ready(list[RawSegment]).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from langscope.capture_manager import CaptureManager
from langscope.ocr_providers import BaseOCR, EasyOCRProvider

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


class FrameChangeGate:
    """
    Decides whether a captured frame is worth running OCR on.

    It remembers the last frame that was sent to OCR and compares each new frame with that one,
    not with the previous frame, so many tiny changes add up instead of slipping through. Two
    checks, cheapest first: an identical hash means nothing changed; otherwise, unless the frame
    has a new size, enough pixels must have changed noticeably (see _content_changed).
    """

    def __init__(self) -> None:
        self._last_hash = ""
        self._last_frame: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the reference, so the next frame is always OCR'd."""
        self._last_hash = ""
        self._last_frame = None

    def should_ocr(self, frame: np.ndarray) -> bool:
        """True if *frame* should go to OCR, in which case it becomes the new reference."""
        fhash = _hash_frame(frame)

        # Check 1: identical hash → skip
        if fhash == self._last_hash:
            return False

        # Check 2: nothing meaningful changed → skip
        if self._last_frame is not None and self._last_frame.shape == frame.shape:
            if not _content_changed(frame, self._last_frame):
                return False

        self._last_hash = fhash
        self._last_frame = frame.copy()  # the capture thread may reuse the buffer
        return True


_NOT_LOADED = object()  # "no language has been loaded yet" (None is a real choice: the default)


class OCRWorker(QThread):
    """
    QThread: polls CaptureManager.latest_frame, runs OCR on changed frames.

    The OCR model is loaded here, on the worker thread, never on the GUI thread: building an
    EasyOCR reader takes seconds and may download ~200 MB. The provider is only ever called
    from this thread, and set_languages() merely records the choice for the loop to act on.

    Signals:
        raw_segments_ready(list[RawSegment])
        status_changed(str)   — "Loading OCR model…", then "OCR model ready" (or a failure)
    """

    raw_segments_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(
        self, capture: CaptureManager, ocr: BaseOCR | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._capture = capture
        self._ocr = ocr or EasyOCRProvider()
        # Set by stop(), cleared by start() before the thread exists. run() never touches it,
        # so a stop() that arrives before the thread gets going can't be undone.
        self._stop_requested = threading.Event()
        # Written by the GUI thread only (plain assignments); read by the worker.
        self._wanted_language: str | None = None
        self._language_requests = 0  # counts set_languages() calls, even repeats of a language
        self._gate = FrameChangeGate()

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

    # ── Public API (main thread) ──────────────────────────────────────────────

    def start(self, *args: Any, **kwargs: Any) -> None:
        self._stop_requested.clear()
        super().start(*args, **kwargs)

    def set_languages(self, from_code: str, to_code: str) -> None:
        # Only records the choice. The worker thread tells the provider, loads the model and
        # re-runs OCR on the current frame, so nothing slow happens here.
        self._wanted_language = from_code
        self._language_requests += 1

    def stop(self) -> None:
        self._stop_requested.set()
        self.wait()

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _load_model(self, language: str | None) -> bool:
        """Point the provider at *language* and load its model. False if that failed."""
        self.status_changed.emit("Loading OCR model…")
        try:
            if language is not None:
                self._ocr.set_language(language)
            self._ocr.load()
        except Exception:
            log.exception("Could not load the OCR model")
            self.status_changed.emit("OCR model failed to load — see the log")
            return False
        self.status_changed.emit("OCR model ready")
        return True

    def run(self) -> None:
        log.debug("OCRWorker started (%.0f FPS)", OCR_FPS)
        # Forget the previous region's frame so the first frame after a restart
        # is always OCR'd, even if the screen content hasn't changed.
        self._gate.reset()
        loaded_language: object = _NOT_LOADED
        model_ready = False
        seen_requests = self._language_requests
        while not self._stop_requested.is_set():
            t0 = time.perf_counter()

            # Any set_languages() call forces a fresh OCR pass, even when the OCR language is
            # unchanged: the translator has dropped its regions and needs new segments to fill
            # them again. Both change gates must be reset, or a static screen is still skipped.
            requests = self._language_requests
            if requests != seen_requests:
                seen_requests = requests
                self._gate.reset()

            # Load the model for the wanted language. A failure is not retried on every pass:
            # it stays failed until the language changes or the worker is restarted.
            wanted = self._wanted_language
            if wanted != loaded_language:
                loaded_language = wanted
                model_ready = self._load_model(wanted)
                self._gate.reset()  # the new model must see the current frame again
            if not model_ready:
                self._stop_requested.wait(OCR_INTERVAL)
                continue

            frame = self._capture.latest_frame

            if frame is None:
                self._stop_requested.wait(OCR_INTERVAL)
                continue

            if not self._gate.should_ocr(frame):
                self.frames_skipped += 1
                self._sleep(t0)
                continue

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
        self._stop_requested.wait(max(0.0, OCR_INTERVAL - (time.perf_counter() - t0)))
