"""
ocr_providers.py — OCR abstraction layer.

BaseOCR defines the interface.
EasyOCRProvider implements it with EasyOCR:
  - the reader is built lazily, on whichever thread calls load() or read()
  - configurable image preprocessing (upscaling, grayscale, CLAHE contrast, sharpening)
  - a minimum confidence score for detections
  - confidence statistics, logged at debug level every _LOG_EVERY calls
  - on a GPU that runs out of memory, recognition batches are halved until the frame fits

Call init_ocr_backend() once at startup, before creating a provider: it probes the GPU and
fills in BACKEND. Importing this module does not touch torch.

Preprocessing pipeline (each step applied in order when enabled; see PreprocessConfig):
  1. Upscale   — bilinear interpolation by upscale_factor (default 1.5, for small regions)
  2. Grayscale — one channel from here on, which EasyOCR accepts
  3. CLAHE     — adaptive histogram equalisation of the contrast
  4. Sharpen   — unsharp mask (Gaussian blur subtracted)
With grayscale off, the frame stays BGR and CLAHE works on the L channel of its Lab version.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from models import RawSegment

log = logging.getLogger("langscope.ocr_providers")

# ── Defaults ───────────────────────────────────────────────────────────────────
CONFIDENCE: float = 0.35  # minimum score to keep a detection; low enough to catch small text

EASYOCR_LANG_MAP: dict[str, list[str]] = {
    # Latin-script langs pair their native model with "en" (accented chars + ASCII); EasyOCR allows this combo.
    "en": ["en"],
    "fr": ["fr", "en"],
    "es": ["es", "en"],
    "de": ["de", "en"],
    "it": ["it", "en"],
    "pt": ["pt", "en"],
    "ro": ["ro", "en"],
    "ru": ["ru"],
    "zh": ["ch_sim"],
    "ja": ["ja"],
    "ar": ["ar"],
}


@dataclass
class PreprocessConfig:
    """
    Controls the image preprocessing pipeline applied before OCR.

    upscale_factor  — multiply frame dimensions before OCR (1.0 = off).
                      Recommended 1.5 for regions <400 px wide, 1.0 otherwise.
    grayscale       — convert to grayscale (improves contrast for mono text).
    clahe_contrast  — apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    sharpen         — unsharp-mask sharpening to accentuate letter edges.
    confidence      — minimum per-detection score to keep (0–1).
    """

    upscale_factor: float = 1.5
    grayscale: bool = True
    clahe_contrast: bool = True
    sharpen: bool = True
    confidence: float = CONFIDENCE


def _detect_gpu() -> bool:
    try:
        import torch  # pyright: ignore[reportMissingImports]  # heavy; CI does not install it

        ok = torch.cuda.is_available()
        if ok:
            log.info("[OCR] CUDA GPU: %s", torch.cuda.get_device_name(0))
        else:
            log.info("[OCR] No CUDA GPU — CPU fallback.")
        return ok
    except Exception as exc:
        log.warning("[OCR] GPU probe failed (%s) — CPU.", exc)
        return False


@dataclass
class OcrBackend:
    """What the OCR engine runs on. init_ocr_backend() fills it in at startup.

    Until then it describes the CPU, which works everywhere.

    use_gpu     — run EasyOCR on the CUDA GPU.
    batch_size  — EasyOCR recognises text boxes in batches. Its default (1) feeds the GPU one
                  box at a time; 16 is ~1.5-2x faster on a screen full of text at the cost of
                  a little VRAM. It stays 1 on the CPU, where batching doesn't help.
    """

    use_gpu: bool = False
    batch_size: int = 1


BACKEND = OcrBackend()


def init_ocr_backend() -> OcrBackend:
    """Probe for a CUDA GPU and set BACKEND accordingly.

    Call once at startup, before creating an EasyOCRProvider. Importing this module does not
    touch torch or the GPU."""
    BACKEND.use_gpu = _detect_gpu()
    BACKEND.batch_size = 16 if BACKEND.use_gpu else 1
    return BACKEND


# GPU half-precision (fp16) — EXPERIMENTAL, opt-in only.
# EasyOCR doesn't officially support this; we cast the underlying torch
# modules ourselves. It CAN throw a dtype-mismatch error at inference time
# (not just at load) if EasyOCR's forward pass doesn't autocast the input
# tensor. Flip to True only after confirming with bench_ocr.py that a
# single readtext() call succeeds on your GPU. Default OFF so the app
# keeps working out of the box.
USE_FP16: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Apply the configured preprocessing pipeline to *frame* (BGR numpy array).
    Returns an array suitable for EasyOCR (grayscale 1-channel when
    cfg.grayscale is on, BGR 3-channel otherwise — EasyOCR accepts both).

    With cfg.grayscale on, the frame becomes a single channel once and stays one: CLAHE and
    the sharpening run directly on the gray plane, which needs one colour conversion and a
    third of the memory of a BGR image. Only the colour path goes through Lab.
    """
    img = frame.copy()

    # 1. Upscale — bilinear, cheap but effective for small text
    if cfg.upscale_factor > 1.01:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (int(w * cfg.upscale_factor), int(h * cfg.upscale_factor)),
            interpolation=cv2.INTER_LINEAR,
        )

    if cfg.grayscale:
        # Stay single-channel for the rest of the pipeline.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cfg.clahe_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if cfg.sharpen:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
            gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        return gray

    # Colour path (only used when cfg.grayscale is explicitly disabled).
    if cfg.clahe_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lum, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lum = clahe.apply(lum)
        img = cv2.cvtColor(cv2.merge([lum, a, b]), cv2.COLOR_LAB2BGR)

    if cfg.sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


class BaseOCR(ABC):
    @abstractmethod
    def read(self, frame: np.ndarray) -> list[RawSegment]: ...

    def set_language(self, from_code: str) -> None:
        """Notify of a source-language change. Should only record it: any model work belongs
        in load(), which the caller runs where blocking is acceptable."""

    def load(self) -> None:
        """Load whatever model read() needs for the current language, so the first read()
        isn't slow. May block for seconds, or download; call it off the GUI thread. No-op by
        default."""

    def set_preprocess_config(self, cfg: PreprocessConfig) -> None:
        """Update preprocessing config at runtime. No-op by default."""


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR provider
# ─────────────────────────────────────────────────────────────────────────────


class EasyOCRProvider(BaseOCR):
    """
    EasyOCR with:
    - Configurable preprocessing pipeline
    - Per-language reader cache
    - GPU/CPU auto-selection
    - Confidence statistics logging (every N calls)
    """

    _LOG_EVERY = 20  # log confidence stats every N OCR calls

    def __init__(self, from_code: str = "en", preprocess_cfg: PreprocessConfig | None = None):
        self._cache: dict[tuple[str, ...], object] = {}
        self._lock = threading.Lock()
        self._langs = EASYOCR_LANG_MAP.get(from_code, ["en"])
        self._cfg = preprocess_cfg or PreprocessConfig()
        self._batch_size = BACKEND.batch_size
        # No reader is built here: that takes seconds and may download ~200 MB. It happens in
        # load(), or on the first read(), on whichever thread calls them.

        # Confidence stats
        self._call_count = 0
        self._score_sum = 0.0
        self._score_min = 1.0
        self._score_max = 0.0
        self._det_count = 0

        # Split timing — set on every read() call, read by OCRWorker for
        # the debug panel / benchmarks. preprocess_ms + inference_ms == the
        # total ocr_ms OCRWorker already tracked; this just shows the split.
        self.preprocess_ms = 0.0
        self.inference_ms = 0.0

    def set_language(self, from_code: str) -> None:
        # Only records the choice; the matching reader is built by load() or the next read().
        self._langs = EASYOCR_LANG_MAP.get(from_code, ["en"])

    def load(self) -> None:
        self._get_reader(self._langs)

    def set_preprocess_config(self, cfg: PreprocessConfig) -> None:
        self._cfg = cfg

    def read(self, frame: np.ndarray) -> list[RawSegment]:
        """Preprocess frame, run EasyOCR, filter by confidence, log stats."""
        # Fetched before the timers start, so a first-time load isn't counted as inference.
        reader = self._get_reader(self._langs)
        t0 = time.perf_counter()
        processed = preprocess(frame, self._cfg)
        t1 = time.perf_counter()
        raw = self._readtext(reader, processed)
        t2 = time.perf_counter()
        self.preprocess_ms = (t1 - t0) * 1000
        self.inference_ms = (t2 - t1) * 1000

        # Scale bboxes back to original frame coordinates if upscaled
        scale = self._cfg.upscale_factor
        segments: list[RawSegment] = []
        for bbox, text, score in raw:
            if not text.strip():
                continue
            # Rescale bbox corners back to source-frame space
            if scale > 1.01:
                bbox = [[pt[0] / scale, pt[1] / scale] for pt in bbox]
            self._update_stats(score)
            if score >= self._cfg.confidence:
                segments.append(RawSegment(bbox=bbox, text=text, score=float(score)))

        self._call_count += 1
        if self._call_count % self._LOG_EVERY == 0:
            self._log_stats()

        return segments

    def _readtext(self, reader: Any, img: np.ndarray) -> list[tuple[Any, str, float]]:
        """readtext with GPU batching, backing off to smaller batches if the
        card runs out of VRAM (large regions) instead of failing the frame."""
        while True:
            try:
                return reader.readtext(img, detail=1, batch_size=self._batch_size)
            except Exception as exc:
                oom = "out of memory" in str(exc).lower()
                if not (oom and BACKEND.use_gpu and self._batch_size > 1):
                    raise
                import torch  # pyright: ignore[reportMissingImports]

                torch.cuda.empty_cache()
                self._batch_size = max(1, self._batch_size // 2)
                log.warning(
                    "[OCR] GPU out of memory — retrying with batch_size=%d.", self._batch_size
                )

    # ── Reader cache ──────────────────────────────────────────────────────────

    def _get_reader(self, langs: list[str]) -> Any:
        import easyocr  # pyright: ignore[reportMissingImports]  # heavy; CI does not install it

        key = tuple(sorted(langs))
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            log.info(
                "[OCR] Loading EasyOCR reader langs=%s gpu=%s fp16=%s…",
                langs,
                BACKEND.use_gpu,
                USE_FP16,
            )
            t0 = time.perf_counter()
            r = easyocr.Reader(list(langs), gpu=BACKEND.use_gpu, verbose=False)
            if USE_FP16:
                self._apply_fp16(r)
            log.info("[OCR] Reader ready in %.0f ms.", (time.perf_counter() - t0) * 1000)
            self._cache[key] = r
            return r

    @staticmethod
    def _apply_fp16(reader: Any) -> None:
        """
        Cast the detector + recognizer nets to fp16 in place. EasyOCR
        doesn't expose this as a constructor argument, so we reach into
        the underlying torch modules directly. Guarded — if EasyOCR's
        internals differ across versions this just no-ops back to fp32
        instead of crashing the app.
        """
        try:
            reader.detector = reader.detector.half()
            if hasattr(reader, "recognizer") and reader.recognizer is not None:
                reader.recognizer = reader.recognizer.half()
            log.info("[OCR] fp16 applied to detector + recognizer.")
        except Exception as exc:
            log.warning("[OCR] fp16 cast failed (%s) — staying fp32.", exc)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _update_stats(self, score: float) -> None:
        self._det_count += 1
        self._score_sum += score
        if score < self._score_min:
            self._score_min = score
        if score > self._score_max:
            self._score_max = score

    def _log_stats(self) -> None:
        if self._det_count == 0:
            log.debug("[OCR] Stats: 0 detections in last %d calls", self._LOG_EVERY)
            return
        avg = self._score_sum / self._det_count
        log.debug(
            "[OCR] Confidence stats (last %d calls): "
            "detections=%d  avg=%.2f  min=%.2f  max=%.2f  threshold=%.2f",
            self._LOG_EVERY,
            self._det_count,
            avg,
            self._score_min,
            self._score_max,
            self._cfg.confidence,
        )
        log.debug(
            "[OCR] Stats — calls=%d  dets=%d  avg=%.2f  min=%.2f  max=%.2f",
            self._call_count,
            self._det_count,
            avg,
            self._score_min,
            self._score_max,
        )
        # Reset window
        self._score_sum = self._score_min = self._score_max = 0.0
        self._score_min = 1.0
        self._det_count = 0
