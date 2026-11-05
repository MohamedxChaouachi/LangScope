"""
ocr_providers.py — OCR abstraction layer.

BaseOCR defines the interface.
EasyOCRProvider adds:
  - Configurable image preprocessing (grayscale, CLAHE contrast, sharpening, upscaling)
  - Tunable confidence threshold
  - Per-cycle confidence statistics logging
  - Reduced missed detections on small text via 1.5–2× upscale

Preprocessing pipeline (applied in order when enabled):
  1. Upscale  — bilinear interpolation (scale 1.0–2.0, default 1.5 for small regions)
  2. Grayscale — converts to 1-channel then back to 3-channel (EasyOCR accepts both)
  3. CLAHE contrast — adaptive histogram equalisation on the L channel (Lab)
  4. Sharpening — unsharp-mask via Gaussian blur subtraction
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from models import RawSegment

log = logging.getLogger("langscope.ocr_providers")

# ── Defaults ───────────────────────────────────────────────────────────────────
CONFIDENCE: float = 0.35        # lowered from 0.4 → catches more small text

EASYOCR_LANG_MAP: dict[str, list[str]] = {
    # Latin-script langs pair their native model with "en" (accented chars + ASCII); EasyOCR allows this combo.
    "en": ["en"], "fr": ["fr", "en"], "es": ["es", "en"], "de": ["de", "en"],
    "it": ["it", "en"], "pt": ["pt", "en"], "ro": ["ro", "en"],
    "ru": ["ru"], "zh": ["ch_sim"], "ja": ["ja"], "ar": ["ar"],
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
    grayscale:      bool  = True
    clahe_contrast: bool  = True
    sharpen:        bool  = True
    confidence:     float = CONFIDENCE


def _detect_gpu() -> bool:
    try:
        import torch
        ok = torch.cuda.is_available()
        print(f"[OCR] CUDA GPU: {torch.cuda.get_device_name(0)}" if ok
              else "[OCR] No CUDA GPU — CPU fallback.")
        return ok
    except Exception as exc:
        log.warning(f"[OCR] GPU probe failed ({exc}) — CPU.")
        return False


_USE_GPU: bool = _detect_gpu()

# EasyOCR recognises text boxes in batches. Its default (1) feeds the GPU one
# box at a time; 16 is ~1.5-2x faster on a screen full of text at the cost of a
# little VRAM. Left at the default on CPU, where batching doesn't help.
_OCR_BATCH_SIZE: int = 16 if _USE_GPU else 1


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Apply the configured preprocessing pipeline to *frame* (BGR numpy array).
    Returns a BGR array suitable for EasyOCR (which accepts both gray and BGR).
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

    # 2. Grayscale → back to 3-channel BGR
    if cfg.grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 3. CLAHE contrast enhancement (on L channel of LAB)
    if cfg.clahe_contrast:
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Unsharp-mask sharpening
    if cfg.sharpen:
        blur    = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        img     = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseOCR(ABC):
    @abstractmethod
    def read(self, frame: np.ndarray) -> list[RawSegment]: ...

    def set_language(self, from_code: str) -> None:
        """Notify of a source-language change. Override when model reload is needed."""

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

    _LOG_EVERY = 20   # log confidence stats every N OCR calls

    def __init__(self, from_code: str = "en",
                 preprocess_cfg: PreprocessConfig | None = None):
        self._cache: dict[tuple[str, ...], object] = {}
        self._lock  = threading.Lock()
        self._langs = EASYOCR_LANG_MAP.get(from_code, ["en"])
        self._cfg   = preprocess_cfg or PreprocessConfig()
        self._batch_size = _OCR_BATCH_SIZE
        self._reader = self._get_reader(self._langs)

        # Confidence stats
        self._call_count = 0
        self._score_sum  = 0.0
        self._score_min  = 1.0
        self._score_max  = 0.0
        self._det_count  = 0

    def set_language(self, from_code: str) -> None:
        new = EASYOCR_LANG_MAP.get(from_code, ["en"])
        if new != self._langs:
            self._langs  = new
            self._reader = self._get_reader(new)

    def set_preprocess_config(self, cfg: PreprocessConfig) -> None:
        self._cfg = cfg

    def read(self, frame: np.ndarray) -> list[RawSegment]:
        """Preprocess frame, run EasyOCR, filter by confidence, log stats."""
        processed = preprocess(frame, self._cfg)
        raw       = self._readtext(processed)

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

    def _readtext(self, img: np.ndarray):
        """readtext with GPU batching, backing off to smaller batches if the
        card runs out of VRAM (large regions) instead of failing the frame."""
        while True:
            try:
                return self._reader.readtext(img, detail=1,
                                             batch_size=self._batch_size)
            except Exception as exc:
                oom = "out of memory" in str(exc).lower()
                if not (oom and _USE_GPU and self._batch_size > 1):
                    raise
                import torch
                torch.cuda.empty_cache()
                self._batch_size = max(1, self._batch_size // 2)
                print(f"[OCR] GPU out of memory — retrying with batch_size="
                      f"{self._batch_size}.")

    # ── Reader cache ──────────────────────────────────────────────────────────

    def _get_reader(self, langs: list[str]):
        import easyocr
        key = tuple(sorted(langs))
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            log.info(f"[OCR] Loading EasyOCR reader langs={langs} gpu={_USE_GPU} fp16={USE_FP16}…")
            t0 = time.perf_counter()
            r  = easyocr.Reader(list(langs), gpu=_USE_GPU, verbose=False)
            if USE_FP16:
                self._apply_fp16(r)
            log.info(f"[OCR] Reader ready in {(time.perf_counter()-t0)*1000:.0f} ms.")
            self._cache[key] = r
            return r

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
            self._LOG_EVERY, self._det_count, avg,
            self._score_min, self._score_max, self._cfg.confidence,
        )
        print(
            f"[OCR] Stats — calls={self._call_count}  dets={self._det_count}  "
            f"avg={avg:.2f}  min={self._score_min:.2f}  max={self._score_max:.2f}"
        )
        # Reset window
        self._score_sum = self._score_min = self._score_max = 0.0
        self._score_min = 1.0
        self._det_count = 0
