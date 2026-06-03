"""
ocr_providers.py — OCR abstraction layer.

BaseOCR defines the interface.
EasyOCRProvider is the production implementation (GPU-accelerated via torch.cuda).
Future engines (PaddleOCR, Tesseract…) implement BaseOCR without touching any other module.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from models import RawSegment

log = logging.getLogger("langscope.ocr_providers")

CONFIDENCE: float = 0.4

EASYOCR_LANG_MAP: dict[str, list[str]] = {
    "en": ["en"], "fr": ["en"], "es": ["en"], "de": ["en"],
    "it": ["en"], "pt": ["en"], "ro": ["en"],
    "ru": ["ru"], "zh": ["ch_sim"], "ja": ["ja"], "ar": ["ar"],
}


def _detect_gpu() -> bool:
    try:
        import torch
        ok = torch.cuda.is_available()
        if ok:
            print(f"[OCR] CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[OCR] No CUDA GPU — CPU fallback.")
        return ok
    except Exception as exc:
        print(f"[OCR] GPU probe failed ({exc}) — CPU.")
        return False


_USE_GPU: bool = _detect_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseOCR(ABC):
    @abstractmethod
    def read(self, frame: np.ndarray) -> list[RawSegment]: ...

    def set_language(self, from_code: str) -> None:
        """Notify of a source-language change. Override when model reload is needed."""


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR
# ─────────────────────────────────────────────────────────────────────────────

class EasyOCRProvider(BaseOCR):
    """EasyOCR with per-language reader cache and GPU/CPU auto-selection."""

    def __init__(self, from_code: str = "en"):
        self._cache: dict[tuple[str, ...], object] = {}
        self._lock  = threading.Lock()
        self._langs = EASYOCR_LANG_MAP.get(from_code, ["en"])
        self._reader = self._get_reader(self._langs)

    def set_language(self, from_code: str) -> None:
        new = EASYOCR_LANG_MAP.get(from_code, ["en"])
        if new != self._langs:
            self._langs  = new
            self._reader = self._get_reader(new)

    def read(self, frame: np.ndarray) -> list[RawSegment]:
        results = self._reader.readtext(frame, detail=1)
        return [
            RawSegment(bbox=bbox, text=text, score=float(score))
            for bbox, text, score in results
            if score >= CONFIDENCE and text.strip()
        ]

    def _get_reader(self, langs: list[str]):
        import easyocr
        key = tuple(sorted(langs))
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            print(f"[OCR] Loading EasyOCR reader langs={langs} gpu={_USE_GPU}…")
            t0 = time.perf_counter()
            r  = easyocr.Reader(list(langs), gpu=_USE_GPU, verbose=False)
            print(f"[OCR] Reader ready in {(time.perf_counter()-t0)*1000:.0f} ms.")
            self._cache[key] = r
            return r
