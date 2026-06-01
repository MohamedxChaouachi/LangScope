"""
engine.py — Capture, OCR, and Translation Worker

Loop design:
  - Capture frame every FRAME_INTERVAL (0.3 s) for a live feed.
  - Run OCR + batch-translate every OCR_INTERVAL (2 s).

Translation filters (applied per detected segment):
  1. Numeric/symbolic only  — fewer than _MIN_ALPHA letters → skip (no overlay).
  2. Already target language — langdetect matches to_code     → skip (no overlay).
  3. from_code == to_code   — same language pair              → skip.
  All other segments are batch-translated in one argostranslate call.
"""

import os, time, logging, re
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from mss import mss
import easyocr
from PyQt6.QtCore import QThread, pyqtSignal

import translator

# ── Optional language detector ────────────────────────────────────────────────
try:
    from langdetect import detect as _ld_detect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0          # make results deterministic
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    print("[Engine] langdetect not installed — target-language filter disabled.")

logging.getLogger("easyocr").setLevel(logging.ERROR)

# ── Timing ────────────────────────────────────────────────────────────────────
FRAME_INTERVAL: float = 0.3   # live feed refresh (seconds)
OCR_INTERVAL:   float = 2.0   # how often OCR + translate runs
CONFIDENCE:     float = 0.4   # EasyOCR confidence threshold

# ── Filter thresholds ─────────────────────────────────────────────────────────
_MIN_ALPHA      = 3    # segments with fewer letters are treated as numeric
_MIN_DETECT_LEN = 8    # shorter strings are skipped for language detection

# Catches timestamps / dates / phone numbers even when they have short
# alphabetic suffixes like timezone codes (CET, UTC, PM) or day abbreviations.
_TIMESTAMP_RE = re.compile(
    r"""
    ^\s*
    (
      # Day abbreviation (optionally followed by a time)
      (mon|tue|wed|thu|fri|sat|sun
      |lun|mar|mer|jeu|ven|sam|dim   # French
      |lun|mie|jue|vie|sáb|dom       # Spanish
      )[\s,.\/\-]*
      (\d{1,2}[:.h]\d{2}(\s*[aApP][mM])?(\s*[A-Z]{2,4})?)?

      |                                    # OR

      # Time with optional AM/PM and/or timezone (e.g.  14:32 CET, 2:30 PM)
      \d{1,2}[:.h]\d{2}
      (\s*[aApP][mM])?
      (\s*[A-Z]{2,5})?

      |                                    # OR

      # Date: DD/MM/YYYY, DD-MM-YY, DD.MM etc.
      \d{1,2}[\/\-\.]\d{1,2}([\/\-\.]\d{2,4})?

      |                                    # OR

      # ISO date  2026-06-01
      \d{4}[\/\-]\d{2}[\/\-]\d{2}

      |                                    # OR

      # Phone number: 7+ digit-like characters with formatting
      \+?[\d\s\-\(\)\.]{7,}
    )
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_numeric_only(text: str) -> bool:
    """
    Return True if the segment should not be translated:
      • fewer than _MIN_ALPHA alphabetic characters  (pure numbers, symbols)
      • OR matches _TIMESTAMP_RE  (time, date, phone — possibly with short
        alphabetic suffixes like timezone codes or day abbreviations)
    """
    stripped = text.strip()
    if sum(1 for c in stripped if c.isalpha()) < _MIN_ALPHA:
        return True
    return bool(_TIMESTAMP_RE.match(stripped))


_EASYOCR_LANGS: dict[str, list[str]] = {
    "en": ["en"], "fr": ["en"], "es": ["en"], "de": ["en"],
    "it": ["en"], "pt": ["en"], "ro": ["en"],
    "ru": ["ru"], "zh": ["ch_sim"], "ja": ["ja"], "ar": ["ar"],
}

Detection = tuple[list, str, str]   # (bbox [[x,y]*4], original, translated)



def _is_target_language(text: str, to_code: str) -> bool:
    """
    Return True if langdetect believes *text* is already in *to_code*.
    Unreliable for very short strings, so skipped below _MIN_DETECT_LEN.
    Falls back to False if langdetect is unavailable or throws.
    """
    if not _LANGDETECT_OK or len(text.strip()) < _MIN_DETECT_LEN:
        return False
    try:
        return _ld_detect(text) == to_code
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CaptureWorker
# ─────────────────────────────────────────────────────────────────────────────

class CaptureWorker(QThread):
    """
    frame_ready(frame_bgr, detections, orig_w, orig_h)
        frame_bgr  : np.ndarray BGR
        detections : list[Detection]  — (bbox, original, translated)
                     When original == translated the UI draws no overlay.
        orig_w/h   : int — needed by UI to scale bbox coordinates.
    """

    frame_ready = pyqtSignal(object, object, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running:   bool = True
        self._region:    dict = {"top": 100, "left": 100, "width": 500, "height": 200}
        self._from_code: str  = "en"
        self._to_code:   str  = "fr"

    def set_region(self, top: int, left: int, width: int, height: int) -> None:
        self._region = {"top": top, "left": left, "width": width, "height": height}

    def set_languages(self, from_code: str, to_code: str) -> None:
        self._from_code = from_code
        self._to_code   = to_code

    def run(self) -> None:
        current_from = self._from_code
        current_to   = self._to_code

        ocr_langs = _EASYOCR_LANGS.get(current_from, ["en"])
        print(f"[CaptureWorker] Loading EasyOCR: {ocr_langs}")
        reader = easyocr.Reader(ocr_langs, gpu=False, verbose=False)
        print("[CaptureWorker] EasyOCR ready.")
        translator.setup_translation_models(current_from, current_to)

        _last_detections: list[Detection] = []
        _last_ocr_time:   float           = 0.0

        with mss() as sct:
            while self._running:
                try:
                    if self._from_code != current_from or self._to_code != current_to:
                        current_from, current_to = self._from_code, self._to_code
                        new_langs = _EASYOCR_LANGS.get(current_from, ["en"])
                        if new_langs != ocr_langs:
                            ocr_langs = new_langs
                            reader = easyocr.Reader(ocr_langs, gpu=False, verbose=False)
                        translator.setup_translation_models(current_from, current_to)
                        _last_ocr_time = 0.0

                    frame = self._capture_frame(sct)
                    orig_h, orig_w = frame.shape[:2]

                    if time.time() - _last_ocr_time >= OCR_INTERVAL:
                        raw = self._run_ocr_raw(reader, frame)
                        _last_detections = self._batch_translate(
                            raw, current_from, current_to
                        )
                        _last_ocr_time = time.time()

                    self.frame_ready.emit(frame.copy(), _last_detections, orig_w, orig_h)

                except Exception as exc:
                    print(f"[CaptureWorker] Error: {exc}")

                time.sleep(FRAME_INTERVAL)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _capture_frame(self, sct: Any) -> np.ndarray:
        return np.array(sct.grab(self._region))[..., :3]   # BGRA → BGR

    def _run_ocr_raw(self, reader: easyocr.Reader, frame: np.ndarray):
        return [
            (bbox, text, score)
            for bbox, text, score in reader.readtext(frame, detail=1)
            if score >= CONFIDENCE and text.strip()
        ]

    def _batch_translate(self, raw, from_code: str, to_code: str) -> list[Detection]:
        """
        Categorise each detected segment, then batch-translate only the ones
        that genuinely need it (one argostranslate call for all of them).

        Pass-through (original == translated, no overlay in UI):
          • same language pair
          • fewer than _MIN_ALPHA alphabetic chars (numbers, timestamps, …)
          • segment already written in the target language (langdetect)
        """
        if not raw:
            return []

        same_lang = from_code == to_code
        to_tx: list[int] = []
        results: list[Detection | None] = [None] * len(raw)

        for i, (bbox, text, _) in enumerate(raw):
            if same_lang or _is_numeric_only(text) or _is_target_language(text, to_code):
                results[i] = (bbox, text, text)   # pass-through
            else:
                to_tx.append(i)

        if to_tx:
            joined = "\n".join(raw[i][1] for i in to_tx)
            trans  = translator.translate_text(joined, from_code, to_code)
            lines  = trans.split("\n")
            for j, idx in enumerate(to_tx):
                bbox, text, _ = raw[idx]
                results[idx]  = (bbox, text, lines[j] if j < len(lines) else text)

        return [
            r if r is not None else (raw[i][0], raw[i][1], raw[i][1])
            for i, r in enumerate(results)
        ]

    def stop(self) -> None:
        self._running = False
        self.wait()
