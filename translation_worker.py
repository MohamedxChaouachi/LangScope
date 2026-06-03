"""
translation_worker.py — Translation worker thread.

Receives list[RawSegment] from OCRWorker via an internal queue.
Deduplicates by text-MD5 so unchanged segments are never retranslated.
Applies the same language / numeric / timestamp filters from the old engine.py.
Maintains `latest_regions` (read by render timer without blocking).
Emits regions_ready(list[TranslationRegion]) after each cycle.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import re
import time

from PyQt6.QtCore import QThread, pyqtSignal

from models import RawSegment, TranslationRegion
from translation_providers import BaseTranslator, ArgosTranslator

log = logging.getLogger("langscope.translation_worker")

REGION_TTL  = 8.0   # seconds — expired regions are pruned
_MIN_ALPHA  = 3
_MIN_DETECT = 8

_TIMESTAMP_RE = re.compile(
    r"""^\s*(
      (?:mon|tue|wed|thu|fri|sat|sun|lun|mar|mer|jeu|ven|sam|dim
        |lun|mie|jue|vie|sáb|dom)[\s,.\\/\-]*
      (?:\d{1,2}[:.h]\d{2}(?:\s*[aApP][mM])?(?:\s*[A-Z]{2,4})?)?
      | \d{1,2}[:.h]\d{2}(?:\s*[aApP][mM])?(?:\s*[A-Z]{2,5})?
      | \d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?
      | \d{4}[\/\-]\d{2}[\/\-]\d{2}
      | \+?[\d\s\-\(\)\.]{7,}
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _numeric(text: str) -> bool:
    s = text.strip()
    return sum(c.isalpha() for c in s) < _MIN_ALPHA or bool(_TIMESTAMP_RE.match(s))


def _already_target(text: str, to_code: str) -> bool:
    if len(text.strip()) < _MIN_DETECT:
        return False
    try:
        from langdetect import detect
        return detect(text) == to_code
    except Exception:
        return False


def _thash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


class TranslationWorker(QThread):
    """
    QThread: deduplicates and translates OCR segments.

    Input:  enqueue(list[RawSegment])  — called from OCRWorker's signal.
    Output: regions_ready(list[TranslationRegion])
    """

    regions_ready = pyqtSignal(object)   # list[TranslationRegion]

    def __init__(self, translator: BaseTranslator | None = None, parent=None):
        super().__init__(parent)
        self._translator: BaseTranslator        = translator or ArgosTranslator()
        self._from_code   = "en"
        self._to_code     = "fr"
        self._running     = True
        self._queue: queue.Queue                = queue.Queue(maxsize=2)
        # text-hash → TranslationRegion (dedup cache)
        self._cache: dict[str, TranslationRegion] = {}
        # latest complete set — read by render timer
        self.latest_regions: list[TranslationRegion] = []

        self.translate_ms  = 0.0
        self.cache_hits    = 0
        self.cache_misses  = 0

    # ── Public API (main thread) ──────────────────────────────────────────────

    def set_languages(self, from_code: str, to_code: str) -> None:
        self._from_code = from_code
        self._to_code   = to_code
        self._cache.clear()
        self.latest_regions = []
        self._translator.clear_cache()

    def enqueue(self, segments: list[RawSegment]) -> None:
        """Non-blocking put; drops oldest entry if queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(segments)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)   # unblock .get()
        except queue.Full:
            pass
        self.wait()

    # ── Worker loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.debug("TranslationWorker started")
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._prune()
                continue

            if item is None or not self._running:
                break

            regions = self._process(item)
            self.latest_regions = regions
            self.regions_ready.emit(regions)

    # ── Processing ────────────────────────────────────────────────────────────

    def _process(self, segments: list[RawSegment]) -> list[TranslationRegion]:
        same    = self._from_code == self._to_code
        fc, tc  = self._from_code, self._to_code
        now     = time.monotonic()
        results: list[TranslationRegion] = []
        to_tx:  list[tuple[int, RawSegment]] = []

        # First pass: cache / filter
        for i, seg in enumerate(segments):
            if same or _numeric(seg.text) or _already_target(seg.text, tc):
                results.append(TranslationRegion(
                    bbox=seg.bbox, original_text=seg.text,
                    translated_text=seg.text, timestamp=now,
                ))
                continue

            h = _thash(seg.text)
            if h in self._cache:
                cached = self._cache[h]
                results.append(TranslationRegion(
                    bbox=seg.bbox,
                    original_text=cached.original_text,
                    translated_text=cached.translated_text,
                    timestamp=now,
                ))
                self.cache_hits += 1
            else:
                results.append(None)   # placeholder
                to_tx.append((i, seg))
                self.cache_misses += 1

        # Second pass: batch-translate misses
        if to_tx:
            joined = "\n".join(s.text for _, s in to_tx)
            t0     = time.perf_counter()
            try:
                translated = self._translator.translate(joined, fc, tc)
            except Exception as exc:
                log.exception("Translation error: %s", exc)
                translated = joined
            self.translate_ms = (time.perf_counter() - t0) * 1000
            log.debug("Batch translate %.1f ms", self.translate_ms)

            lines = translated.split("\n")
            for j, (i, seg) in enumerate(to_tx):
                tx  = lines[j].strip() if j < len(lines) else seg.text
                reg = TranslationRegion(
                    bbox=seg.bbox, original_text=seg.text,
                    translated_text=tx or seg.text, timestamp=now,
                )
                results[i] = reg
                self._cache[_thash(seg.text)] = reg

        # Fill any remaining None slots (shouldn't happen, safety net)
        final = []
        for i, r in enumerate(results):
            if r is None:
                s = segments[i]
                final.append(TranslationRegion(
                    bbox=s.bbox, original_text=s.text,
                    translated_text=s.text, timestamp=now,
                ))
            else:
                final.append(r)
        return final

    def _prune(self) -> None:
        cutoff = time.monotonic() - REGION_TTL
        self._cache = {k: v for k, v in self._cache.items() if v.timestamp > cutoff}
        if self.latest_regions:
            self.latest_regions = [r for r in self.latest_regions if r.timestamp > cutoff]
