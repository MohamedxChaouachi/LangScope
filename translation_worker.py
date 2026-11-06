"""
translation_worker.py — Translation worker thread.

Receives list[RawSegment] from OCRWorker via an internal queue.

New in this version:
  • Region tracking  — TranslationRegions persist across OCR cycles.
                       New detections are matched to existing regions by
                       bbox IoU + text similarity before creating new ones.
  • Bbox smoothing   — EMA applied on each cycle a region is seen.
  • TTL persistence  — regions stay alive for REGION_TTL seconds after
                       they stop being detected. miss_count tracks gaps.
  • Deduplication    — text-hash cache avoids re-translating identical text.
  • Filters          — numeric, timestamp, same-language guards (unchanged).

Region matching algorithm:
  For each new RawSegment:
    1. Find all live regions whose smoothed_bbox has IoU ≥ IOU_THRESHOLD with
       the segment's bbox.
    2. Among those candidates, pick the one with the highest text-similarity
       (difflib SequenceMatcher ratio).
    3. If best match ≥ TEXT_SIM_THRESHOLD → update that region in-place.
    4. Otherwise → create a new region.
  Unmatched live regions are not destroyed immediately; miss_count is
  incremented. Regions reaching MAX_MISSES are removed.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict, deque

from PyQt6.QtCore import QThread

from models import RawSegment, RegionSnapshot, TranslationRegion
from translation_providers import ArgosTranslator, BaseTranslator

log = logging.getLogger("langscope.translation_worker")

# ── Tuning constants ──────────────────────────────────────────────────────────
REGION_TTL = 8.0  # seconds before a missed region is pruned
MAX_MISSES = 5  # consecutive OCR cycles a missed region stays *tracked*
DISPLAY_GRACE_MISSES = 0  # OCR cycles a missed region stays *visible* (0 = clear
# at once; raise to 1-2 to ride out single-frame OCR misses
# at the cost of brief ghost translations)
IOU_THRESHOLD = 0.30  # minimum bbox overlap to consider a match
TEXT_SIM_THRESHOLD = 0.55  # minimum text similarity to accept a match
EMA_ALPHA = 0.35  # bbox smoothing strength (0=frozen, 1=no smoothing)
MAX_PENDING_BATCHES = 2  # OCR batches kept while the worker is busy; the oldest is dropped
TX_CACHE_MAX = 512  # translated texts remembered; the least recently used is dropped

# ── Text filters ──────────────────────────────────────────────────────────────
_MIN_ALPHA = 3
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


@functools.lru_cache(maxsize=4096)
def _detect_is_target(text: str, to_code: str) -> bool:
    try:
        from langdetect import detect

        return detect(text) == to_code
    except Exception:
        return False


def _already_target(text: str, to_code: str) -> bool:
    """True if *text* is already in the target language.

    langdetect costs ~20 ms per call and the same on-screen text is re-checked
    every OCR cycle, so results are memoised per (text, language)."""
    if len(text.strip()) < _MIN_DETECT:
        return False
    return _detect_is_target(text, to_code)


def _thash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ── Geometry helpers ──────────────────────────────────────────────────────────


def _bbox_rect(bbox: list) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) from a 4-point bbox.

    Plain-Python min/max: ~10x faster than building a NumPy array for 4 points,
    and this runs for every region pair on every OCR cycle."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _iou(bbox_a: list, bbox_b: list) -> float:
    """Intersection-over-Union of two 4-point bboxes."""
    ax1, ay1, ax2, ay2 = _bbox_rect(bbox_a)
    bx1, by1, bx2, by2 = _bbox_rect(bbox_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _text_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio (0–1)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ─────────────────────────────────────────────────────────────────────────────
# Mailbox
# ─────────────────────────────────────────────────────────────────────────────


class _Mailbox:
    """
    A thread-safe, ordered inbox for TranslationWorker.

    Messages come out in the order they went in. When too many "segments" batches are waiting
    the oldest batch is dropped, because the worker only needs the freshest view of the screen.
    Every other kind of message, such as a language change, is never dropped, so a slow
    translation can't make a control message disappear.
    """

    def __init__(self, max_batches: int):
        self._max_batches = max_batches
        self._messages: deque[tuple[str, object]] = deque()
        self._changed = threading.Condition()

    def post(self, kind: str, payload: object = None) -> None:
        with self._changed:
            self._messages.append((kind, payload))
            if kind == "segments":
                waiting = [m for m in self._messages if m[0] == "segments"]
                if len(waiting) > self._max_batches:
                    self._messages.remove(waiting[0])
            self._changed.notify()

    def get(self, timeout: float) -> tuple[str, object] | None:
        """The oldest message, waiting up to *timeout* seconds for one. None if there is none."""
        with self._changed:
            if not self._messages:
                self._changed.wait(timeout)
            return self._messages.popleft() if self._messages else None

    def has_pending(self, kind: str) -> bool:
        with self._changed:
            return any(m[0] == kind for m in self._messages)

    def drain(self) -> list[tuple[str, object]]:
        """Remove and return everything that is waiting, oldest first."""
        with self._changed:
            messages = list(self._messages)
            self._messages.clear()
            return messages


# ─────────────────────────────────────────────────────────────────────────────
# Translation cache
# ─────────────────────────────────────────────────────────────────────────────


class _TranslationCache:
    """
    A bounded text-hash → translated-text cache that drops the least recently used entry.

    OrderedDict keeps entries in use order, so a hit (move_to_end) and an eviction
    (popitem(last=False)) are both O(1). It belongs to the worker thread alone, so it needs
    no lock. The limit is read from TX_CACHE_MAX on each insert.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, str] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> str | None:
        """The cached translation, marking it as recently used; None if there isn't one."""
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, value: str) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > TX_CACHE_MAX:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


# ─────────────────────────────────────────────────────────────────────────────
# TranslationWorker
# ─────────────────────────────────────────────────────────────────────────────


class TranslationWorker(QThread):
    """
    QThread: matches, deduplicates, and translates OCR segments.

    Input:  enqueue(list[RawSegment])   — called from OCRWorker's signal.
    Output: latest_regions (polled by the render timer @ 60 FPS)

    Threading: while the thread runs, only that thread touches the tracking state (regions,
    translation cache, language pair). Other threads post messages to a mailbox, and the worker
    publishes `latest_regions` as a tuple of immutable RegionSnapshot objects that the render
    timer can read from any thread. While the thread is not running, set_languages() applies
    directly, since nothing else is touching that state.
    """

    def __init__(self, translator: BaseTranslator | None = None, parent=None):
        super().__init__(parent)
        self._translator: BaseTranslator = translator or ArgosTranslator()
        self._from_code = "en"
        self._to_code = "fr"
        self._mailbox = _Mailbox(MAX_PENDING_BATCHES)
        # Guards the question "is the worker thread consuming the mailbox?", so a language
        # change is either posted to a live thread or applied directly, never lost.
        self._lifecycle = threading.Lock()
        self._thread_active = False
        self._warmed_pair: tuple[str, str] | None = None

        # Active tracked regions — keyed by region_id
        self._regions: dict[str, TranslationRegion] = {}

        # text-hash → translated string cache (bounded, least recently used out first)
        self._tx_cache = _TranslationCache()

        # Latest snapshot for the render timer. Replaced as a whole (an atomic assignment) and
        # never modified, so readers on other threads need no lock.
        self.latest_regions: tuple[RegionSnapshot, ...] = ()

        self.translate_ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    # ── Public API (any thread) ───────────────────────────────────────────────

    def start(self, *args, **kwargs) -> None:
        # Marked active before the thread exists, so a set_languages() call that follows is
        # posted to it instead of being applied from the caller's thread.
        with self._lifecycle:
            self._thread_active = True
        super().start(*args, **kwargs)

    def set_languages(self, from_code: str, to_code: str) -> None:
        """Switch the language pair and forget everything tracked under the old one."""
        with self._lifecycle:
            if self._thread_active:
                self._mailbox.post("languages", (from_code, to_code))
                # Blank the display now; the worker resets its own state when it gets there.
                self.latest_regions = ()
                return
        self._apply_languages(from_code, to_code)

    def enqueue(self, segments: list[RawSegment]) -> None:
        """Non-blocking; if the worker is behind, the oldest waiting batch is dropped."""
        self._mailbox.post("segments", segments)

    def stop(self) -> None:
        self._mailbox.post("stop")
        self.wait()
        # Leave a clean slate for a restart: pending segments and tracked regions belong to
        # the old capture area, and the stop message is spent.
        self._mailbox.drain()
        self._regions.clear()
        self.latest_regions = ()

    # ── Worker loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.debug("TranslationWorker started")
        try:
            while True:
                self._warm_up_if_needed()
                message = self._mailbox.get(timeout=1.0)
                if message is None:
                    self._prune()
                    continue

                kind, payload = message
                if kind == "stop":
                    break
                if kind == "languages":
                    self._apply_languages(*payload)
                elif kind == "segments":
                    self._update(payload)
                    # A language change that is already waiting makes this result stale.
                    if not self._mailbox.has_pending("languages"):
                        self._publish()
        finally:
            # Apply any language change that arrived too late to be handled in the loop, then
            # hand control back, so set_languages() never posts to a thread that is gone.
            with self._lifecycle:
                for kind, payload in self._mailbox.drain():
                    if kind == "languages":
                        self._apply_languages(*payload)
                self._thread_active = False

    def _apply_languages(self, from_code: str, to_code: str) -> None:
        self._from_code = from_code
        self._to_code = to_code
        self._regions.clear()
        self._tx_cache.clear()
        self.latest_regions = ()
        try:
            self._translator.clear_cache()
        except Exception:
            log.exception("Clearing the translator cache failed")

    def _publish(self) -> None:
        """Publish an immutable snapshot of what the latest OCR pass actually saw.

        Regions it missed stay tracked in self._regions (so they re-match instantly if they
        reappear) but are not drawn: otherwise translations of text that has left the screen
        linger on top of the new content."""
        live = [r for r in self._regions.values() if r.miss_count <= DISPLAY_GRACE_MISSES]
        self.latest_regions = tuple(r.snapshot() for r in live)

    def _warm_up_if_needed(self) -> None:
        """Load the model for the current pair once (and again after a language
        change) so the first on-screen translation isn't delayed by model load."""
        pair = (self._from_code, self._to_code)
        if pair == self._warmed_pair or self._from_code == self._to_code:
            return
        try:
            _already_target("Loading the language detector profiles.", pair[1])
            if self._translator.warm_up(*pair):
                self._warmed_pair = pair
        except Exception as exc:
            log.warning("Warm-up failed: %s", exc)

    # ── Region update ─────────────────────────────────────────────────────────

    def _update(self, segments: list[RawSegment]) -> None:
        """
        Match each segment to an existing region or create a new one.
        Increment miss_count for unmatched live regions.
        """
        now = time.monotonic()
        same = self._from_code == self._to_code
        fc, tc = self._from_code, self._to_code

        # Track which existing regions were matched this cycle
        matched_ids: set[str] = set()

        # Collect segments that need translation (cache misses)
        to_translate: list[tuple[str, RawSegment, TranslationRegion | None]] = []

        for seg in segments:
            # Find best matching existing region (each region may be claimed by
            # at most one detection per cycle, otherwise two new lines would
            # overwrite each other and one of them would never be shown)
            existing = self._find_match(seg, exclude=matched_ids)

            if existing is not None:
                matched_ids.add(existing.region_id)
                existing.apply_smoothing(seg.bbox, EMA_ALPHA)
                existing.confidence = seg.score
                existing.last_seen = now
                existing.miss_count = 0

                # Text changed? Mark for re-translation
                if seg.text != existing.original_text:
                    existing.original_text = seg.text
                    existing.translated_text = seg.text  # interim passthrough
                    if not same and not _numeric(seg.text) and not _already_target(seg.text, tc):
                        to_translate.append((_thash(seg.text), seg, existing))
            else:
                # New region — create, schedule translation
                reg = TranslationRegion(
                    bbox=seg.bbox,
                    original_text=seg.text,
                    translated_text=seg.text,  # passthrough until translated
                    confidence=seg.score,
                    last_seen=now,
                )
                self._regions[reg.region_id] = reg
                matched_ids.add(reg.region_id)  # seen this cycle: not a miss

                if not same and not _numeric(seg.text) and not _already_target(seg.text, tc):
                    to_translate.append((_thash(seg.text), seg, reg))
                else:
                    reg.translated_text = seg.text

        # Increment miss_count for unmatched regions
        for rid, reg in self._regions.items():
            if rid not in matched_ids:
                reg.miss_count += 1

        # Batch-translate all cache misses
        if to_translate:
            self._batch_translate(to_translate, fc, tc)

    def _find_match(
        self,
        seg: RawSegment,
        exclude: set | None = None,
    ) -> TranslationRegion | None:
        """
        Return the best-matching live region for *seg*, or None.

        Candidates must have IoU ≥ IOU_THRESHOLD with seg.bbox (using their
        smoothed_bbox for stability). Among candidates, we pick the one with
        the highest text similarity. Returned only if ≥ TEXT_SIM_THRESHOLD.
        """
        best: TranslationRegion | None = None
        best_score: float = -1.0

        for reg in self._regions.values():
            if reg.miss_count > MAX_MISSES:
                continue
            if exclude and reg.region_id in exclude:
                continue
            iou = _iou(reg.smoothed_bbox, seg.bbox)
            if iou < IOU_THRESHOLD:
                continue
            sim = _text_sim(reg.original_text, seg.text)
            combined = 0.6 * iou + 0.4 * sim  # weighted score
            if combined > best_score:
                best_score = combined
                best = reg

        if best is not None and best_score >= (0.6 * IOU_THRESHOLD + 0.4 * TEXT_SIM_THRESHOLD):
            return best
        return None

    def _batch_translate(
        self,
        items: list[tuple[str, RawSegment, TranslationRegion]],
        fc: str,
        tc: str,
    ) -> None:
        """Translate all cache-miss texts in one Argos call."""
        # Deduplicate by hash so identical texts aren't sent twice
        unique: dict[str, list[TranslationRegion]] = {}
        for h, _seg, reg in items:
            cached = self._tx_cache.get(h)
            if cached is not None:
                reg.translated_text = cached
                self.cache_hits += 1
            else:
                unique.setdefault(h, [])
                unique[h].append(reg)
                self.cache_misses += 1

        if not unique:
            return

        # Gather one representative text per hash
        hash_order = list(unique.keys())
        texts_by_hash = {}
        for h, seg, _reg in items:
            if h in unique and h not in texts_by_hash:
                texts_by_hash[h] = seg.text

        texts = [texts_by_hash[h] for h in hash_order]
        t0 = time.perf_counter()
        try:
            # One batched model call; result is aligned 1:1 with `texts`
            # (no newline join/split, so multi-line OCR text can't misalign).
            translated = self._translator.translate_many(texts, fc, tc)
        except Exception as exc:
            log.exception("Translation error: %s", exc)
            translated = texts
        self.translate_ms = (time.perf_counter() - t0) * 1000
        log.debug("Batch translate %.1f ms (%d texts)", self.translate_ms, len(hash_order))

        for h, original, tx in zip(hash_order, texts, translated):
            tx = (tx or "").strip() or original
            self._tx_cache.put(h, tx)
            for reg in unique[h]:
                reg.translated_text = tx

    # ── Prune ─────────────────────────────────────────────────────────────────

    def _prune(self) -> None:
        """Remove regions that have exceeded MAX_MISSES or REGION_TTL."""
        cutoff = time.monotonic() - REGION_TTL
        # The TTL only applies to regions OCR has stopped seeing. A region that
        # was seen in the latest pass is still on screen even if the screen has
        # been static (no OCR cycles) for longer than REGION_TTL.
        dead = [
            rid
            for rid, reg in self._regions.items()
            if reg.miss_count > MAX_MISSES or (reg.miss_count > 0 and reg.last_seen < cutoff)
        ]
        for rid in dead:
            del self._regions[rid]
        if dead:
            log.debug("Pruned %d expired regions", len(dead))
