"""
models.py — Shared data structures for LangScope.
No project-level dependencies — safe to import from any module.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class RawSegment:
    """A single OCR detection before translation."""

    bbox: list  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class RegionSnapshot:
    """
    An immutable copy of a TranslationRegion, as the render timer should draw it.

    TranslationWorker keeps changing its own TranslationRegion objects (position, text, miss
    count) while it tracks, so another thread must never hold one. It publishes these copies
    instead: nothing can change them, so they are safe to read from any thread.
    """

    region_id: str
    smoothed_bbox: tuple[tuple[float, float], ...]
    original_text: str
    translated_text: str
    confidence: float

    @property
    def has_translation(self) -> bool:
        return bool(self.translated_text) and self.translated_text != self.original_text


@dataclass
class TranslationRegion:
    """
    A text region tracked across OCR passes, with the data for smoothing and pruning.

    Fields:
        bbox              — the 4-point bbox of the first sighting; never updated.
        original_text     — the latest OCR text for this region.
        translated_text   — its translation; the original text until translated, and for text
                            that is not translated.
        timestamp         — monotonic time the region was created.
        region_id         — stable short id used for the debug overlay and matching.
        confidence        — OCR score of the best-matching detection.
        last_seen         — monotonic time of the most recent OCR sighting.
                            Separate from `timestamp` (creation time).
        smoothed_bbox     — exponentially smoothed bbox corners (4-pt list).
                            Initialised to bbox on first creation.
        miss_count        — how many consecutive OCR cycles this region was
                            not matched. Decides when the region is hidden, no longer
                            matched and pruned.
    """

    bbox: list
    original_text: str
    translated_text: str
    timestamp: float = field(default_factory=time.monotonic)
    region_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    confidence: float = 1.0
    last_seen: float = field(default_factory=time.monotonic)
    smoothed_bbox: list = field(default=None)  # set in __post_init__
    miss_count: int = 0

    def __post_init__(self) -> None:
        if self.smoothed_bbox is None:
            # Deep-copy bbox as initial smoothed position
            self.smoothed_bbox = [list(pt) for pt in self.bbox]

    @property
    def has_translation(self) -> bool:
        return bool(self.translated_text) and self.translated_text != self.original_text

    def snapshot(self) -> RegionSnapshot:
        """An immutable copy of what is drawn for this region right now."""
        return RegionSnapshot(
            region_id=self.region_id,
            smoothed_bbox=tuple((float(pt[0]), float(pt[1])) for pt in self.smoothed_bbox),
            original_text=self.original_text,
            translated_text=self.translated_text,
            confidence=self.confidence,
        )

    # ── Smoothing ──────────────────────────────────────────────────────────────

    def apply_smoothing(self, new_bbox: list, alpha: float = 0.35) -> None:
        """
        Exponential Moving Average on each corner coordinate.

        alpha = 0.35 means the new detection contributes 35 % of the final
        position, so rapid jitter is damped but genuine movement follows
        within 2–3 OCR cycles (~0.3 s at 8 FPS).

        Formula: smoothed = alpha * new + (1 - alpha) * old
        """
        for i, pt in enumerate(new_bbox):
            if i >= len(self.smoothed_bbox):
                self.smoothed_bbox.append(list(pt))
                continue
            sx = self.smoothed_bbox[i][0]
            sy = self.smoothed_bbox[i][1]
            self.smoothed_bbox[i][0] = alpha * pt[0] + (1.0 - alpha) * sx
            self.smoothed_bbox[i][1] = alpha * pt[1] + (1.0 - alpha) * sy
