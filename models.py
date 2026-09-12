"""
models.py — Shared data structures for LangScope.
No project-level dependencies — safe to import from any module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import uuid


@dataclass
class RawSegment:
    """A single OCR detection before translation."""
    bbox:  list   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text:  str
    score: float


@dataclass
class TranslationRegion:
    """
    A fully processed detection with persistence and smoothing data.

    Fields added vs. v1:
        region_id         — stable UUID used for debug overlay and matching.
        confidence        — OCR score of the best-matching detection.
        last_seen         — monotonic time of the most recent OCR sighting.
                            Separate from `timestamp` (creation time).
        smoothed_bbox     — exponentially smoothed bbox corners (4-pt list).
                            Initialised to bbox on first creation.
        miss_count        — how many consecutive OCR cycles this region was
                            not matched. Used to decide when to prune.
    """
    bbox:            list
    original_text:   str
    translated_text: str
    timestamp:       float = field(default_factory=time.monotonic)
    region_id:       str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    confidence:      float = 1.0
    last_seen:       float = field(default_factory=time.monotonic)
    smoothed_bbox:   list  = field(default=None)          # set in __post_init__
    miss_count:      int   = 0

    def __post_init__(self) -> None:
        if self.smoothed_bbox is None:
            # Deep-copy bbox as initial smoothed position
            self.smoothed_bbox = [list(pt) for pt in self.bbox]

    @property
    def has_translation(self) -> bool:
        return bool(self.translated_text) and self.translated_text != self.original_text

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
