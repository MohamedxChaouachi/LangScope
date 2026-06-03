"""
models.py — Shared data structures for LangScope.
No project-level dependencies — safe to import from any module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time


@dataclass
class RawSegment:
    """A single OCR detection before translation."""
    bbox:  list   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text:  str
    score: float


@dataclass
class TranslationRegion:
    """A fully processed detection: original text + its translation."""
    bbox:            list
    original_text:   str
    translated_text: str
    timestamp:       float = field(default_factory=time.monotonic)

    @property
    def has_translation(self) -> bool:
        return bool(self.translated_text) and self.translated_text != self.original_text
