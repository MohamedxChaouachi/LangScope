"""Helpers shared by the widget tests."""

from __future__ import annotations

import sys
import traceback

import numpy as np
from PyQt6.QtGui import QImage


def bbox(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    """A 4-point bbox (clockwise from top-left), as EasyOCR returns them."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def render(widget) -> QImage:
    """Paint the widget into an image, failing the test with the traceback if painting raises.

    When paintEvent raises, PyQt reports the exception through sys.excepthook and Python also
    stores it in sys.last_*. That keeps the failed paintEvent's frame, and the QPainter still
    active in it, alive until after the render has finished and its paint device is gone, which
    makes Qt abort the whole test process ("Cannot destroy paint device that is being painted").
    So the hook formats the error and drops sys.last_*: the painter then ends while its device
    still exists, and the test fails normally with the original error.
    """
    problems: list[str] = []

    def record(exc_type, exc, tb):
        problems.append("".join(traceback.format_exception(exc_type, exc, tb)))
        for name in ("last_type", "last_value", "last_traceback", "last_exc"):
            if hasattr(sys, name):
                setattr(sys, name, None)

    previous_hook, sys.excepthook = sys.excepthook, record
    try:
        image = widget.grab().toImage()
    finally:
        sys.excepthook = previous_hook
    assert not problems, "paintEvent raised:\n" + "\n".join(problems)
    return image


def painted_columns(image: QImage) -> tuple[int, int] | None:
    """First and last x coordinate that has any non-transparent pixel, or None if blank."""
    argb = image.convertToFormat(QImage.Format.Format_ARGB32)
    pixels = np.frombuffer(argb.constBits().asarray(argb.sizeInBytes()), np.uint8)
    pixels = pixels.reshape(argb.height(), argb.bytesPerLine() // 4, 4)[:, : argb.width()]
    columns = np.flatnonzero((pixels[..., 3] > 0).any(axis=0))
    return (int(columns[0]), int(columns[-1])) if columns.size else None
