"""Helpers shared by the widget tests."""

from __future__ import annotations

import contextlib
import importlib
import sys
import traceback
from collections.abc import Iterator
from types import ModuleType

import numpy as np
from PyQt6.QtGui import QImage

_MISSING = object()


@contextlib.contextmanager
def fresh_import(name: str) -> Iterator[ModuleType]:
    """Import the module `name` anew, then put back whatever was there before.

    Both places that name a submodule are restored: sys.modules, and the attribute on its
    parent package. After `import a.b`, `from a import b` answers with that attribute, so a
    stale one would hand a module that is bound to fakes to the tests that run later."""
    parent_name, _, child = name.rpartition(".")
    parent = importlib.import_module(parent_name)
    old_module = sys.modules.pop(name, _MISSING)
    old_attr = getattr(parent, child, _MISSING)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        if old_module is not _MISSING:
            sys.modules[name] = old_module
        if hasattr(parent, child):
            delattr(parent, child)
        if old_attr is not _MISSING:
            setattr(parent, child, old_attr)


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
