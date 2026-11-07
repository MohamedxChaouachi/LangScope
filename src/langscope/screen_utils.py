"""
screen_utils.py — Multi-monitor / DPI coordinate helpers.

Two coordinate spaces are involved:

  * Logical  — Qt device-independent pixels (what widgets, mouse events and
               setGeometry() use).
  * Native   — physical pixels of the virtual desktop (what mss / dxcam use).

Qt keeps each QScreen's top-left in native coordinates and only divides the
*size* by devicePixelRatio, so for a screen `s` with geometry `g` and ratio `d`:

    native = g.topLeft() + (logical - g.topLeft()) * d
    logical = g.topLeft() + (native - g.topLeft()) / d

A region never spans two screens (it is drawn on one screen), so every
conversion is done relative to a single QScreen.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QGuiApplication, QScreen

from langscope.models import Region


def screen_native_rect(screen: QScreen) -> QRect:
    """Bounding rectangle of `screen` in native (physical) desktop pixels."""
    g = screen.geometry()
    d = screen.devicePixelRatio()
    return QRect(g.x(), g.y(), round(g.width() * d), round(g.height() * d))


def screen_at_native_point(x: int, y: int) -> QScreen:
    """Screen whose native rect contains (x, y); falls back to the primary."""
    for s in QGuiApplication.screens():
        if screen_native_rect(s).contains(x, y):
            return s
    primary = QGuiApplication.primaryScreen()
    if primary is None:  # Qt reports no primary screen when there are no screens at all
        raise RuntimeError("No screen is available")
    return primary


def screen_index(screen: QScreen) -> int:
    """1-based position of `screen` in QGuiApplication.screens(), or 0."""
    try:
        return QGuiApplication.screens().index(screen) + 1
    except ValueError:
        return 0


def local_rect_to_native(screen: QScreen, rect: QRect) -> tuple[int, int, int, int]:
    """
    Convert a rect in a screen-sized widget's local logical coordinates to
    native desktop pixels. Returns (top, left, width, height).
    """
    g = screen.geometry()
    d = screen.devicePixelRatio()
    left = g.x() + round(rect.left() * d)
    top = g.y() + round(rect.top() * d)
    width = max(1, round(rect.width() * d))
    height = max(1, round(rect.height() * d))
    return top, left, width, height


def native_region_to_logical(region: Region) -> tuple[QRect, float, QScreen]:
    """
    Convert a native-pixel Region to a logical QRect suitable for
    QWidget.setGeometry().

    Returns (logical_rect, device_pixel_ratio, screen).
    """
    cx = region["left"] + region["width"] // 2
    cy = region["top"] + region["height"] // 2
    screen = screen_at_native_point(cx, cy)
    g = screen.geometry()
    d = screen.devicePixelRatio() or 1.0
    rect = QRect(
        g.x() + round((region["left"] - g.x()) / d),
        g.y() + round((region["top"] - g.y()) / d),
        max(1, round(region["width"] / d)),
        max(1, round(region["height"] / d)),
    )
    return rect, d, screen
