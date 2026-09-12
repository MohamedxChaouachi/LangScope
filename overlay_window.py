"""
overlay_window.py — Optional click-through desktop overlay.

Paints translated text directly over the capture region on the desktop.
The window is:
  - Frameless / transparent background
  - Always-on-top
  - Click-through (Qt.WindowTransparentForInput)

Enable/disable via OverlayWindow.set_active(bool).
Update content via OverlayWindow.update_regions(regions, region_rect).
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore  import Qt
from PyQt6.QtGui   import QPainter, QPaintEvent, QKeyEvent
from PyQt6.QtWidgets import QApplication, QWidget

from models import TranslationRegion
from render_utils import draw_translated_text


class OverlayWindow(QWidget):
    """
    Transparent, always-on-top, click-through overlay window.

    Positioned exactly over the capture region so translated text appears
    in-place on the real desktop rather than inside the control panel.
    """

    def __init__(self, parent=None):
        super().__init__(parent,
            Qt.WindowType.FramelessWindowHint        |
            Qt.WindowType.WindowStaysOnTopHint       |
            Qt.WindowType.Tool                       |
            Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._regions: list[TranslationRegion] = []
        self._active   = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self.show()
            self.raise_()
        else:
            self.hide()

    def update_regions(
        self,
        regions:     list[TranslationRegion],
        capture_rect: dict,           # {"top", "left", "width", "height"}
    ) -> None:
        """Reposition window over capture region and update displayed regions."""
        if not self._active:
            return
        self.setGeometry(
            capture_rect["left"], capture_rect["top"],
            capture_rect["width"], capture_rect["height"],
        )
        self._regions = regions
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _: QPaintEvent) -> None:
        if not self._active or not self._regions:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        W, H = self.width(), self.height()

        for reg in self._regions:
            if not reg.has_translation:
                continue

            pts  = np.array(reg.bbox, dtype=float)
            x1   = int(pts[:, 0].min())
            y1   = int(pts[:, 1].min())
            x2   = int(pts[:, 0].max())
            y2   = int(pts[:, 1].max())
            bw   = max(1, x2 - x1)
            bh   = max(1, y2 - y1)

            # Shared bbox-clamped, word-wrapped, shrink-to-fit renderer keeps
            # the desktop overlay visually consistent with the live feed.
            draw_translated_text(
                p, reg.translated_text,
                x1, y1, bw, bh,
                W, H,
                max_font_pt=24,
            )

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.set_active(False)
        else:
            super().keyPressEvent(e)
