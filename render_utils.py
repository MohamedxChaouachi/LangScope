"""
render_utils.py — Shared text-rendering helpers for LangScope.

Both LiveFeedWidget (ui.py) and OverlayWindow (overlay_window.py) need to draw
translated text into a bounding box with:
  - font auto-sized to the bbox height (optionally capped)
  - font shrunk until it fits the bbox width
  - word-wrapped multiline layout
  - a filled background clamped to the drawing surface bounds
  - vertically-centred lines

Keeping this logic in one place ensures the in-panel live feed and the
click-through desktop overlay stay visually consistent.

No project-level dependencies beyond PyQt6 — safe to import from any UI module.
"""

from __future__ import annotations

from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics

# Default appearance constants (previously duplicated inline in both widgets).
_BG_COLOR = QColor(10, 12, 30, 235)
_TEXT_COLOR = QColor("#E0F7FA")
_PADDING = 8  # horizontal padding budget when fitting text (px)
_TEXT_INSET = 4  # left/top inset of text inside the background (px)
_MIN_FONT = 6
_HEIGHT_RATIO = 0.72  # initial font size as a fraction of bbox height


def _wrap_lines(text: str, fm: QFontMetrics, max_width: int) -> list[str]:
    """Greedy word-wrap `text` so each line fits within `max_width` px."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if fm.horizontalAdvance(test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = [text]
    return lines


def draw_translated_text(
    painter: QPainter,
    text: str,
    x1: int,
    y1: int,
    bw: int,
    bh: int,
    surface_w: int,
    surface_h: int,
    *,
    max_font_pt: int | None = 24,
    font_family: str = "Segoe UI",
    bg_color: QColor = _BG_COLOR,
    text_color: QColor = _TEXT_COLOR,
) -> None:
    """
    Draw `text` into the box at (x1, y1, bw, bh), fitting and wrapping as needed.

    The font is sized to the box height (× _HEIGHT_RATIO), optionally capped at
    `max_font_pt`, then shrunk until the widest word/line fits the box width.
    Text is word-wrapped, the background is clamped to (surface_w, surface_h),
    and lines are vertically centred within the background block.

    Parameters
    ----------
    painter      : active QPainter on the target widget.
    text         : the translated string to render.
    x1, y1       : top-left corner of the box, in surface (widget) coords.
    bw, bh       : width / height of the box in px.
    surface_w/h  : dimensions of the drawing surface, used to clamp the
                   background so nothing renders off-window.
    max_font_pt  : optional upper bound on the font point size. Pass None for
                   no cap (matches the previous OverlayWindow behaviour).
    """
    # Auto-size font to bbox height, then optionally cap.
    font_pt = max(_MIN_FONT, int(bh * _HEIGHT_RATIO))
    if max_font_pt is not None:
        font_pt = min(max_font_pt, font_pt)

    font = QFont(font_family, font_pt, QFont.Weight.Bold)

    # Shrink font until the full text fits horizontally (with padding).
    while font_pt > _MIN_FONT:
        fm = QFontMetrics(font)
        if fm.horizontalAdvance(text) <= bw - _PADDING:
            break
        font_pt -= 1
        font.setPointSize(font_pt)

    fm = QFontMetrics(font)
    line_h = fm.height()

    # Word-wrap into lines that fit the box width.
    lines = _wrap_lines(text, fm, bw - _PADDING)
    needed_h = line_h * len(lines) + 4
    bg_h = max(bh, needed_h)

    # Clamp background to the drawing surface bounds.
    draw_x1 = max(0, x1)
    draw_y1 = max(0, y1)
    draw_x2 = min(surface_w, x1 + bw)
    draw_y2 = min(surface_h, y1 + bg_h)
    draw_bw = draw_x2 - draw_x1
    draw_bh = draw_y2 - draw_y1

    if draw_bw <= 0 or draw_bh <= 0:
        return

    painter.fillRect(draw_x1, draw_y1, draw_bw, draw_bh, bg_color)
    painter.setFont(font)
    painter.setPen(text_color)

    # Draw each line, vertically centred within the background block.
    top_offset = max(0, (draw_bh - needed_h) // 2)
    for li, line in enumerate(lines):
        ty = draw_y1 + top_offset + li * line_h + fm.ascent()
        if ty > draw_y2:
            break
        painter.drawText(draw_x1 + _TEXT_INSET, ty, line)
