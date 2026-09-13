"""
ui.py — Live Translator UI

LiveFeedWidget renders:
  1. The live screen capture frame (updated at ~60 FPS by render timer).
  2. For each TranslationRegion: a filled background covering the original
     text, then the translated text drawn at a font size matched to bbox height.

ControlPanel adds:
  - Overlay Mode toggle checkbox
  - Live FPS counter in the status bar
"""

from __future__ import annotations

import numpy as np
import qtawesome as qta

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication, QComboBox, QFrame,
    QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal
from PyQt6.QtGui  import (
    QFont, QFontMetrics, QPainter, QColor, QPen,
    QPixmap, QImage, QKeyEvent, QMouseEvent, QPaintEvent, QIcon,
)

from render_utils import draw_translated_text
from theme import (
    COLOR_BG, COLOR_TEXT, COLOR_WHITE,
    COLOR_ACCENT, COLOR_ACCENT_SOFT, COLOR_SURFACE, COLOR_BADGE,
    COLOR_STATUS, COLOR_SEPARATOR, COLOR_BORDER,
    COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_PRIMARY_PRESSED,
    COLOR_SECONDARY, COLOR_SECONDARY_HOVER, COLOR_SECONDARY_PRESSED,
    COLOR_DANGER, COLOR_DANGER_HOVER, COLOR_DANGER_PRESSED,
    COLOR_DISABLED_BG, COLOR_DISABLED_TEXT,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG, SPACE_XL, SPACE_XXL,
    RADIUS_SM, RADIUS_MD,
)

# ─────────────────────────────────────────────────────────────────────────────
# Language catalogue
# ─────────────────────────────────────────────────────────────────────────────

ALL_LANGUAGES: list[tuple[str, str]] = [
    ("🇬🇧  English",    "en"),
    ("🇫🇷  French",     "fr"),
    ("🇪🇸  Spanish",    "es"),
    ("🇩🇪  German",     "de"),
    ("🇮🇹  Italian",    "it"),
    ("🇵🇹  Portuguese", "pt"),
    ("🇷🇴  Romanian",   "ro"),
    ("🇷🇺  Russian",    "ru"),
    ("🇨🇳  Chinese",    "zh"),
    ("🇯🇵  Japanese",   "ja"),
    ("🇦🇪  Arabic",     "ar"),
]

_DEFAULT_FROM_IDX = 0   # English
_DEFAULT_TO_IDX   = 1   # French


# ─────────────────────────────────────────────────────────────────────────────
# RegionSelector  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class RegionSelector(QWidget):
    region_selected = pyqtSignal(int, int, int, int)
    _DIM_ALPHA = 130

    def __init__(self, parent=None):
        super().__init__(parent)
        self._origin:  QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(QApplication.primaryScreen().geometry())

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.hide()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = self._current = e.pos()
            self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._origin:
            self._current = e.pos()
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self._origin:
            rect = self._rect()
            if rect.width() > 5 and rect.height() > 5:
                self.region_selected.emit(
                    rect.top(), rect.left(), rect.width(), rect.height()
                )
            self._origin = self._current = None
            self.hide()

    def paintEvent(self, _: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, self._DIM_ALPHA))
        if self._origin and self._current:
            r = self._rect()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(r, QColor(0, 0, 0, 255))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(QColor("#4FC3F7"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            p.setPen(QColor("#FFFFFF"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(r.left() + 4, r.top() - 6, f"{r.width()} x {r.height()} px")
        else:
            p.setPen(QColor("#FFFFFF"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Click and drag to select the capture region\n(ESC to cancel)",
            )

    def _rect(self) -> QRect:
        return QRect(self._origin, self._current).normalized()


# ─────────────────────────────────────────────────────────────────────────────
# LiveFeedWidget
# ─────────────────────────────────────────────────────────────────────────────

class LiveFeedWidget(QWidget):
    """
    Renders the latest captured frame plus translated-text overlays.

    Detections format: list[(bbox, original_text, translated_text, region_id?, confidence?)]
    The last two fields are optional — when absent, debug mode shows no metadata.

    Improvements over v1:
      - Uses smoothed_bbox coordinates (passed as bbox by render timer)
      - Proper multiline wrapping via Qt.TextFlag.TextWordWrap
      - Font auto-sized to fit the bbox height, then shrunk to fit width
      - Translated background clipped to widget bounds
      - Optional debug overlay: raw OCR bbox, confidence score, region ID
    """

    _PLACEHOLDER = "📺  Select a region and press  ▶ Start\nto see the live preview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap:      QPixmap | None = None
        self._detections:  list           = []
        self._orig_w:      int            = 1
        self._orig_h:      int            = 1
        self._debug_mode:  bool           = False

    def set_debug(self, enabled: bool) -> None:
        self._debug_mode = enabled
        self.update()

    def update_frame(
        self,
        frame_bgr:  np.ndarray,
        detections: list,
        orig_w:     int,
        orig_h:     int,
    ) -> None:
        """Called from the main thread every ~16 ms by the render timer."""
        self._detections = detections
        self._orig_w     = max(1, orig_w)
        self._orig_h     = max(1, orig_h)
        rgb  = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        qimg = QImage(rgb.data, orig_w, orig_h, orig_w * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def paintEvent(self, _: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        W, H = self.width(), self.height()

        if self._pixmap is None:
            p.fillRect(0, 0, W, H, QColor("#0A0A1A"))
            p.setPen(QColor("#3A3A6A"))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._PLACEHOLDER)
            return

        scaled = self._pixmap.scaled(
            W, H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        px = (W - scaled.width())  // 2
        py = (H - scaled.height()) // 2

        p.fillRect(0, 0, W, H, QColor("#0A0A1A"))
        p.drawPixmap(px, py, scaled)

        if not self._detections:
            return

        sx = scaled.width()  / self._orig_w
        sy = scaled.height() / self._orig_h

        for entry in self._detections:
            # Support both 3-tuple (legacy) and 5-tuple (debug) formats
            bbox, original, translated = entry[0], entry[1], entry[2]
            region_id  = entry[3] if len(entry) > 3 else ""
            confidence = entry[4] if len(entry) > 4 else None

            pts = np.array(bbox, dtype=float)
            x1  = int(pts[:, 0].min() * sx) + px
            y1  = int(pts[:, 1].min() * sy) + py
            x2  = int(pts[:, 0].max() * sx) + px
            y2  = int(pts[:, 1].max() * sy) + py
            bw  = max(1, x2 - x1)
            bh  = max(1, y2 - y1)

            # ── Debug overlay ─────────────────────────────────────────────────
            if self._debug_mode:
                # Raw OCR bounding box in cyan
                p.setPen(QPen(QColor(0, 220, 220, 200), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(x1, y1, bw, bh)
                # Confidence + region ID label
                dbg_font = QFont("Segoe UI", 7)
                p.setFont(dbg_font)
                p.setPen(QColor(0, 255, 200, 230))
                conf_str = f"{confidence:.2f}" if confidence is not None else "?"
                p.drawText(x1 + 2, y1 - 2, f"[{region_id}] {conf_str}")

            # ── Translation overlay ───────────────────────────────────────────
            if not translated or translated == original:
                continue

            # Shared bbox-clamped, word-wrapped, shrink-to-fit renderer.
            draw_translated_text(
                p, translated,
                x1, y1, bw, bh,
                W, H,
                max_font_pt=24,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ControlPanel
# ─────────────────────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    start_requested       = pyqtSignal()
    stop_requested        = pyqtSignal()
    region_changed        = pyqtSignal(int, int, int, int)
    language_pair_changed = pyqtSignal(str, str)
    overlay_toggled       = pyqtSignal(bool)
    debug_toggled         = pyqtSignal(bool)   # NEW

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.setWindowTitle("LangScope")
        self.setMinimumWidth(720)
        self.setMinimumHeight(420)
        self.resize(900, 620)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self._apply_styles()
        self._selector = RegionSelector()
        self._selector.region_selected.connect(self._on_region_selected)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(SPACE_SM)

        # ── Compact top toolbar ───────────────────────────────────────────────
        # A single framed toolbar holding two rows so the live feed gets the
        # rest of the vertical space by default.
        toolbar = QFrame(); toolbar.setObjectName("toolbar")
        tb = QVBoxLayout(toolbar)
        tb.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_SM)
        tb.setSpacing(SPACE_SM)

        # Row 1 — brand + language pickers
        row1 = QHBoxLayout(); row1.setSpacing(SPACE_MD)

        title_icon = QLabel()
        title_icon.setObjectName("title_icon")
        title_icon.setPixmap(
            qta.icon("fa5s.globe", color=COLOR_ACCENT).pixmap(QSize(20, 20))
        )
        row1.addWidget(title_icon)
        title = QLabel("LangScope")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setObjectName("title")
        row1.addWidget(title)

        row1.addSpacing(SPACE_MD)

        # FROM picker (label + combo + badge kept together)
        row1.addWidget(self._lbl("FROM"))
        self._from_combo = self._combo(_DEFAULT_FROM_IDX)
        row1.addWidget(self._from_combo, stretch=1)
        self._from_badge = QLabel(""); self._from_badge.setObjectName("badge")
        self._from_badge.setFixedWidth(70)
        row1.addWidget(self._from_badge)

        # arrow separator between the two pickers
        arrow = QLabel()
        arrow.setObjectName("title_icon")
        arrow.setPixmap(
            qta.icon("fa5s.arrow-right", color=COLOR_ACCENT_SOFT).pixmap(QSize(14, 14))
        )
        row1.addWidget(arrow)

        # TO picker
        row1.addWidget(self._lbl("TO"))
        self._to_combo = self._combo(_DEFAULT_TO_IDX)
        row1.addWidget(self._to_combo, stretch=1)
        self._to_badge = QLabel(""); self._to_badge.setObjectName("badge")
        self._to_badge.setFixedWidth(70)
        row1.addWidget(self._to_badge)

        tb.addLayout(row1)

        self._from_combo.currentIndexChanged.connect(self._on_lang_changed)
        self._to_combo.currentIndexChanged.connect(self._on_lang_changed)

        # Row 2 — actions + toggles + region info
        row2 = QHBoxLayout(); row2.setSpacing(SPACE_MD)

        self._select_btn = QPushButton("  Select Region")
        self._select_btn.setObjectName("secondary")
        self._select_btn.setIcon(qta.icon("fa5s.mouse-pointer", color=COLOR_WHITE))
        self._select_btn.setIconSize(QSize(16, 16))
        self._select_btn.clicked.connect(self._open_selector)
        row2.addWidget(self._select_btn)

        self._start_btn = QPushButton("  Start")
        self._start_btn.setObjectName("primary")
        self._start_btn.setIcon(qta.icon("fa5s.play", color=COLOR_WHITE))
        self._start_btn.setIconSize(QSize(16, 16))
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._toggle)
        row2.addWidget(self._start_btn)

        self._overlay_chk = QCheckBox("  Desktop Overlay")
        self._overlay_chk.setObjectName("overlay_chk")
        self._overlay_chk.setIcon(qta.icon("fa5s.desktop", color=COLOR_ACCENT_SOFT))
        self._overlay_chk.setIconSize(QSize(16, 16))
        self._overlay_chk.toggled.connect(self.overlay_toggled.emit)
        row2.addWidget(self._overlay_chk)

        self._debug_chk = QCheckBox("  Debug Mode")
        self._debug_chk.setObjectName("overlay_chk")   # reuse same style
        self._debug_chk.setIcon(qta.icon("fa5s.search", color=COLOR_ACCENT_SOFT))
        self._debug_chk.setIconSize(QSize(16, 16))
        self._debug_chk.toggled.connect(self.debug_toggled.emit)
        row2.addWidget(self._debug_chk)

        row2.addStretch()

        # Region info (icon + text) kept compact at the trailing edge
        self._region_icon = QLabel()
        self._region_icon.setObjectName("region_icon")
        self._region_icon.setPixmap(
            qta.icon("fa5s.map-marker-alt", color=COLOR_ACCENT_SOFT).pixmap(QSize(14, 14))
        )
        row2.addWidget(self._region_icon)
        self._region_label = QLabel("No region selected")
        self._region_label.setObjectName("info")
        self._region_label.setWordWrap(False)
        row2.addWidget(self._region_label)

        tb.addLayout(row2)

        root.addWidget(toolbar)

        # ── Live feed (gets all remaining vertical space) ─────────────────────
        self._feed = LiveFeedWidget()
        root.addWidget(self._feed, stretch=1)

        # ── Status bar row: status text + FPS counter ────────────────────────
        status_row = QHBoxLayout()
        self._status = QLabel("Ready — select a region to begin")
        self._status.setObjectName("status")
        status_row.addWidget(self._status, stretch=1)
        self._fps_label = QLabel("")
        self._fps_label.setObjectName("fps")
        status_row.addWidget(self._fps_label)
        root.addLayout(status_row)

        self.setLayout(root)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lbl(self, t: str) -> QLabel:
        l = QLabel(t); l.setObjectName("section_label"); return l

    def _sep(self) -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setObjectName("separator"); return f

    def _combo(self, idx: int) -> QComboBox:
        c = QComboBox(); c.setObjectName("combo")
        for name, code in ALL_LANGUAGES:
            c.addItem(name, userData=code)
        c.setCurrentIndex(idx)
        return c

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_selector(self) -> None:
        self._selector.show()
        self._selector.activateWindow()
        self._selector.raise_()

    def _on_region_selected(self, top: int, left: int, w: int, h: int) -> None:
        self._region_label.setText(f"({left}, {top})  —  {w} x {h} px")
        self._start_btn.setEnabled(True)
        self.region_changed.emit(top, left, w, h)
        self.activateWindow(); self.raise_()

    def _on_lang_changed(self, _: int) -> None:
        self.language_pair_changed.emit(
            self._from_combo.currentData(),
            self._to_combo.currentData(),
        )

    def _toggle(self) -> None:
        if not self._running:
            self._running = True
            self._start_btn.setText("  Stop")
            self._start_btn.setIcon(qta.icon("fa5s.stop", color=COLOR_WHITE))
            self._start_btn.setObjectName("danger")
            self._select_btn.setEnabled(False)
            self._apply_styles()
            self.start_requested.emit()
        else:
            self._running = False
            self._start_btn.setText("  Start")
            self._start_btn.setIcon(qta.icon("fa5s.play", color=COLOR_WHITE))
            self._start_btn.setObjectName("primary")
            self._select_btn.setEnabled(True)
            self._apply_styles()
            self.stop_requested.emit()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_live_feed(self, frame_bgr, detections: list,
                         orig_w: int, orig_h: int) -> None:
        self._feed.update_frame(frame_bgr, detections, orig_w, orig_h)

    def set_debug(self, enabled: bool) -> None:
        self._feed.set_debug(enabled)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_fps(self, fps: float) -> None:
        self._fps_label.setText(f"{fps:.0f} FPS")

    def set_from_badge(self, text: str) -> None:
        self._from_badge.setText(text)

    def set_to_badge(self, text: str) -> None:
        self._to_badge.setText(text)

    def selected_from_language(self) -> str:
        return self._from_combo.currentData()

    def selected_to_language(self) -> str:
        return self._to_combo.currentData()

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{COLOR_BG}; color:{COLOR_TEXT};
                      font-family:'Segoe UI',sans-serif; font-size:12px; }}
            QLabel#title {{ color:{COLOR_ACCENT}; font-size:15px; font-weight:bold; }}
            QLabel#section_label {{ color:{COLOR_ACCENT_SOFT}; font-size:11px;
                                   font-weight:bold; letter-spacing:0.5px; }}
            QLabel#info {{ color:{COLOR_ACCENT_SOFT}; font-size:11px; padding:{SPACE_XS}px {SPACE_MD}px;
                          background:{COLOR_SURFACE}; border-radius:{RADIUS_SM}px; }}
            QLabel#badge  {{ color:{COLOR_BADGE}; font-size:10px; font-style:italic; }}
            QLabel#status {{ color:{COLOR_STATUS}; font-size:11px; font-style:italic; }}
            QLabel#fps    {{ color:{COLOR_ACCENT}; font-size:11px; font-weight:bold;
                            padding-left:{SPACE_MD}px; }}
            QFrame#separator {{ color:{COLOR_SEPARATOR}; max-height:1px; background:{COLOR_SEPARATOR}; }}
            QFrame#toolbar {{ background:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER};
                             border-radius:{RADIUS_MD}px; }}
            QComboBox#combo {{ background:{COLOR_SURFACE}; color:{COLOR_TEXT};
                              border:1px solid {COLOR_BORDER}; border-radius:{RADIUS_SM}px;
                              padding:{SPACE_SM}px {SPACE_XL}px; }}
            QComboBox#combo:hover {{ border-color:{COLOR_ACCENT}; }}
            QComboBox#combo::drop-down {{ border:none; padding-right:{SPACE_MD}px; }}
            QComboBox QAbstractItemView {{ background:{COLOR_SURFACE}; color:{COLOR_TEXT};
                selection-background-color:{COLOR_SECONDARY}; border:1px solid {COLOR_BORDER}; }}
            QCheckBox#overlay_chk {{ color:{COLOR_ACCENT_SOFT}; font-size:11px; spacing:{SPACE_SM}px; }}
            QCheckBox#overlay_chk::indicator {{ width:{SPACE_XXL}px; height:{SPACE_XXL}px; }}
            QPushButton {{ border-radius:{RADIUS_MD}px; padding:{SPACE_LG}px {SPACE_XXL}px;
                          font-size:12px; font-weight:bold; border:none; }}
            QPushButton#primary  {{ background:{COLOR_PRIMARY}; color:{COLOR_WHITE}; }}
            QPushButton#primary:hover   {{ background:{COLOR_PRIMARY_HOVER}; }}
            QPushButton#primary:pressed {{ background:{COLOR_PRIMARY_PRESSED}; }}
            QPushButton#primary:disabled {{ background:{COLOR_DISABLED_BG}; color:{COLOR_DISABLED_TEXT}; }}
            QPushButton#secondary {{ background:{COLOR_SECONDARY}; color:{COLOR_WHITE}; }}
            QPushButton#secondary:hover   {{ background:{COLOR_SECONDARY_HOVER}; }}
            QPushButton#secondary:pressed {{ background:{COLOR_SECONDARY_PRESSED}; }}
            QPushButton#secondary:disabled {{ background:{COLOR_DISABLED_BG}; color:{COLOR_DISABLED_TEXT}; }}
            QPushButton#danger  {{ background:{COLOR_DANGER}; color:{COLOR_WHITE}; }}
            QPushButton#danger:hover   {{ background:{COLOR_DANGER_HOVER}; }}
            QPushButton#danger:pressed {{ background:{COLOR_DANGER_PRESSED}; }}
        """)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)
