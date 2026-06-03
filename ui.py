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

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication, QComboBox, QFrame,
    QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui  import (
    QFont, QFontMetrics, QPainter, QColor, QPen,
    QPixmap, QImage, QKeyEvent, QMouseEvent, QPaintEvent,
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
    update_frame() is called by the 60 FPS render timer in main.py.
    The detections argument is list[(bbox, original_text, translated_text)]
    which matches both the legacy engine.py format and the new pipeline.
    """

    _PLACEHOLDER = "📺  Select a region and press  ▶ Start\nto see the live preview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap:     QPixmap | None = None
        self._detections: list           = []
        self._orig_w:     int            = 1
        self._orig_h:     int            = 1

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

        for bbox, original, translated in self._detections:
            if not translated or translated == original:
                continue
            pts  = np.array(bbox, dtype=float)
            x1   = int(pts[:, 0].min() * sx) + px
            y1   = int(pts[:, 1].min() * sy) + py
            x2   = int(pts[:, 0].max() * sx) + px
            y2   = int(pts[:, 1].max() * sy) + py
            bw   = max(1, x2 - x1)
            bh   = max(1, y2 - y1)

            font_pt = max(6, int(bh * 0.72))
            font    = QFont("Segoe UI", font_pt, QFont.Weight.Bold)
            fm      = QFontMetrics(font)
            text_w  = fm.horizontalAdvance(translated)
            bg_w    = max(bw, text_w + 8)

            p.fillRect(x1, y1, bg_w, bh, QColor(10, 12, 30, 235))
            p.setFont(font)
            p.setPen(QColor("#E0F7FA"))
            p.drawText(
                QRect(x1 + 4, y1, bg_w - 4, bh),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                translated,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ControlPanel
# ─────────────────────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    start_requested       = pyqtSignal()
    stop_requested        = pyqtSignal()
    region_changed        = pyqtSignal(int, int, int, int)
    language_pair_changed = pyqtSignal(str, str)
    overlay_toggled       = pyqtSignal(bool)   # NEW

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.setWindowTitle("LangScope")
        self.setMinimumWidth(460)
        self.setMinimumHeight(420)
        self.resize(520, 680)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self._apply_styles()
        self._selector = RegionSelector()
        self._selector.region_selected.connect(self._on_region_selected)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(9)

        # Title
        title = QLabel("🌐 LangScope")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._sep())

        # FROM row
        root.addWidget(self._lbl("FROM language (source):"))
        from_row = QHBoxLayout(); from_row.setSpacing(8)
        self._from_combo = self._combo(_DEFAULT_FROM_IDX)
        from_row.addWidget(self._from_combo, stretch=1)
        self._from_badge = QLabel(""); self._from_badge.setObjectName("badge")
        self._from_badge.setFixedWidth(90)
        from_row.addWidget(self._from_badge)
        root.addLayout(from_row)

        # TO row
        root.addWidget(self._lbl("TO language (target):"))
        to_row = QHBoxLayout(); to_row.setSpacing(8)
        self._to_combo = self._combo(_DEFAULT_TO_IDX)
        to_row.addWidget(self._to_combo, stretch=1)
        self._to_badge = QLabel(""); self._to_badge.setObjectName("badge")
        self._to_badge.setFixedWidth(90)
        to_row.addWidget(self._to_badge)
        root.addLayout(to_row)

        self._from_combo.currentIndexChanged.connect(self._on_lang_changed)
        self._to_combo.currentIndexChanged.connect(self._on_lang_changed)

        # Region label
        self._region_label = QLabel("📍 No region selected")
        self._region_label.setObjectName("info")
        self._region_label.setWordWrap(True)
        root.addWidget(self._region_label)

        # Buttons row
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self._select_btn = QPushButton("🖱  Select Region")
        self._select_btn.setObjectName("secondary")
        self._select_btn.clicked.connect(self._open_selector)
        btn_row.addWidget(self._select_btn)
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setObjectName("primary")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._toggle)
        btn_row.addWidget(self._start_btn)
        root.addLayout(btn_row)

        # Overlay toggle
        overlay_row = QHBoxLayout()
        self._overlay_chk = QCheckBox("🖥  Desktop Overlay Mode")
        self._overlay_chk.setObjectName("overlay_chk")
        self._overlay_chk.toggled.connect(self.overlay_toggled.emit)
        overlay_row.addWidget(self._overlay_chk)
        overlay_row.addStretch()
        root.addLayout(overlay_row)

        root.addWidget(self._sep())

        # Live feed
        root.addWidget(self._lbl("📺  Live Feed  +  In-Place Translation"))
        self._feed = LiveFeedWidget()
        root.addWidget(self._feed, stretch=1)

        # Status bar row: status text + FPS counter
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
        self._region_label.setText(f"📍 ({left}, {top})  —  {w} x {h} px")
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
            self._start_btn.setText("⏹  Stop")
            self._start_btn.setObjectName("danger")
            self._select_btn.setEnabled(False)
            self._apply_styles()
            self.start_requested.emit()
        else:
            self._running = False
            self._start_btn.setText("▶  Start")
            self._start_btn.setObjectName("primary")
            self._select_btn.setEnabled(True)
            self._apply_styles()
            self.stop_requested.emit()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_live_feed(self, frame_bgr, detections: list,
                         orig_w: int, orig_h: int) -> None:
        self._feed.update_frame(frame_bgr, detections, orig_w, orig_h)

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
        self.setStyleSheet("""
            QWidget { background:#1A1A2E; color:#E0E0E0;
                      font-family:'Segoe UI',sans-serif; font-size:12px; }
            QLabel#title { color:#4FC3F7; font-size:15px; font-weight:bold; }
            QLabel#section_label { color:#90CAF9; font-size:11px;
                                   font-weight:bold; letter-spacing:0.5px; }
            QLabel#info { color:#90CAF9; font-size:11px; padding:5px 8px;
                          background:#16213E; border-radius:6px; }
            QLabel#badge  { color:#80CBC4; font-size:10px; font-style:italic; }
            QLabel#status { color:#B0BEC5; font-size:11px; font-style:italic; }
            QLabel#fps    { color:#4FC3F7; font-size:11px; font-weight:bold;
                            padding-left:8px; }
            QFrame#separator { color:#2A2A4A; max-height:1px; background:#2A2A4A; }
            QComboBox#combo { background:#16213E; color:#E0E0E0;
                              border:1px solid #2A2A6A; border-radius:6px;
                              padding:6px 10px; }
            QComboBox#combo:hover { border-color:#4FC3F7; }
            QComboBox#combo::drop-down { border:none; padding-right:8px; }
            QComboBox QAbstractItemView { background:#16213E; color:#E0E0E0;
                selection-background-color:#1565C0; border:1px solid #2A2A6A; }
            QCheckBox#overlay_chk { color:#90CAF9; font-size:11px; spacing:6px; }
            QCheckBox#overlay_chk::indicator { width:14px; height:14px; }
            QPushButton { border-radius:8px; padding:9px 14px;
                          font-size:12px; font-weight:bold; border:none; }
            QPushButton#primary  { background:#4CAF50; color:#FFF; }
            QPushButton#primary:hover   { background:#66BB6A; }
            QPushButton#primary:pressed { background:#388E3C; }
            QPushButton#primary:disabled { background:#37474F; color:#78909C; }
            QPushButton#secondary { background:#1565C0; color:#FFF; }
            QPushButton#secondary:hover   { background:#1976D2; }
            QPushButton#secondary:pressed { background:#0D47A1; }
            QPushButton#secondary:disabled { background:#37474F; color:#78909C; }
            QPushButton#danger  { background:#E53935; color:#FFF; }
            QPushButton#danger:hover   { background:#EF5350; }
            QPushButton#danger:pressed { background:#B71C1C; }
        """)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)
