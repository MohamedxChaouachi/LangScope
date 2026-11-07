"""
ui.py — Live Translator UI

The control panel window and the widgets inside it.

LiveFeedWidget renders:
  1. The live screen capture frame (updated by the render timer).
  2. For each tracked region: a filled background covering the original text, then the
     translated text, drawn by render_utils.draw_translated_text.

RegionSelector lets the user drag out the capture region on any screen; its result is
emitted in native pixels (see screen_utils).

ControlPanel holds the language pickers, region selection, start/stop, the Desktop Overlay and
Debug Mode toggles, the status line with the FPS counter, and the debug stats strip.
"""

from __future__ import annotations

import os

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QGuiApplication,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QScreen,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from render_utils import draw_translated_text
from screen_utils import local_rect_to_native, screen_at_native_point, screen_index
from theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_SOFT,
    COLOR_BADGE,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_DANGER_HOVER,
    COLOR_DANGER_PRESSED,
    COLOR_DISABLED_BG,
    COLOR_DISABLED_TEXT,
    COLOR_FEED_BG,
    COLOR_FEED_PLACEHOLDER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_PRIMARY_PRESSED,
    COLOR_SECONDARY,
    COLOR_SECONDARY_HOVER,
    COLOR_SECONDARY_PRESSED,
    COLOR_SECONDARY_TEXT,
    COLOR_SELECTION,
    COLOR_SEPARATOR,
    COLOR_STATUS,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_WHITE,
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    FONT_SIZE_BASE,
    FONT_SIZE_LG,
    FONT_SIZE_SM,
    FONT_SIZE_TITLE,
    FONT_SIZE_XS,
    RADIUS_MD,
    RADIUS_SM,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    SPACE_XS,
    SPACE_XXL,
)

# Primary UI font family (first available), used where QFont is set directly.
_UI_FONT = "Segoe UI Variable Display"

# ─────────────────────────────────────────────────────────────────────────────
# Assets
# ─────────────────────────────────────────────────────────────────────────────

# Path to the app icon, resolved relative to this script so it keeps working
# from any working directory and when packaged (e.g. PyInstaller onedir).
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
APP_ICON_PATH = os.path.join(_ASSETS_DIR, "langscope.png")

# White checkmark used for the :checked checkbox indicator. QSS url() needs
# forward slashes, so normalise the path for the stylesheet.
_CHECK_ICON_PATH = os.path.join(_ASSETS_DIR, "check.png")
_CHECK_ICON_URL = _CHECK_ICON_PATH.replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
# Language catalogue
# ─────────────────────────────────────────────────────────────────────────────

ALL_LANGUAGES: list[tuple[str, str]] = [
    ("🇬🇧  English", "en"),
    ("🇫🇷  French", "fr"),
    ("🇪🇸  Spanish", "es"),
    ("🇩🇪  German", "de"),
    ("🇮🇹  Italian", "it"),
    ("🇵🇹  Portuguese", "pt"),
    ("🇷🇴  Romanian", "ro"),
    ("🇷🇺  Russian", "ru"),
    ("🇨🇳  Chinese", "zh"),
    ("🇯🇵  Japanese", "ja"),
    ("🇦🇪  Arabic", "ar"),
]

_DEFAULT_FROM_IDX = 0  # English
_DEFAULT_TO_IDX = 1  # French


# ─────────────────────────────────────────────────────────────────────────────
# RegionSelector  (multi-monitor)
# ─────────────────────────────────────────────────────────────────────────────


class _ScreenSelectorWindow(QWidget):
    """
    Full-screen dimmed drag-to-select window covering exactly one QScreen.

    Emits `region_picked(top, left, width, height)` in NATIVE desktop pixels
    (see screen_utils), so the result is valid on any monitor and DPI.
    """

    region_picked = pyqtSignal(int, int, int, int)
    cancelled = pyqtSignal()
    _DIM_ALPHA = 130

    def __init__(self, screen: QScreen, label: str):
        super().__init__()
        self._screen = screen
        self._label = label
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())

    def showEvent(self, e) -> None:
        # Re-apply after the native window exists so mixed-DPI setups land on
        # the right screen.
        self.setScreen(self._screen)
        self.setGeometry(self._screen.geometry())
        super().showEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = self._current = e.pos()
            self.update()
        elif e.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._origin:
            self._current = e.pos()
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self._origin:
            rect = self._rect()
            self._origin = self._current = None
            if rect.width() > 5 and rect.height() > 5:
                self.region_picked.emit(*local_rect_to_native(self._screen, rect))
            else:
                self.cancelled.emit()

    def paintEvent(self, _: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, self._DIM_ALPHA))
        if self._origin and self._current:
            r = self._rect()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(r, QColor(0, 0, 0, 255))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(QColor(COLOR_ACCENT), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            p.setPen(QColor("#FFFFFF"))
            p.setFont(QFont(_UI_FONT, 10))
            d = self._screen.devicePixelRatio()
            p.drawText(
                r.left() + 4, r.top() - 6, f"{round(r.width() * d)} x {round(r.height() * d)} px"
            )
        else:
            p.setPen(QColor("#FFFFFF"))
            p.setFont(QFont(_UI_FONT, 14))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                f"{self._label}\n"
                "Click and drag to select the capture region\n"
                "(ESC or right-click to cancel)",
            )

    def _rect(self) -> QRect:
        return QRect(self._origin, self._current).normalized()


class RegionSelector(QObject):
    """
    Opens a selector window on every connected screen so the capture region
    can be drawn on any monitor.

    Emits `region_selected(top, left, width, height)` in native virtual-desktop
    pixels (origin = top-left of the primary screen; may be negative for
    monitors placed left of / above the primary).
    """

    region_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._windows: list[_ScreenSelectorWindow] = []

    def show(self) -> None:
        self._close_windows()
        screens = QGuiApplication.screens()
        for i, screen in enumerate(screens, start=1):
            w = _ScreenSelectorWindow(screen, f"Screen {i} of {len(screens)}")
            w.region_picked.connect(self._on_picked)
            w.cancelled.connect(self._close_windows)
            self._windows.append(w)
            w.show()
        self._focus_window_under_cursor()

    # ControlPanel shows, raises and activates the selector as if it were a single window.
    def activateWindow(self) -> None:
        self._focus_window_under_cursor()

    def raise_(self) -> None:
        for w in self._windows:
            w.raise_()

    def hide(self) -> None:
        self._close_windows()

    def _focus_window_under_cursor(self) -> None:
        if not self._windows:
            return
        cur = QCursor.pos()
        target = next((w for w in self._windows if w.geometry().contains(cur)), self._windows[0])
        target.activateWindow()
        target.setFocus()

    def _on_picked(self, top: int, left: int, w: int, h: int) -> None:
        self._close_windows()
        self.region_selected.emit(top, left, w, h)

    def _close_windows(self) -> None:
        windows, self._windows = self._windows, []
        for w in windows:
            w.hide()
            w.deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
# LiveFeedWidget
# ─────────────────────────────────────────────────────────────────────────────


class LiveFeedWidget(QWidget):
    """
    Renders the latest captured frame plus translated-text overlays.

    Detections format: list[(bbox, original_text, translated_text, region_id?, confidence?)]
    The last two fields are optional — when absent, debug mode shows no metadata.

    The render timer passes each region's smoothed_bbox as the bbox. The translated text is
    drawn by render_utils.draw_translated_text: word-wrapped, with the font sized to the box
    height and then shrunk to fit its width, on a background clipped to the widget.
    Debug mode adds the raw OCR box, the confidence score and the region id.
    """

    _PLACEHOLDER = "📺  Select a region and press  ▶ Start\nto see the live preview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._detections: list = []
        self._orig_w: int = 1
        self._orig_h: int = 1
        self._debug_mode: bool = False

    def set_debug(self, enabled: bool) -> None:
        self._debug_mode = enabled
        self.update()

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        detections: list,
        orig_w: int,
        orig_h: int,
    ) -> None:
        """Called from the main thread every ~16 ms by the render timer."""
        self._detections = detections
        self._orig_w = max(1, orig_w)
        self._orig_h = max(1, orig_h)
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        qimg = QImage(rgb.data, orig_w, orig_h, orig_w * 3, QImage.Format.Format_RGB888)
        # fromImage copies the pixels into the pixmap, so the QImage (which only wraps the
        # numpy buffer) needs no copy of its own first.
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, _: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        W, H = self.width(), self.height()

        if self._pixmap is None:
            p.fillRect(0, 0, W, H, QColor(COLOR_FEED_BG))
            p.setPen(QColor(COLOR_FEED_PLACEHOLDER))
            p.setFont(QFont(_UI_FONT, 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._PLACEHOLDER)
            return

        scaled = self._pixmap.scaled(
            W,
            H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        px = (W - scaled.width()) // 2
        py = (H - scaled.height()) // 2

        p.fillRect(0, 0, W, H, QColor(COLOR_FEED_BG))
        p.drawPixmap(px, py, scaled)

        if not self._detections:
            return

        sx = scaled.width() / self._orig_w
        sy = scaled.height() / self._orig_h

        for entry in self._detections:
            # An entry has 3 fields; the region id and confidence (the last 2) are optional.
            bbox, original, translated = entry[0], entry[1], entry[2]
            region_id = entry[3] if len(entry) > 3 else ""
            confidence = entry[4] if len(entry) > 4 else None

            pts = np.array(bbox, dtype=float)
            x1 = int(pts[:, 0].min() * sx) + px
            y1 = int(pts[:, 1].min() * sy) + py
            x2 = int(pts[:, 0].max() * sx) + px
            y2 = int(pts[:, 1].max() * sy) + py
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

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
                p,
                translated,
                x1,
                y1,
                bw,
                bh,
                W,
                H,
                max_font_pt=24,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ControlPanel
# ─────────────────────────────────────────────────────────────────────────────


class ControlPanel(QWidget):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    region_changed = pyqtSignal(int, int, int, int)
    language_pair_changed = pyqtSignal(str, str)
    overlay_toggled = pyqtSignal(bool)
    debug_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.setWindowTitle("LangScope")
        if os.path.exists(APP_ICON_PATH):
            self.setWindowIcon(QIcon(APP_ICON_PATH))
        self.setMinimumWidth(760)
        self.setMinimumHeight(440)
        self.resize(940, 640)
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
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        tb = QVBoxLayout(toolbar)
        tb.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_SM)
        tb.setSpacing(SPACE_SM)

        # Row 1 — brand + language pickers
        row1 = QHBoxLayout()
        row1.setSpacing(SPACE_MD)

        title_icon = QLabel()
        title_icon.setObjectName("title_icon")
        if os.path.exists(APP_ICON_PATH):
            _logo = QPixmap(APP_ICON_PATH).scaled(
                QSize(28, 28),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            title_icon.setPixmap(_logo)
        row1.addWidget(title_icon)
        title = QLabel("LangScope")
        title.setFont(QFont(_UI_FONT, FONT_SIZE_TITLE, QFont.Weight.DemiBold))
        title.setObjectName("title")
        row1.addWidget(title)

        row1.addSpacing(SPACE_MD)

        # FROM picker (label + combo + badge kept together)
        row1.addWidget(self._lbl("FROM"))
        self._from_combo = self._combo(_DEFAULT_FROM_IDX)
        row1.addWidget(self._from_combo, stretch=1)
        self._from_badge = QLabel("")
        self._from_badge.setObjectName("badge")
        self._from_badge.setFixedWidth(70)
        row1.addWidget(self._from_badge)

        # arrow separator between the two pickers
        arrow = QLabel()
        arrow.setObjectName("title_icon")
        arrow.setPixmap(qta.icon("fa5s.arrow-right", color=COLOR_ACCENT_SOFT).pixmap(QSize(14, 14)))
        row1.addWidget(arrow)

        # TO picker
        row1.addWidget(self._lbl("TO"))
        self._to_combo = self._combo(_DEFAULT_TO_IDX)
        row1.addWidget(self._to_combo, stretch=1)
        self._to_badge = QLabel("")
        self._to_badge.setObjectName("badge")
        self._to_badge.setFixedWidth(70)
        row1.addWidget(self._to_badge)

        tb.addLayout(row1)

        self._from_combo.currentIndexChanged.connect(self._on_lang_changed)
        self._to_combo.currentIndexChanged.connect(self._on_lang_changed)

        # Row 2 — actions + toggles + region info
        row2 = QHBoxLayout()
        row2.setSpacing(SPACE_MD)

        self._select_btn = QPushButton("  Select Region")
        self._select_btn.setObjectName("secondary")
        self._select_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_btn.setIcon(qta.icon("fa5s.mouse-pointer", color=COLOR_SECONDARY_TEXT))
        self._select_btn.setIconSize(QSize(16, 16))
        self._select_btn.setMinimumWidth(150)
        self._select_btn.clicked.connect(self._open_selector)
        row2.addWidget(self._select_btn)

        self._start_btn = QPushButton("  Start")
        self._start_btn.setObjectName("primary")
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setIcon(qta.icon("fa5s.play", color=COLOR_WHITE))
        self._start_btn.setIconSize(QSize(16, 16))
        self._start_btn.setMinimumWidth(120)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._toggle)
        row2.addWidget(self._start_btn)

        self._overlay_chk = QCheckBox("  Desktop Overlay")
        self._overlay_chk.setObjectName("overlay_chk")
        self._overlay_chk.setCursor(Qt.CursorShape.PointingHandCursor)
        self._overlay_chk.setIcon(qta.icon("fa5s.desktop", color=COLOR_ACCENT_SOFT))
        self._overlay_chk.setIconSize(QSize(16, 16))
        self._overlay_chk.setMinimumWidth(150)
        self._overlay_chk.toggled.connect(self.overlay_toggled.emit)
        row2.addWidget(self._overlay_chk)

        self._debug_chk = QCheckBox("  Debug Mode")
        self._debug_chk.setObjectName("overlay_chk")  # reuse same style
        self._debug_chk.setCursor(Qt.CursorShape.PointingHandCursor)
        self._debug_chk.setIcon(qta.icon("fa5s.search", color=COLOR_ACCENT_SOFT))
        self._debug_chk.setIconSize(QSize(16, 16))
        self._debug_chk.setMinimumWidth(130)
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

        # ── Debug stats strip (hidden unless Debug Mode is on) ────────────────
        self._stats_label = QLabel("")
        self._stats_label.setObjectName("stats")
        self._stats_label.setTextFormat(Qt.TextFormat.PlainText)
        self._stats_label.setVisible(False)
        root.addWidget(self._stats_label)

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
        label = QLabel(t)
        label.setObjectName("section_label")
        return label

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setObjectName("separator")
        return f

    def _combo(self, idx: int) -> QComboBox:
        c = QComboBox()
        c.setObjectName("combo")
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
        screen = screen_at_native_point(left + w // 2, top + h // 2)
        n_screens = len(QGuiApplication.screens())
        where = f"Screen {screen_index(screen)}/{n_screens}  " if n_screens > 1 else ""
        self._region_label.setText(f"{where}({left}, {top})  —  {w} x {h} px")
        self._start_btn.setEnabled(True)
        self.region_changed.emit(top, left, w, h)
        self.activateWindow()
        self.raise_()

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

    def update_live_feed(self, frame_bgr, detections: list, orig_w: int, orig_h: int) -> None:
        self._feed.update_frame(frame_bgr, detections, orig_w, orig_h)

    def set_debug(self, enabled: bool) -> None:
        self._feed.set_debug(enabled)
        # Stats strip is only visible while Debug Mode is on.
        self._stats_label.setVisible(enabled)
        if not enabled:
            self._stats_label.clear()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_fps(self, fps: float) -> None:
        self._fps_label.setText(f"{fps:.0f} FPS")

    def set_stats(
        self,
        ocr_ms: float,
        preprocess_ms: float,
        inference_ms: float,
        translate_ms: float,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        """Update the debug stats strip. No-op display unless Debug Mode is on."""
        total = cache_hits + cache_misses
        hit_rate = (cache_hits / total * 100.0) if total else 0.0
        self._stats_label.setText(
            f"OCR {ocr_ms:.1f} ms  "
            f"(prep {preprocess_ms:.1f} / inf {inference_ms:.1f})   ·   "
            f"Translate {translate_ms:.1f} ms   ·   "
            f"Cache {hit_rate:.0f}%  ({cache_hits}/{total})"
        )

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
        # White tick drawn inside the checked indicator (if the asset exists).
        _check_img = f" image:url('{_CHECK_ICON_URL}');" if os.path.exists(_CHECK_ICON_PATH) else ""
        self.setStyleSheet(f"""
            /* ── Base ─────────────────────────────────────────────────────── */
            QWidget {{ background:{COLOR_WHITE}; color:{COLOR_TEXT};
                      font-family:{FONT_FAMILY}; font-size:{FONT_SIZE_BASE}px; }}

            /* ── Typography ───────────────────────────────────────────────── */
            QLabel#title {{ color:{COLOR_TEXT}; font-size:{FONT_SIZE_TITLE}px;
                           font-weight:600; }}
            QLabel#section_label {{ color:{COLOR_ACCENT_SOFT}; font-size:{FONT_SIZE_XS}px;
                                   font-weight:700; letter-spacing:1.2px; }}
            QLabel#info {{ color:{COLOR_STATUS}; font-size:{FONT_SIZE_SM}px; font-weight:500; }}
            QLabel#badge  {{ color:{COLOR_BADGE}; font-size:{FONT_SIZE_XS}px;
                            font-weight:700; letter-spacing:0.4px; }}
            QLabel#status {{ color:{COLOR_STATUS}; font-size:{FONT_SIZE_SM}px; font-weight:500; }}
            QLabel#fps    {{ color:{COLOR_ACCENT}; font-size:{FONT_SIZE_SM}px; font-weight:700;
                            letter-spacing:0.4px; padding-left:{SPACE_MD}px; }}
            QLabel#stats  {{ color:{COLOR_STATUS}; font-size:{FONT_SIZE_XS}px; font-weight:500;
                            font-family:{FONT_FAMILY_MONO}; letter-spacing:0.2px;
                            padding-top:{SPACE_XS}px; }}

            /* ── Containers ───────────────────────────────────────────────── */
            QFrame#separator {{ color:{COLOR_SEPARATOR}; max-height:1px; background:{COLOR_SEPARATOR}; }}
            QFrame#toolbar {{ background:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER};
                             border-radius:{RADIUS_MD}px; }}

            /* ── Combo boxes ──────────────────────────────────────────────── */
            QComboBox#combo {{ background:{COLOR_SURFACE}; color:{COLOR_TEXT};
                              border:1px solid {COLOR_BORDER}; border-radius:{RADIUS_SM}px;
                              padding:{SPACE_SM}px {SPACE_XL}px; min-height:18px; }}
            QComboBox#combo:hover {{ border-color:{COLOR_ACCENT}; }}
            QComboBox#combo:focus {{ border-color:{COLOR_ACCENT}; }}
            QComboBox#combo::drop-down {{ border:none; width:26px; padding-right:{SPACE_SM}px; }}
            QComboBox#combo::down-arrow {{ width:10px; height:10px; }}
            QComboBox QAbstractItemView {{ background:{COLOR_SURFACE}; color:{COLOR_TEXT};
                outline:none; padding:4px;
                selection-background-color:{COLOR_SELECTION};
                selection-color:{COLOR_TEXT};
                border:1px solid {COLOR_BORDER}; border-radius:{RADIUS_SM}px; }}
            QComboBox QAbstractItemView::item {{ padding:6px 10px; border-radius:{RADIUS_SM}px; }}

            /* ── Check boxes ──────────────────────────────────────────────── */
            QCheckBox#overlay_chk {{ color:{COLOR_ACCENT_SOFT}; font-size:{FONT_SIZE_SM}px;
                                    font-weight:500; spacing:{SPACE_SM}px; }}
            QCheckBox#overlay_chk::indicator {{ width:18px; height:18px;
                border:1.5px solid {COLOR_BORDER}; border-radius:5px;
                background:{COLOR_SURFACE}; }}
            QCheckBox#overlay_chk::indicator:hover {{ border-color:{COLOR_ACCENT}; }}
            QCheckBox#overlay_chk::indicator:checked {{
                background:{COLOR_ACCENT}; border-color:{COLOR_ACCENT};{_check_img} }}

            /* ── Buttons (slight rounding) ────────────────────────────────── */
            QPushButton {{ border-radius:{RADIUS_MD}px; padding:{SPACE_LG}px {SPACE_XXL}px;
                          font-size:{FONT_SIZE_LG}px; font-weight:600; border:none; }}
            QPushButton#primary  {{ background:{COLOR_PRIMARY}; color:{COLOR_WHITE}; }}
            QPushButton#primary:hover   {{ background:{COLOR_PRIMARY_HOVER}; }}
            QPushButton#primary:pressed {{ background:{COLOR_PRIMARY_PRESSED}; }}
            QPushButton#primary:disabled {{ background:{COLOR_DISABLED_BG}; color:{COLOR_DISABLED_TEXT}; }}
            QPushButton#secondary {{ background:{COLOR_SECONDARY}; color:{COLOR_SECONDARY_TEXT};
                                    border:1.5px solid {COLOR_BORDER}; }}
            QPushButton#secondary:hover   {{ background:{COLOR_SECONDARY_HOVER};
                                            border-color:{COLOR_ACCENT}; }}
            QPushButton#secondary:pressed {{ background:{COLOR_SECONDARY_PRESSED}; }}
            QPushButton#secondary:disabled {{ background:{COLOR_DISABLED_BG}; color:{COLOR_DISABLED_TEXT};
                                             border-color:{COLOR_BORDER}; }}
            QPushButton#danger  {{ background:{COLOR_DANGER}; color:{COLOR_WHITE}; }}
            QPushButton#danger:hover   {{ background:{COLOR_DANGER_HOVER}; }}
            QPushButton#danger:pressed {{ background:{COLOR_DANGER_PRESSED}; }}

            /* ── Scrollbars (slim, light) ─────────────────────────────────── */
            QScrollBar:vertical {{ background:transparent; width:10px; margin:0; }}
            QScrollBar::handle:vertical {{ background:{COLOR_BORDER}; border-radius:5px;
                                          min-height:30px; }}
            QScrollBar::handle:vertical:hover {{ background:{COLOR_ACCENT_SOFT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)
