"""
main.py — Application Entry Point

New threading model:
  CaptureManager (daemon thread)   — 60 FPS screen grab, writes latest_frame
  OCRWorker      (QThread)         — 8  FPS, reads latest_frame, emits raw_segments_ready
  TranslationWorker (QThread)      — on-demand, translates new/changed text
  QTimer @ 60 FPS                  — reads latest_frame + latest_regions → UI

  PackageSetupThread (QThread)     — one-shot Argos pack downloader (unchanged)
"""

import logging
import os
import sys

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

import translator
from capture_manager import CaptureManager
from ocr_worker import OCRWorker
from overlay_window import OverlayWindow
from render_gate import RenderGate
from translation_worker import TranslationWorker
from ui import APP_ICON_PATH, ControlPanel

# ─────────────────────────────────────────────────────────────────────────────
# PackageSetupThread  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────


class PackageSetupThread(QThread):
    setup_finished = pyqtSignal(str, str, bool)

    def __init__(self, from_code: str, to_code: str, parent=None):
        super().__init__(parent)
        self._from_code = from_code
        self._to_code = to_code

    def run(self) -> None:
        ok = translator.setup_translation_models(self._from_code, self._to_code)
        self.setup_finished.emit(self._from_code, self._to_code, ok)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Logging ───────────────────────────────────────────────────────────────
    # Level configurable via LANGSCOPE_LOGLEVEL (e.g. DEBUG, INFO, WARNING).
    # Defaults to INFO. This makes the logging calls throughout the pipeline
    # actually produce visible output.
    _loglevel_name = os.environ.get("LANGSCOPE_LOGLEVEL", "INFO").upper()
    _loglevel = getattr(logging, _loglevel_name, logging.INFO)
    logging.basicConfig(
        level=_loglevel,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)

    # App-wide window/taskbar icon (path is script-relative, packaging-safe).
    if os.path.exists(APP_ICON_PATH):
        app.setWindowIcon(QIcon(APP_ICON_PATH))

    # ── UI ────────────────────────────────────────────────────────────────────
    panel = ControlPanel()
    panel.show()

    overlay = OverlayWindow()

    # ── Pipeline workers ──────────────────────────────────────────────────────
    capture_mgr = CaptureManager()
    ocr_worker = OCRWorker(capture_mgr)
    tx_worker = TranslationWorker()

    # OCR → TranslationWorker (cross-thread, via queue)
    ocr_worker.raw_segments_ready.connect(tx_worker.enqueue)
    # The OCR model loads on the worker thread; show its progress in the status bar.
    ocr_worker.status_changed.connect(panel.set_status)

    # ── Render timer (60 FPS) ─────────────────────────────────────────────────
    _capture_region: dict = {"top": 100, "left": 100, "width": 500, "height": 200}
    render_gate = RenderGate()

    def _render_tick() -> None:
        frame = capture_mgr.latest_frame
        regs = tx_worker.latest_regions
        tick = render_gate.tick(frame, regs)

        # Converting and copying a full frame is only worth doing for a frame (or a set of
        # regions) the UI hasn't been given yet.
        if tick.render:
            h, w = frame.shape[:2]
            # Convert the region snapshots → 5-tuples for LiveFeedWidget
            # (bbox=smoothed, original, translated, region_id, confidence)
            detections = [
                (r.smoothed_bbox, r.original_text, r.translated_text, r.region_id, r.confidence)
                for r in regs
            ]
            panel.update_live_feed(frame, detections, w, h)

            # Update overlay if active
            overlay.update_regions(regs, _capture_region)

        if tick.fps is not None:
            panel.set_fps(tick.fps)

            # Debug stats — read straight off the workers, same as FPS above.
            # ControlPanel only shows the strip while Debug Mode is checked.
            panel.set_stats(
                ocr_ms=ocr_worker.avg_ocr_ms,
                preprocess_ms=ocr_worker.preprocess_ms,
                inference_ms=ocr_worker.inference_ms,
                translate_ms=tx_worker.translate_ms,
                cache_hits=tx_worker.cache_hits,
                cache_misses=tx_worker.cache_misses,
            )

    render_timer = QTimer()
    render_timer.setInterval(16)  # ~60 FPS
    render_timer.timeout.connect(_render_tick)

    # ── Package setup helpers (unchanged logic) ───────────────────────────────
    _setup_threads: list[PackageSetupThread] = []

    def _on_setup_finished(from_code: str, to_code: str, success: bool) -> None:
        if success:
            panel.set_status(f"✅ {from_code}→{to_code} ready")
            panel.set_from_badge("✅ Cached")
            panel.set_to_badge("✅ Cached")
        else:
            panel.set_status(f"❌ Download failed for {from_code}→{to_code}")
            panel.set_from_badge("❌ Failed")
            panel.set_to_badge("❌ Failed")

    def trigger_package_setup(from_code: str, to_code: str) -> None:
        if from_code == to_code:
            panel.set_from_badge("(same lang)")
            panel.set_to_badge("")
            return
        if translator.is_package_installed(from_code, to_code):
            panel.set_status(f"✅ {from_code}→{to_code} already cached")
            panel.set_from_badge("✅ Cached")
            panel.set_to_badge("✅ Cached")
            return
        panel.set_status(f"⬇ Downloading {from_code}→{to_code} pack…")
        panel.set_from_badge("⬇ Downloading…")
        panel.set_to_badge("⬇ Downloading…")
        t = PackageSetupThread(from_code, to_code)
        t.setup_finished.connect(_on_setup_finished)
        t.start()
        _setup_threads.append(t)

    # ── Language pair changed ─────────────────────────────────────────────────

    def on_language_pair_changed(from_code: str, to_code: str) -> None:
        ocr_worker.set_languages(from_code, to_code)
        tx_worker.set_languages(from_code, to_code)
        trigger_package_setup(from_code, to_code)

    panel.language_pair_changed.connect(on_language_pair_changed)

    # ── Seed defaults ─────────────────────────────────────────────────────────
    _init_from = panel.selected_from_language()
    _init_to = panel.selected_to_language()
    ocr_worker.set_languages(_init_from, _init_to)
    tx_worker.set_languages(_init_from, _init_to)
    trigger_package_setup(_init_from, _init_to)

    # ── Region selector ───────────────────────────────────────────────────────

    def on_region_changed(top: int, left: int, w: int, h: int) -> None:
        global _capture_region
        _capture_region = {"top": top, "left": left, "width": w, "height": h}
        capture_mgr.set_region(_capture_region)

    panel.region_changed.connect(on_region_changed)

    # ── Overlay toggle ────────────────────────────────────────────────────────

    def on_overlay_toggled(active: bool) -> None:
        overlay.set_active(active)
        # A static screen produces no new frames, so without this the overlay would stay blank
        # until something changed.
        render_gate.invalidate()

    panel.overlay_toggled.connect(on_overlay_toggled)

    def on_debug_toggled(active: bool) -> None:
        panel.set_debug(active)

    panel.debug_toggled.connect(on_debug_toggled)

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def on_start() -> None:
        capture_mgr.start(_capture_region)
        if not ocr_worker.isRunning():
            ocr_worker.start()
        if not tx_worker.isRunning():
            tx_worker.start()
        render_timer.start()
        panel.set_status("⏳ Loading OCR & translation models…")

    def on_stop() -> None:
        render_timer.stop()
        capture_mgr.stop()
        ocr_worker.stop()
        tx_worker.stop()
        panel.set_status("⏹ Stopped")

    panel.start_requested.connect(on_start)
    panel.stop_requested.connect(on_stop)

    # ── Clean shutdown ────────────────────────────────────────────────────────

    def cleanup() -> None:
        render_timer.stop()
        capture_mgr.stop()
        ocr_worker.stop()
        tx_worker.stop()
        for t in _setup_threads:
            t.quit()
            t.wait()

    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec())
