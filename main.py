"""
main.py — Application Entry Point

Architecture:
  CaptureWorker (QThread)
    ├── EasyOCR       — text detection
    ├── ArgosTranslate — offline translation
    └── text_ready(raw, translated) ──► main thread ──► UI update

  PackageSetupThread (QThread)
    └── Triggered immediately when the user changes the language pair.
"""

import sys

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import QThread, pyqtSignal, QObject
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QApplication

from ui import ControlPanel
from engine import CaptureWorker
import translator


# ─────────────────────────────────────────────────────────────────────────────
# PackageSetupThread
# Runs translator.setup_translation_models() in the background whenever the
# user picks a new language pair — keeps the UI responsive during download.
# ─────────────────────────────────────────────────────────────────────────────

class PackageSetupThread(QThread):
    """
    One-shot thread: installs an Argos language pack, then emits `finished`.
    Instantiate a new one for each download request (QThreads are not reusable).
    """

    # (from_code, to_code, success)
    setup_finished = pyqtSignal(str, str, bool)

    def __init__(self, from_code: str, to_code: str, parent=None):
        super().__init__(parent)
        self._from_code = from_code
        self._to_code   = to_code

    def run(self) -> None:
        ok = translator.setup_translation_models(self._from_code, self._to_code)
        self.setup_finished.emit(self._from_code, self._to_code, ok)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)

    panel   = ControlPanel()
    panel.show()

    capture_worker = CaptureWorker()

    # Active setup threads — keep references so Qt doesn't garbage-collect them
    _setup_threads: list[PackageSetupThread] = []

    # ── Package setup logic ───────────────────────────────────────────────────

    def _update_badge(from_code: str, to_code: str, installed: bool) -> None:
        """Refresh the FROM/TO badges to show cache status."""
        icon = "✅ Cached" if installed else "⬇ Missing"
        # FROM badge: check whether any from_code→X pack exists
        from_ok = translator.is_package_installed(from_code, to_code)
        panel.set_from_badge("✅ Ready" if from_ok or from_code == to_code else "⬇ Pack needed")
        panel.set_to_badge("✅ Ready"   if from_ok or from_code == to_code else "")

    def _on_setup_finished(from_code: str, to_code: str, success: bool) -> None:
        """Called on the main thread when a PackageSetupThread completes."""
        if success:
            panel.set_status(f"✅ {from_code}→{to_code} ready")
            panel.set_from_badge("✅ Cached")
            panel.set_to_badge("✅ Cached")
        else:
            panel.set_status(f"❌ Download failed for {from_code}→{to_code}")
            panel.set_from_badge("❌ Failed")
            panel.set_to_badge("❌ Failed")

    def trigger_package_setup(from_code: str, to_code: str) -> None:
        """
        Start a background PackageSetupThread for *from_code* → *to_code*.
        Skips if already cached or if it's a same-language pair.
        """
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
        _setup_threads.append(t)   # keep alive

    # ── Language pair changed ──────────────────────────────────────────────────

    def on_language_pair_changed(from_code: str, to_code: str) -> None:
        """Called when either combo changes in the UI."""
        capture_worker.set_languages(from_code, to_code)
        trigger_package_setup(from_code, to_code)

    panel.language_pair_changed.connect(on_language_pair_changed)

    # ── Seed defaults ──────────────────────────────────────────────────────────
    # Apply the panel's initial selections to the worker, and trigger a download
    # for English→French (the default pair) so it's ready immediately.
    _init_from = panel.selected_from_language()
    _init_to   = panel.selected_to_language()
    capture_worker.set_languages(_init_from, _init_to)
    trigger_package_setup(_init_from, _init_to)   # en→fr on first launch

    # ── Region selector ────────────────────────────────────────────────────────

    def on_region_changed(top: int, left: int, w: int, h: int) -> None:
        capture_worker.set_region(top, left, w, h)

    panel.region_changed.connect(on_region_changed)

    # ── frame_ready -> UI ──────────────────────────────────────────────────────
    def on_frame_ready(frame, detections: list, orig_w: int, orig_h: int) -> None:
        """Runs on main thread — pass frame + per-region translations to feed."""
        panel.update_live_feed(frame, detections, orig_w, orig_h)
        if any(t for _, _, t in detections):
            panel.set_status("OK Translated")

    capture_worker.frame_ready.connect(on_frame_ready)

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def on_start() -> None:
        capture_worker._running = True
        if not capture_worker.isRunning():
            capture_worker.start()
        panel.set_status("⏳ Loading OCR & translation models…")

    def on_stop() -> None:
        capture_worker._running = False
        panel.set_status("⏹ Stopped")

    panel.start_requested.connect(on_start)
    panel.stop_requested.connect(on_stop)

    # ── Clean shutdown ─────────────────────────────────────────────────────────

    def cleanup() -> None:
        capture_worker.stop()
        for t in _setup_threads:
            t.quit()
            t.wait()

    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec())
