"""
controller.py — Wires the control panel to the capture, OCR and translation pipeline.

AppController owns the pipeline's lifecycle: it connects the panel's signals to the workers,
starts and stops them, and runs the 60 FPS render timer that moves the latest frame and regions
to the live feed and the desktop overlay.

Threads:
  CaptureManager (daemon thread)   — 60 FPS screen grab, writes latest_frame
  OCRWorker      (QThread)         — 8  FPS, reads latest_frame, emits raw_segments_ready
  TranslationWorker (QThread)      — on-demand, translates new/changed text
  QTimer @ 60 FPS                  — reads latest_frame + latest_regions → UI
  PackageSetupThread (daemon thread) — one-shot language-pack downloader

The controller gets its collaborators from its caller, so it can be tested with fakes;
build_controller() makes the real ones.
"""

from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from langscope.capture_manager import CaptureManager
from langscope.models import Region
from langscope.ocr_worker import OCRWorker
from langscope.overlay_window import OverlayWindow
from langscope.render_gate import RenderGate
from langscope.translation_providers import ArgosLanguagePacks, LanguagePacks
from langscope.translation_worker import TranslationWorker
from langscope.ui import ControlPanel

log = logging.getLogger("langscope.controller")

DEFAULT_REGION: Region = {"top": 100, "left": 100, "width": 500, "height": 200}
RENDER_INTERVAL_MS = 16  # ~60 FPS

# Argos pivots a pair through English when it has no direct pack (translator.PIVOT_LANGUAGE).
# Repeated here so the controller does not import Argos.
_PIVOT_LANGUAGE = "en"


def _pack_legs(from_code: str, to_code: str) -> set[tuple[str, str]]:
    """Every pack that installing from_code -> to_code may download: the direct one, and the two
    legs of the pivot through English."""
    legs = {(from_code, to_code)}
    if _PIVOT_LANGUAGE not in (from_code, to_code):
        legs |= {(from_code, _PIVOT_LANGUAGE), (_PIVOT_LANGUAGE, to_code)}
    return legs


class PackageSetupThread(QObject):
    """Installs the language packs for one pair, off the GUI thread.

    It runs on a daemon thread, not a QThread. An install is one long blocking download that
    cannot be interrupted, and a QThread that is still running when it is destroyed aborts the
    process. A daemon thread is simply dropped when the application exits, so closing the app
    never has to wait for a download. It offers the QThread calls the controller and tests use
    (start, isRunning, wait).
    """

    setup_finished = pyqtSignal(str, str, bool)

    def __init__(
        self, packs: LanguagePacks, from_code: str, to_code: str, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._packs = packs
        self.from_code = from_code
        self.to_code = to_code
        self._thread = threading.Thread(
            target=self._run, name=f"PackageSetup-{from_code}-{to_code}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def isRunning(self) -> bool:
        return self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the install ends (or *timeout* seconds pass). True if it has ended."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            ok = self._packs.install(self.from_code, self.to_code)
        except Exception:
            log.exception("Installing %s -> %s failed", self.from_code, self.to_code)
            ok = False
        try:
            self.setup_finished.emit(self.from_code, self.to_code, ok)
        except RuntimeError:
            pass  # the Qt side is already gone: the application is exiting


class AppController(QObject):
    def __init__(
        self,
        panel: ControlPanel,
        overlay: OverlayWindow,
        capture: CaptureManager,
        ocr_worker: OCRWorker,
        tx_worker: TranslationWorker,
        packs: LanguagePacks,
        render_gate: RenderGate | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._overlay = overlay
        self._capture = capture
        self._ocr_worker = ocr_worker
        self._tx_worker = tx_worker
        self._packs = packs
        self._render_gate = render_gate if render_gate is not None else RenderGate()
        self._region: Region = DEFAULT_REGION.copy()
        self._setup_threads: list[PackageSetupThread] = []
        # A pair that was asked for while an install it overlaps was running; see
        # _trigger_package_setup().
        self._deferred_pair: tuple[str, str] | None = None

        # OCR → TranslationWorker (cross-thread, via queue)
        ocr_worker.raw_segments_ready.connect(tx_worker.enqueue)
        # The OCR model loads on the worker thread; show its progress in the status bar.
        ocr_worker.status_changed.connect(panel.set_status)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_tick)

        panel.language_pair_changed.connect(self._on_language_pair_changed)
        # Start with the languages the panel shows.
        self._on_language_pair_changed(panel.selected_from_language(), panel.selected_to_language())

        panel.region_changed.connect(self._on_region_changed)
        panel.overlay_toggled.connect(self._on_overlay_toggled)
        panel.debug_toggled.connect(panel.set_debug)
        panel.start_requested.connect(self.start)
        panel.stop_requested.connect(self.stop)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def region(self) -> Region:
        """The capture region, in native desktop pixels."""
        return self._region

    def show(self) -> None:
        self._panel.show()

    def start(self) -> None:
        self._capture.start(self._region)
        if not self._ocr_worker.isRunning():
            self._ocr_worker.start()
        if not self._tx_worker.isRunning():
            self._tx_worker.start()
        self._render_timer.start()
        self._panel.set_status("⏳ Loading OCR & translation models…")

    def stop(self) -> None:
        self._halt_pipeline()
        self._panel.set_status("⏹ Stopped")

    def shutdown(self) -> None:
        """Stop everything for good.

        A language pack that is still downloading is not waited for: its daemon thread ends with
        the process, and its result is ignored."""
        self._halt_pipeline()
        for thread in self._setup_threads:
            thread.setup_finished.disconnect(self._on_setup_finished)
        self._setup_threads.clear()
        self._deferred_pair = None

    # ── Render timer ──────────────────────────────────────────────────────────

    def _render_tick(self) -> None:
        frame = self._capture.latest_frame
        regs = self._tx_worker.latest_regions
        tick = self._render_gate.tick(frame, regs)

        # Converting and copying a full frame is only worth doing for a frame (or a set of
        # regions) the UI hasn't been given yet. The gate only asks for a render when there
        # is a frame.
        if tick.render and frame is not None:
            h, w = frame.shape[:2]
            # Convert the region snapshots → 5-tuples for LiveFeedWidget
            # (bbox=smoothed, original, translated, region_id, confidence)
            detections = [
                (r.smoothed_bbox, r.original_text, r.translated_text, r.region_id, r.confidence)
                for r in regs
            ]
            self._panel.update_live_feed(frame, detections, w, h)

            # Update overlay if active
            self._overlay.update_regions(regs, self._region)

        if tick.fps is not None:
            self._panel.set_fps(tick.fps)

            # Debug stats — read straight off the workers, same as FPS above.
            # ControlPanel only shows the strip while Debug Mode is checked.
            self._panel.set_stats(
                ocr_ms=self._ocr_worker.avg_ocr_ms,
                preprocess_ms=self._ocr_worker.preprocess_ms,
                inference_ms=self._ocr_worker.inference_ms,
                translate_ms=self._tx_worker.translate_ms,
                cache_hits=self._tx_worker.cache_hits,
                cache_misses=self._tx_worker.cache_misses,
            )

    # ── Panel signals ─────────────────────────────────────────────────────────

    def _on_language_pair_changed(self, from_code: str, to_code: str) -> None:
        self._ocr_worker.set_languages(from_code, to_code)
        self._tx_worker.set_languages(from_code, to_code)
        self._trigger_package_setup(from_code, to_code)

    def _on_region_changed(self, top: int, left: int, w: int, h: int) -> None:
        self._region = {"top": top, "left": left, "width": w, "height": h}
        self._capture.set_region(self._region)

    def _on_overlay_toggled(self, active: bool) -> None:
        self._overlay.set_active(active)
        # A static screen produces no new frames, so without this the overlay would stay blank
        # until something changed.
        self._render_gate.invalidate()

    # ── Language pack setup ───────────────────────────────────────────────────

    def _trigger_package_setup(self, from_code: str, to_code: str) -> None:
        panel = self._panel
        self._deferred_pair = None  # whatever was waiting is no longer what the user wants
        if from_code == to_code:
            panel.set_from_badge("(same lang)")
            panel.set_to_badge("")
            return
        if self._packs.is_installed(from_code, to_code):
            panel.set_status(f"✅ {from_code}→{to_code} already cached")
            panel.set_from_badge("✅ Cached")
            panel.set_to_badge("✅ Cached")
            return
        panel.set_status(f"⬇ Downloading {from_code}→{to_code} pack…")
        panel.set_from_badge("⬇ Downloading…")
        panel.set_to_badge("⬇ Downloading…")
        # Two installs that need the same pack would download and write it at the same time.
        # Wait for the running one instead, then look at this pair again: it may have
        # installed what this one needs.
        wanted = _pack_legs(from_code, to_code)
        if any(wanted & _pack_legs(t.from_code, t.to_code) for t in self._setup_threads):
            self._deferred_pair = (from_code, to_code)
            return
        thread = PackageSetupThread(self._packs, from_code, to_code)
        thread.setup_finished.connect(self._on_setup_finished)
        thread.start()
        self._setup_threads.append(thread)

    def _on_setup_finished(self, from_code: str, to_code: str, success: bool) -> None:
        panel = self._panel
        self._setup_threads = [
            t for t in self._setup_threads if (t.from_code, t.to_code) != (from_code, to_code)
        ]
        if success:
            panel.set_status(f"✅ {from_code}→{to_code} ready")
            panel.set_from_badge("✅ Cached")
            panel.set_to_badge("✅ Cached")
            # The translator may have given up warming up while the pack was missing.
            self._tx_worker.retry_warm_up()
        else:
            panel.set_status(f"❌ Download failed for {from_code}→{to_code}")
            panel.set_from_badge("❌ Failed")
            panel.set_to_badge("❌ Failed")
        deferred, self._deferred_pair = self._deferred_pair, None
        if deferred is not None:
            self._trigger_package_setup(*deferred)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _halt_pipeline(self) -> None:
        self._render_timer.stop()
        self._capture.stop()
        self._ocr_worker.stop()
        self._tx_worker.stop()


def build_controller() -> AppController:
    """The real application: the control panel, the overlay and the whole pipeline."""
    capture = CaptureManager()
    return AppController(
        panel=ControlPanel(),
        overlay=OverlayWindow(),
        capture=capture,
        ocr_worker=OCRWorker(capture),
        tx_worker=TranslationWorker(),
        packs=ArgosLanguagePacks(),
    )
