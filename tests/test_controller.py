"""
AppController: the wiring between the control panel and the pipeline.

The panel is the real ControlPanel (offscreen), so the signal names and connections are
exercised. The workers, the capture, the overlay and the language packs are fakes that record
what the controller asks of them. A few tests read the render timer and the panel's labels
directly, since those are what the controller drives.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from controller import (
    DEFAULT_REGION,
    RENDER_INTERVAL_MS,
    AppController,
    PackageSetupThread,
)
from models import RegionSnapshot
from render_gate import RenderGate
from translation_providers import LanguagePacks
from ui import ControlPanel


class FakeCapture:
    def __init__(self):
        self.latest_frame = None
        self.calls = []

    def start(self, region):
        self.calls.append(("start", dict(region)))

    def stop(self):
        self.calls.append(("stop",))

    def set_region(self, region):
        self.calls.append(("set_region", dict(region)))


class FakeWorker(QObject):
    """What the controller uses of OCRWorker and TranslationWorker."""

    def __init__(self):
        super().__init__()
        self.running = False
        self.calls = []

    def isRunning(self):
        return self.running

    def start(self):
        self.calls.append(("start",))
        self.running = True

    def stop(self):
        self.calls.append(("stop",))
        self.running = False

    def set_languages(self, from_code, to_code):
        self.calls.append(("set_languages", from_code, to_code))


class FakeOcrWorker(FakeWorker):
    raw_segments_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    avg_ocr_ms = 11.5
    preprocess_ms = 3.5
    inference_ms = 8.0


class FakeTranslationWorker(FakeWorker):
    translate_ms = 4.5
    cache_hits = 7
    cache_misses = 2

    def __init__(self):
        super().__init__()
        self.latest_regions = ()
        self.enqueued = []

    def enqueue(self, segments):
        self.enqueued.append(segments)

    def retry_warm_up(self):
        self.calls.append(("retry_warm_up",))


class FakeOverlay:
    def __init__(self):
        self.calls = []

    def set_active(self, active):
        self.calls.append(("set_active", active))

    def update_regions(self, regions, rect):
        self.calls.append(("update_regions", tuple(regions), dict(rect)))


class FakePacks(LanguagePacks):
    def __init__(self):
        self.installed = set()
        self.install_result = True
        self.installs = []
        self.hold_install = None  # an Event the install waits for, when a test needs one

    def is_installed(self, from_code, to_code):
        return (from_code, to_code) in self.installed

    def install(self, from_code, to_code):
        self.installs.append((from_code, to_code))
        if self.hold_install is not None:
            self.hold_install.wait(5)
        return self.install_result


def snapshot(region_id="r1", x=0.0, original="hello", translated="bonjour"):
    return RegionSnapshot(
        region_id=region_id,
        smoothed_bbox=((x, 0.0), (x + 50, 0.0), (x + 50, 20.0), (x, 20.0)),
        original_text=original,
        translated_text=translated,
        confidence=0.9,
    )


@pytest.fixture
def parts(qtbot):
    panel = ControlPanel()
    qtbot.addWidget(panel)
    now = [100.0]
    packs = FakePacks()
    # The pair the panel starts with is installed, so building a controller starts no download.
    packs.installed.add((panel.selected_from_language(), panel.selected_to_language()))
    return SimpleNamespace(
        panel=panel,
        overlay=FakeOverlay(),
        capture=FakeCapture(),
        ocr=FakeOcrWorker(),
        tx=FakeTranslationWorker(),
        packs=packs,
        now=now,
        gate=RenderGate(clock=lambda: now[0]),
    )


@pytest.fixture
def build(parts):
    """Build a controller over `parts` (arrange the parts first), and always shut it down."""
    made = []

    def make():
        c = AppController(
            parts.panel,
            parts.overlay,
            parts.capture,
            parts.ocr,
            parts.tx,
            parts.packs,
            render_gate=parts.gate,
        )
        made.append(c)
        return c

    yield make
    for c in made:
        c.shutdown()


@pytest.fixture
def recorded(parts, monkeypatch):
    """Record what the controller sends to the panel's drawing methods."""
    calls = SimpleNamespace(feed=[], fps=[], stats=[])
    monkeypatch.setattr(
        parts.panel, "update_live_feed", lambda *args: calls.feed.append(args), raising=True
    )
    monkeypatch.setattr(parts.panel, "set_fps", calls.fps.append, raising=True)
    monkeypatch.setattr(parts.panel, "set_stats", lambda **kw: calls.stats.append(kw))
    return calls


def status(panel):
    return panel._status.text()


def badges(panel):
    return panel._from_badge.text(), panel._to_badge.text()


class TestStartup:
    def test_both_workers_start_with_the_languages_the_panel_shows(self, parts, build):
        pair = (parts.panel.selected_from_language(), parts.panel.selected_to_language())
        build()
        assert parts.ocr.calls == [("set_languages", *pair)]
        assert parts.tx.calls == [("set_languages", *pair)]

    def test_an_installed_pair_is_reported_as_cached(self, parts, build):
        from_code = parts.panel.selected_from_language()
        to_code = parts.panel.selected_to_language()
        build()
        assert status(parts.panel) == f"✅ {from_code}→{to_code} already cached"
        assert badges(parts.panel) == ("✅ Cached", "✅ Cached")
        assert parts.packs.installs == []

    def test_a_pair_that_is_not_installed_is_downloaded_at_startup(self, parts, build, qtbot):
        pair = (parts.panel.selected_from_language(), parts.panel.selected_to_language())
        parts.packs.installed.clear()
        build()
        qtbot.waitUntil(lambda: status(parts.panel).endswith("ready"), timeout=5000)
        assert parts.packs.installs == [pair]

    def test_nothing_runs_until_start(self, parts, build):
        c = build()
        assert parts.capture.calls == []
        assert ("start",) not in parts.ocr.calls
        assert ("start",) not in parts.tx.calls
        assert not c._render_timer.isActive()

    def test_the_render_timer_ticks_at_about_60_fps(self, build):
        assert RENDER_INTERVAL_MS == 16
        assert build()._render_timer.interval() == 16

    def test_the_capture_region_starts_at_the_default(self, build):
        c = build()
        assert c.region == DEFAULT_REGION
        assert c.region is not DEFAULT_REGION  # a region change must not edit the default

    def test_the_panel_is_shown_on_request(self, build, parts):
        c = build()
        assert not parts.panel.isVisible()
        c.show()
        assert parts.panel.isVisible()


class TestSignalsFromThePanel:
    def test_ocr_results_are_queued_for_translation(self, parts, build):
        build()
        parts.ocr.raw_segments_ready.emit(["segment"])
        assert parts.tx.enqueued == [["segment"]]

    def test_the_ocr_status_is_shown_in_the_status_bar(self, parts, build):
        build()
        parts.ocr.status_changed.emit("OCR model ready")
        assert status(parts.panel) == "OCR model ready"

    def test_a_new_region_is_sent_to_the_capture_and_remembered(self, parts, build):
        c = build()
        parts.panel.region_changed.emit(10, 20, 300, 100)
        expected = {"top": 10, "left": 20, "width": 300, "height": 100}
        assert parts.capture.calls == [("set_region", expected)]
        assert c.region == expected

    def test_toggling_the_overlay_switches_it(self, parts, build):
        build()
        parts.panel.overlay_toggled.emit(True)
        parts.panel.overlay_toggled.emit(False)
        assert parts.overlay.calls == [("set_active", True), ("set_active", False)]

    def test_toggling_debug_mode_reaches_the_panel(self, parts, build, monkeypatch):
        seen = []
        monkeypatch.setattr(parts.panel, "set_debug", seen.append)  # before the controller binds it
        build()
        parts.panel.debug_toggled.emit(True)
        parts.panel.debug_toggled.emit(False)
        assert seen == [True, False]

    def test_the_start_button_starts_and_the_stop_button_stops(self, parts, build):
        build()
        parts.panel.start_requested.emit()
        assert parts.capture.calls == [("start", DEFAULT_REGION)]
        parts.panel.stop_requested.emit()
        assert parts.capture.calls[-1] == ("stop",)


class TestLanguagePair:
    def test_a_new_pair_goes_to_both_workers(self, parts, build):
        build()
        parts.packs.installed.add(("fr", "en"))
        parts.panel.language_pair_changed.emit("fr", "en")
        assert parts.ocr.calls[-1] == ("set_languages", "fr", "en")
        assert parts.tx.calls[-1] == ("set_languages", "fr", "en")

    def test_an_installed_pair_is_reported_as_cached(self, parts, build):
        build()
        parts.packs.installed.add(("fr", "en"))
        parts.panel.language_pair_changed.emit("fr", "en")
        assert status(parts.panel) == "✅ fr→en already cached"
        assert badges(parts.panel) == ("✅ Cached", "✅ Cached")
        assert parts.packs.installs == []

    def test_the_same_language_on_both_sides_needs_no_pack(self, parts, build):
        build()
        parts.packs.installed.clear()
        before = status(parts.panel)
        parts.panel.language_pair_changed.emit("de", "de")
        assert badges(parts.panel) == ("(same lang)", "")
        assert status(parts.panel) == before
        assert parts.packs.installs == []
        assert parts.ocr.calls[-1] == ("set_languages", "de", "de")

    def test_a_missing_pair_is_downloaded_and_reported(self, parts, build, qtbot):
        parts.packs.hold_install = threading.Event()
        build()
        parts.panel.language_pair_changed.emit("ja", "en")
        assert status(parts.panel) == "⬇ Downloading ja→en pack…"
        assert badges(parts.panel) == ("⬇ Downloading…", "⬇ Downloading…")
        parts.packs.hold_install.set()
        qtbot.waitUntil(lambda: status(parts.panel) == "✅ ja→en ready", timeout=5000)
        assert badges(parts.panel) == ("✅ Cached", "✅ Cached")
        assert parts.packs.installs == [("ja", "en")]

    def test_a_finished_download_asks_the_translator_to_try_its_warm_up_again(
        self, parts, build, qtbot
    ):
        build()
        parts.panel.language_pair_changed.emit("ja", "en")
        qtbot.waitUntil(lambda: ("retry_warm_up",) in parts.tx.calls, timeout=5000)

    def test_a_failed_download_is_reported_and_does_not_retry_the_warm_up(
        self, parts, build, qtbot
    ):
        parts.packs.install_result = False
        build()
        parts.panel.language_pair_changed.emit("zz", "en")
        qtbot.waitUntil(lambda: "Download failed" in status(parts.panel), timeout=5000)
        assert status(parts.panel) == "❌ Download failed for zz→en"
        assert badges(parts.panel) == ("❌ Failed", "❌ Failed")
        assert ("retry_warm_up",) not in parts.tx.calls


class TestStartStop:
    def test_start_launches_the_capture_the_workers_and_the_render_timer(self, parts, build):
        c = build()
        c.start()
        assert parts.capture.calls == [("start", DEFAULT_REGION)]
        assert ("start",) in parts.ocr.calls
        assert ("start",) in parts.tx.calls
        assert c._render_timer.isActive()
        assert status(parts.panel) == "⏳ Loading OCR & translation models…"

    def test_start_uses_the_region_the_user_picked(self, parts, build):
        c = build()
        parts.panel.region_changed.emit(10, 20, 300, 100)
        c.start()
        assert parts.capture.calls[-1] == (
            "start",
            {"top": 10, "left": 20, "width": 300, "height": 100},
        )

    def test_running_workers_are_not_started_again(self, parts, build):
        c = build()
        c.start()
        c.start()
        assert parts.ocr.calls.count(("start",)) == 1
        assert parts.tx.calls.count(("start",)) == 1

    def test_stop_halts_everything_and_says_so(self, parts, build):
        c = build()
        c.start()
        c.stop()
        assert not c._render_timer.isActive()
        assert parts.capture.calls[-1] == ("stop",)
        assert parts.ocr.calls[-1] == ("stop",)
        assert parts.tx.calls[-1] == ("stop",)
        assert status(parts.panel) == "⏹ Stopped"

    def test_the_pipeline_can_be_started_again_after_a_stop(self, parts, build):
        c = build()
        c.start()
        c.stop()
        c.start()
        assert parts.ocr.calls.count(("start",)) == 2
        assert c._render_timer.isActive()

    def test_shutdown_halts_everything_without_changing_the_status(self, parts, build):
        c = build()
        c.start()
        parts.panel.set_status("Running")
        c.shutdown()
        assert not c._render_timer.isActive()
        assert parts.capture.calls[-1] == ("stop",)
        assert parts.ocr.calls[-1] == ("stop",)
        assert parts.tx.calls[-1] == ("stop",)
        assert status(parts.panel) == "Running"

    def test_shutdown_waits_for_a_language_pack_that_is_still_installing(self, parts, build, qtbot):
        parts.packs.hold_install = threading.Event()
        c = build()
        parts.panel.language_pair_changed.emit("ja", "en")
        thread = c._setup_threads[0]
        assert thread.isRunning()
        threading.Timer(0.2, parts.packs.hold_install.set).start()
        c.shutdown()  # blocks until the install returns
        assert thread.isFinished()


class TestRenderTick:
    def test_nothing_is_drawn_before_the_first_frame(self, parts, build, recorded):
        c = build()
        parts.tx.latest_regions = (snapshot(),)
        c._render_tick()
        assert recorded.feed == []
        assert parts.overlay.calls == []

    def test_a_frame_is_drawn_with_its_regions_in_the_live_feed_and_the_overlay(
        self, parts, build, recorded
    ):
        c = build()
        frame = np.zeros((40, 80, 3), np.uint8)
        region = snapshot("r1", x=5.0)
        parts.capture.latest_frame = frame
        parts.tx.latest_regions = (region,)
        c._render_tick()

        (args,) = recorded.feed
        assert args[0] is frame
        assert args[1] == [(region.smoothed_bbox, "hello", "bonjour", "r1", 0.9)]
        assert args[2:] == (80, 40)  # width, then height
        assert parts.overlay.calls == [("update_regions", (region,), DEFAULT_REGION)]

    def test_an_unchanged_frame_and_regions_are_not_drawn_again(self, parts, build, recorded):
        c = build()
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        parts.tx.latest_regions = (snapshot(),)
        c._render_tick()
        c._render_tick()
        c._render_tick()
        assert len(recorded.feed) == 1
        assert len(parts.overlay.calls) == 1

    def test_a_new_frame_is_drawn(self, parts, build, recorded):
        c = build()
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        c._render_tick()
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        c._render_tick()
        assert len(recorded.feed) == 2

    def test_new_regions_on_the_same_frame_are_drawn(self, parts, build, recorded):
        c = build()
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        parts.tx.latest_regions = (snapshot("a"),)
        c._render_tick()
        parts.tx.latest_regions = (snapshot("a"), snapshot("b", x=60.0))
        c._render_tick()
        assert len(recorded.feed) == 2
        assert [d[3] for d in recorded.feed[1][1]] == ["a", "b"]

    def test_the_overlay_follows_the_region_the_user_picked(self, parts, build, recorded):
        c = build()
        parts.panel.region_changed.emit(10, 20, 300, 100)
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        c._render_tick()
        assert parts.overlay.calls[-1][2] == {"top": 10, "left": 20, "width": 300, "height": 100}

    def test_switching_the_overlay_on_redraws_a_static_screen(self, parts, build, recorded):
        c = build()
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        parts.tx.latest_regions = (snapshot(),)
        c._render_tick()
        c._render_tick()
        assert len(parts.overlay.calls) == 1
        parts.panel.overlay_toggled.emit(True)  # a set_active call is recorded too
        c._render_tick()
        assert [call[0] for call in parts.overlay.calls] == [
            "update_regions",
            "set_active",
            "update_regions",
        ]

    def test_the_fps_and_the_debug_stats_are_sent_once_per_second(self, parts, build, recorded):
        c = build()
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        c._render_tick()  # t = 100.0: the window starts, and this frame is drawn
        assert recorded.fps == [] and recorded.stats == []

        parts.now[0] = 100.5
        parts.capture.latest_frame = np.zeros((40, 80, 3), np.uint8)
        c._render_tick()  # drawn, still inside the window
        assert recorded.fps == []

        parts.now[0] = 101.0
        c._render_tick()  # the window closes: 2 frames drawn in 1 second
        assert recorded.fps == [pytest.approx(2.0)]
        assert recorded.stats == [
            {
                "ocr_ms": 11.5,
                "preprocess_ms": 3.5,
                "inference_ms": 8.0,
                "translate_ms": 4.5,
                "cache_hits": 7,
                "cache_misses": 2,
            }
        ]


class TestPackageSetupThread:
    def test_it_installs_the_pair_and_reports_the_outcome(self, qtbot):
        packs = FakePacks()
        thread = PackageSetupThread(packs, "ja", "en")
        with qtbot.waitSignal(thread.setup_finished, timeout=5000) as outcome:
            thread.start()
        thread.wait()
        assert outcome.args == ["ja", "en", True]
        assert packs.installs == [("ja", "en")]

    def test_it_reports_a_failure(self, qtbot):
        packs = FakePacks()
        packs.install_result = False
        thread = PackageSetupThread(packs, "zz", "en")
        with qtbot.waitSignal(thread.setup_finished, timeout=5000) as outcome:
            thread.start()
        thread.wait()
        assert outcome.args == ["zz", "en", False]
