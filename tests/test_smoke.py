"""
Smoke tests: every module imports, and the Qt pieces build and paint offscreen.

These exist because a bad merge once left the app unable to start while all the other
tests still passed. They target that class of failure: names missing at import or paint
time, signals that no longer exist (an AttributeError inside a worker thread), and
widgets that no longer construct.

No display is needed (conftest.py selects Qt's offscreen platform). The ML packages are
not needed either: modules that import argostranslate/ctranslate2 are skipped, with a
reason, when those packages are not installed (as in CI).
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest
from PyQt6.QtWidgets import QCheckBox, QLabel

from tests.helpers import bbox, render
from translation_providers import BaseTranslator

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Top-level scripts that are not part of the application and must never be imported by a test,
# such as the git-ignored diag_ocr.py scratch script, which runs OCR as soon as it is imported.
_NOT_APP_MODULES = {"diag_ocr"}

# Modules that import the ML packages at module level, and the packages they need.
_NEEDS_ML_PACKAGES = {
    "translator": ("argostranslate", "ctranslate2"),
    "main": ("argostranslate", "ctranslate2"),  # imports translator
}


def _app_modules() -> list[str]:
    return sorted(p.stem for p in ROOT.glob("*.py") if p.stem not in _NOT_APP_MODULES)


# ── Imports ──────────────────────────────────────────────────────────────────


def test_module_discovery_finds_the_application():
    # Guards against the import test below silently running on nothing, for example after
    # the modules move.
    assert {"ui", "main", "translation_worker", "ocr_providers"} <= set(_app_modules())


@pytest.mark.parametrize("module_name", _app_modules())
def test_module_imports(module_name):
    for package in _NEEDS_ML_PACKAGES.get(module_name, ()):
        pytest.importorskip(
            package,
            reason=f"{module_name} imports {package}, a heavy dependency that CI does not install",
        )
    importlib.import_module(module_name)


def test_main_imports_with_translator_stubbed(monkeypatch):
    """main.py's imports from the other modules still resolve, even without argostranslate.

    Linters cannot see across modules: this is what catches main.py importing a name that
    ui.py no longer defines. Loaded under a private name so the cached module is not reused
    by test_module_imports.
    """
    monkeypatch.setitem(sys.modules, "translator", types.ModuleType("translator"))
    spec = importlib.util.spec_from_file_location("main_under_test", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "PackageSetupThread")


# ── Widgets ──────────────────────────────────────────────────────────────────


def test_control_panel_builds_with_its_controls(qtbot):
    from ui import ControlPanel

    panel = ControlPanel()
    qtbot.addWidget(panel)
    panel.show()

    # Exactly one checkbox per toggle. (The toolbar once ended up with a duplicate Debug Mode.)
    checkboxes = panel.findChildren(QCheckBox)
    assert sorted(box.text().strip() for box in checkboxes) == ["Debug Mode", "Desktop Overlay"]
    by_label = {box.text().strip(): box for box in checkboxes}

    with qtbot.waitSignal(panel.overlay_toggled, timeout=1000) as overlay_signal:
        by_label["Desktop Overlay"].setChecked(True)
    assert overlay_signal.args == [True]

    with qtbot.waitSignal(panel.debug_toggled, timeout=1000) as debug_signal:
        by_label["Debug Mode"].setChecked(True)
    assert debug_signal.args == [True]


def test_control_panel_shows_stats_only_in_debug_mode(qtbot):
    from ui import ControlPanel

    panel = ControlPanel()
    qtbot.addWidget(panel)
    panel.show()
    stats = panel.findChild(QLabel, "stats")
    assert stats is not None
    assert stats.isHidden()

    panel.set_stats(12.0, 4.0, 8.0, 3.0, cache_hits=3, cache_misses=1)
    panel.set_debug(True)
    assert not stats.isHidden()
    assert stats.text() != ""

    panel.set_debug(False)
    assert stats.isHidden()
    assert stats.text() == ""


def test_live_feed_paints_translations(qtbot):
    from ui import LiveFeedWidget

    feed = LiveFeedWidget()
    qtbot.addWidget(feed)
    feed.resize(480, 240)
    frame = np.full((120, 240, 3), 200, np.uint8)
    detection = (bbox(10, 10, 200, 40), "hello", "bonjour", "r1", 0.9)

    feed.update_frame(frame, [], 240, 120)
    without_translation = render(feed)
    feed.update_frame(frame, [detection], 240, 120)
    with_translation = render(feed)
    assert with_translation != without_translation

    feed.set_debug(True)  # the debug overlay has its own paint path
    assert not render(feed).isNull()


def test_overlay_window_builds_and_paints_regions(qtbot):
    from models import TranslationRegion
    from overlay_window import OverlayWindow

    overlay = OverlayWindow()
    qtbot.addWidget(overlay)
    capture_rect = {"top": 0, "left": 0, "width": 400, "height": 120}
    region = TranslationRegion(
        bbox=bbox(10, 10, 200, 40), original_text="hello", translated_text="bonjour"
    )

    overlay.set_active(True)
    overlay.update_regions([], capture_rect)
    empty = render(overlay)
    overlay.update_regions([region], capture_rect)
    painted = render(overlay)
    assert painted != empty


# ── Worker thread ────────────────────────────────────────────────────────────


class _FakeTranslator(BaseTranslator):
    def translate(self, text, from_code, to_code):
        return f"[{to_code}] {text}"


def test_translation_worker_thread_publishes_translated_regions(qtbot, monkeypatch):
    """Runs the real worker thread. Linters cannot see an AttributeError inside run()."""
    import translation_worker
    from models import RawSegment

    # Language detection is irrelevant here, and langdetect is non-deterministic on short text.
    monkeypatch.setattr(translation_worker, "_already_target", lambda text, to_code: False)

    worker = translation_worker.TranslationWorker(translator=_FakeTranslator())
    worker.set_languages("en", "fr")
    worker.start()
    try:
        text = "Hello brave new world"
        worker.enqueue([RawSegment(bbox=bbox(10, 10, 200, 40), text=text, score=0.9)])
        qtbot.waitUntil(lambda: bool(worker.latest_regions), timeout=10_000)
        (region,) = worker.latest_regions
        assert region.original_text == text
        assert region.translated_text == f"[fr] {text}"
    finally:
        worker.stop()
    assert not worker.isRunning()
