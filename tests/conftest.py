"""
Test configuration for LangScope.

The tests import the installed langscope package (pip install -e .), and Qt is made to run
without a display.
"""

import os
import time

import pytest

from tests.helpers import fresh_import

# Qt's offscreen platform renders into memory, so widget tests need no display (CI has none).
# This must be set before the first QApplication is created. Set QT_QPA_PLATFORM yourself
# (for example to "windows") to watch the tests run in real windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeClock:
    """A time.monotonic() that only moves when a test says so."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    """Replace time.monotonic with a FakeClock for the length of the test.

    Only for tests that run no other threads: it patches the time module itself."""
    fake = FakeClock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.fixture
def no_language_detection(monkeypatch):
    """Make the worker treat all text as not yet in the target language.

    langdetect is slow, and non-deterministic on short strings."""
    from langscope import translation_worker

    monkeypatch.setattr(translation_worker, "_already_target", lambda text, to_code: False)


@pytest.fixture
def fake_argos(monkeypatch, tmp_path):
    """Fake argostranslate and ctranslate2, even when the real packages are installed."""
    from tests.fake_argos import FakeArgos

    fake = FakeArgos(tmp_path / "argos-data")
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def translator_module(fake_argos, monkeypatch):
    """The real translator.py, freshly imported against the fakes from `fake_argos`."""
    # Not left behind afterwards: other tests must not find a module that is bound to fakes.
    with fresh_import("langscope.translator") as translator:
        yield translator
