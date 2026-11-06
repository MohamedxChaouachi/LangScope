"""
Test configuration for LangScope.

Ensures the project root (the directory containing translation_worker.py and
models.py) is importable when pytest is run from anywhere, and makes Qt run without
a display.
"""

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Qt's offscreen platform renders into memory, so widget tests need no display (CI has none).
# This must be set before the first QApplication is created. Set QT_QPA_PLATFORM yourself
# (for example to "windows") to watch the tests run in real windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def no_language_detection(monkeypatch):
    """Make the worker treat all text as not yet in the target language.

    langdetect is slow, and non-deterministic on short strings."""
    import translation_worker

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
    monkeypatch.delitem(sys.modules, "translator", raising=False)
    import translator

    yield translator
    # Do not leave a module bound to fakes behind for other tests to find.
    sys.modules.pop("translator", None)
