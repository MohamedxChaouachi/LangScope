"""Language detection is seeded once at startup, so the same text always gets the same answer."""

import pytest

from langscope import main as main_module
from langscope import translation_worker

langdetect = pytest.importorskip("langdetect")


@pytest.fixture
def unseeded(monkeypatch):
    """langdetect as it is when nothing has seeded it: every detection is random."""
    monkeypatch.setattr(langdetect.DetectorFactory, "seed", None)


def test_init_seeds_langdetect_with_zero(unseeded):
    translation_worker.init_language_detection()
    assert langdetect.DetectorFactory.seed == 0


def test_detection_is_repeatable_once_seeded(unseeded):
    translation_worker.init_language_detection()
    # Few words that many languages share: unseeded, the guess can change from call to call.
    text = "Hotel restaurant bar taxi pizza"
    answers = {str(langdetect.detect_langs(text)) for _ in range(50)}
    assert len(answers) == 1


def test_init_without_langdetect_installed_does_not_raise(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "langdetect", None)  # makes the import fail
    translation_worker.init_language_detection()


def test_the_application_seeds_language_detection_at_startup(monkeypatch):
    calls = []

    class FakeApp:
        def __init__(self, argv):
            self.aboutToQuit = type("S", (), {"connect": lambda self, slot: None})()

        def setWindowIcon(self, icon):
            pass

        def exec(self):
            return 0

    class FakeController:
        shutdown = staticmethod(lambda: None)

        def show(self):
            pass

    monkeypatch.setattr(main_module, "configure_logging", lambda config: None)
    monkeypatch.setattr(main_module, "load_config", lambda: None)
    monkeypatch.setattr(main_module.ocr_providers, "init_ocr_backend", lambda: None)
    monkeypatch.setattr(main_module.translator, "init_translation_backend", lambda: None)
    monkeypatch.setattr(
        main_module.translation_worker, "init_language_detection", lambda: calls.append("seed")
    )
    monkeypatch.setattr(main_module, "QApplication", FakeApp)
    monkeypatch.setattr(main_module, "build_controller", lambda: FakeController())

    assert main_module.main([]) == 0
    assert calls == ["seed"]
