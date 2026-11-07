"""
translate_text: translating one string, with the log messages that explain what happened.

The batched path, translate_many, has its own tests (test_translate_many.py). translate_text
is what BaseTranslator.translate uses, one text at a time.
"""

import logging

import pytest

LOGGER = "langscope.translator"


@pytest.fixture
def translate_text(translator_module):
    return translator_module.translate_text


@pytest.fixture
def en_fr(fake_argos):
    """en -> fr through a kind of translation that can translate one text at a time."""
    return fake_argos.remote("en", "fr")


class TestResult:
    def test_the_text_is_translated(self, translate_text, en_fr):
        assert translate_text("Hello world", "en", "fr") == "fr/Hello fr/world"
        assert en_fr.calls == ["Hello world"]

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_blank_text_is_returned_as_it_is(self, translate_text, en_fr, text):
        assert translate_text(text, "en", "fr") == text
        assert en_fr.calls == []

    def test_the_same_language_is_returned_as_it_is(self, translate_text, fake_argos):
        assert translate_text("Bonjour le monde", "fr", "fr") == "Bonjour le monde"
        assert fake_argos.language_lookups == 0

    def test_whitespace_around_the_translation_is_trimmed(self, translate_text, en_fr):
        en_fr.translate = lambda text: "  Bonjour \n"
        assert translate_text("Hello", "en", "fr") == "Bonjour"

    @pytest.mark.parametrize("answer", ["", "   ", None])
    def test_a_blank_translation_gives_back_the_original(self, translate_text, en_fr, answer):
        en_fr.translate = lambda text: answer
        assert translate_text("Hello", "en", "fr") == "Hello"


class TestWhenItCannotTranslate:
    def test_a_missing_language_gives_back_the_text_and_lists_what_is_installed(
        self, translate_text, fake_argos, caplog
    ):
        fake_argos.language("en")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert translate_text("Hello", "en", "fr") == "Hello"
        (record,) = [r for r in caplog.records if "Language not found" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "from=en, to=None" in record.getMessage()
        assert "['en']" in record.getMessage()

    def test_no_path_between_two_languages_gives_back_the_text(
        self, translate_text, fake_argos, caplog
    ):
        fake_argos.language("en")
        fake_argos.language("ja")
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert translate_text("Hello", "en", "ja") == "Hello"
        (record,) = [r for r in caplog.records if "No translation path" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "en->ja" in record.getMessage()

    def test_an_error_gives_back_the_text_and_is_logged_with_the_pair(
        self, translate_text, en_fr, caplog
    ):
        def broken(text):
            raise RuntimeError("boom")

        en_fr.translate = broken
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert translate_text("Hello", "en", "fr") == "Hello"
        (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert "en->fr" in record.getMessage()
        assert "boom" in record.getMessage()

    def test_an_error_while_listing_the_languages_gives_back_the_text(
        self, translate_text, fake_argos, monkeypatch
    ):
        def broken():
            raise OSError("the data directory is gone")

        translate_module = fake_argos.modules["argostranslate.translate"]
        monkeypatch.setattr(translate_module, "get_installed_languages", broken)
        assert translate_text("Hello", "en", "fr") == "Hello"


class TestDebugLogging:
    def test_a_translation_logs_its_input_and_result_at_debug_level(
        self, translate_text, en_fr, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            translate_text("Hello world", "en", "fr")
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert "[Translator] Translating (en->fr): 'Hello world'" in messages
        assert "[Translator] OK Result: 'fr/Hello fr/world'" in messages

    def test_long_text_and_long_results_are_cut_to_50_characters_in_the_log(
        self, translate_text, en_fr, caplog
    ):
        text = "word " * 30
        result = " ".join(f"fr/{word}" for word in text.split())
        assert len(text) > 50 and len(result) > 50
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert translate_text(text, "en", "fr") == result
        messages = [r.getMessage() for r in caplog.records]
        assert f"[Translator] Translating (en->fr): {text[:50]!r}" in messages
        assert f"[Translator] OK Result: {result[:50]!r}" in messages
