"""
translate_many: translating many OCR lines with one batched model call per hop, and handing the
results back aligned 1:1 with the inputs.

The real translator.py runs here against fake Argos translation objects (see fake_argos.py).
The fake model tags every word with the language it was translated into, so a result shows
which input it came from and which languages it passed through:

    "Hello world" into French              ->  "fr/Hello fr/world"
    "Hello world" Romanian -> English -> French  ->  "fr/en/Hello fr/en/world"
"""

import logging
import random

import pytest

from tests.fake_argos import IdentityTranslation, tag_words


@pytest.fixture
def translate_many(translator_module):
    return translator_module.translate_many


def expected(text, *via):
    """What the fake models make of *text* when it is translated into each language in `via`
    in turn. Blank text is never translated."""
    if not text.strip():
        return text
    for code in via:
        text = tag_words(text, code)
    return text


def sent(model, call=0):
    """The sentences one translate_batch call of a fake model was asked to translate."""
    return [" ".join(tokens) for tokens in model.batches[call][0]]


class TestAlignment:
    def test_a_failed_model_batch_returns_originals_with_failure_status(
        self, translator_module, fake_argos
    ):
        fake_argos.pair("en", "fr")
        fake_argos.model("en", "fr").error = RuntimeError("temporary model error")

        translated, succeeded = translator_module.translate_many_with_status(
            ["Hello world", "Good morning"], "en", "fr"
        )

        assert translated == ["Hello world", "Good morning"]
        assert succeeded == [False, False]

    def test_each_result_belongs_to_the_input_at_its_position(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        out = translate_many(["Hello world", "Good morning", "Thank you very much"], "en", "fr")
        assert out == [
            "fr/Hello fr/world",
            "fr/Good fr/morning",
            "fr/Thank fr/you fr/very fr/much",
        ]

    def test_an_identical_line_is_translated_once_and_fills_every_position(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr")
        texts = ["Red car", "Blue sky", "Red car", "Red car", "Blue sky"]
        assert translate_many(texts, "en", "fr") == [expected(t, "fr") for t in texts]
        assert sent(fake_argos.model("en", "fr")) == ["Red car", "Blue sky"]

    def test_blank_lines_pass_through_unchanged_and_in_place(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        texts = ["", "Hello", "   ", "\n", "World", "\t "]
        assert translate_many(texts, "en", "fr") == ["", "fr/Hello", "   ", "\n", "fr/World", "\t "]
        assert sent(fake_argos.model("en", "fr")) == ["Hello", "World"]

    def test_many_distinct_lines_stay_in_order(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        texts = [f"Line number {i} here" for i in range(500)]
        assert translate_many(texts, "en", "fr") == [expected(t, "fr") for t in texts]

    def test_an_empty_list_gives_an_empty_list_and_asks_argos_nothing(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr")
        assert translate_many([], "en", "fr") == []
        assert fake_argos.language_lookups == 0

    def test_blank_lines_alone_never_reach_the_model(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        texts = ["", "  "]
        assert translate_many(texts, "en", "fr") == texts
        assert fake_argos.model("en", "fr").batches == []
        assert fake_argos.language_lookups == 0

    def test_the_same_language_returns_the_inputs_without_asking_argos(
        self, translate_many, fake_argos
    ):
        texts = ["Bonjour le monde", "", "Bonjour le monde"]
        assert translate_many(texts, "fr", "fr") == texts
        assert fake_argos.language_lookups == 0

    def test_the_input_list_is_not_modified_or_returned(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        texts = ["Hello world", "", "Hello world"]
        before = list(texts)
        out = translate_many(texts, "en", "fr")
        assert texts == before
        assert out is not texts

    def test_random_mixes_of_lines_stay_aligned(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        model = fake_argos.model("en", "fr")
        pool = [
            "Hello world",
            "Good morning",
            "One. Two. Three",
            "Wait! What? No",
            "Note: this is it",
            "Thank you very much",
            "A b c d e f",
            "Yes",
            "",
            "  ",
        ]
        rng = random.Random(20240607)  # fixed, so a failure can be reproduced
        for _ in range(300):
            texts = [rng.choice(pool) for _ in range(rng.randint(0, 14))]
            batches_before = len(model.batches)
            assert translate_many(texts, "en", "fr") == [expected(t, "fr") for t in texts], texts
            has_text = any(t.strip() for t in texts)
            assert len(model.batches) - batches_before == (1 if has_text else 0), texts


class TestBatching:
    def test_all_lines_go_to_the_model_in_a_single_call(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        texts = ["Hello world", "Good morning", "Thank you"]
        translate_many(texts, "en", "fr")
        model = fake_argos.model("en", "fr")
        assert len(model.batches) == 1
        assert sent(model) == texts

    def test_every_sentence_of_every_line_shares_that_one_call(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        out = translate_many(["One. Two.", "Three", "Four! Five? Six"], "en", "fr")
        model = fake_argos.model("en", "fr")
        assert len(model.batches) == 1
        assert sent(model) == ["One.", "Two.", "Three", "Four!", "Five?", "Six"]
        # Each line gets back its own sentences, in order.
        assert out == ["fr/One. fr/Two.", "fr/Three", "fr/Four! fr/Five? fr/Six"]

    @pytest.mark.parametrize(
        "text", ["Hello world", "Wait!", "Hello.", "no punctuation at all", "你好！"]
    )
    def test_a_line_without_a_sentence_break_skips_the_sentence_splitter(
        self, translate_many, fake_argos, text
    ):
        pack = fake_argos.pair("en", "fr")
        assert translate_many([text], "en", "fr") == [expected(text, "fr")]
        assert pack.underlying.sentencizer.calls == []

    @pytest.mark.parametrize(
        "text", ["Hi. There", "Really? Yes", "a; b", "Note: this", "你好。世界"]
    )
    def test_a_line_with_a_sentence_break_is_given_to_the_sentence_splitter(
        self, translate_many, fake_argos, text
    ):
        pack = fake_argos.pair("en", "fr")
        assert translate_many([text], "en", "fr") == [expected(text, "fr")]
        assert pack.underlying.sentencizer.calls == [text]

    def test_lines_are_stripped_before_they_are_split_and_tokenized(
        self, translate_many, fake_argos
    ):
        pack = fake_argos.pair("en", "fr")
        out = translate_many(["  Hello world \n", "  Hi. There  "], "en", "fr")
        assert sent(fake_argos.model("en", "fr")) == ["Hello world", "Hi.", "There"]
        assert pack.underlying.sentencizer.calls == ["Hi. There"]
        assert out == ["fr/Hello fr/world", "fr/Hi. fr/There"]

    def test_the_model_is_called_with_the_decoding_settings_argos_uses(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr")
        fake_argos.settings.batch_size = 7
        fake_argos.settings.beam_size = 3
        translate_many(["Hello world"], "en", "fr")
        _, kwargs = fake_argos.model("en", "fr").batches[0]
        assert kwargs == {
            "target_prefix": None,
            "replace_unknowns": True,
            "max_batch_size": 7,
            "batch_type": "tokens",
            "beam_size": 3,
            "num_hypotheses": 1,
            "length_penalty": 0.2,
            "return_scores": False,
        }


class TestPivot:
    """A pair with no direct pack is a CompositeTranslation of two packs, such as ro -> en -> fr."""

    @pytest.fixture
    def ro_fr(self, fake_argos):
        fake_argos.chain("ro", "en", "fr")
        return fake_argos

    def test_a_pivoted_pair_goes_through_the_middle_language(self, translate_many, ro_fr):
        out = translate_many(["Hello world", "Good morning"], "ro", "fr")
        assert out == ["fr/en/Hello fr/en/world", "fr/en/Good fr/en/morning"]

    def test_each_hop_is_one_call_and_the_second_translates_the_output_of_the_first(
        self, translate_many, ro_fr
    ):
        translate_many(["Hello world", "Good morning"], "ro", "fr")
        first, second = ro_fr.model("ro", "en"), ro_fr.model("en", "fr")
        assert len(first.batches) == 1 and len(second.batches) == 1
        assert sent(first) == ["Hello world", "Good morning"]
        assert sent(second) == ["en/Hello en/world", "en/Good en/morning"]

    def test_a_repeated_line_goes_through_each_hop_once(self, translate_many, ro_fr):
        texts = ["Hello world", "Hello world", "Good morning"]
        assert translate_many(texts, "ro", "fr") == [expected(t, "en", "fr") for t in texts]
        assert sent(ro_fr.model("ro", "en")) == ["Hello world", "Good morning"]
        assert len(sent(ro_fr.model("en", "fr"))) == 2

    def test_blank_lines_skip_both_hops(self, translate_many, ro_fr):
        texts = ["", "Hello world", "  "]
        assert translate_many(texts, "ro", "fr") == ["", "fr/en/Hello fr/en/world", "  "]
        assert sent(ro_fr.model("ro", "en")) == ["Hello world"]
        assert sent(ro_fr.model("en", "fr")) == ["en/Hello en/world"]

    def test_lines_with_several_sentences_stay_aligned_through_both_hops(
        self, translate_many, ro_fr
    ):
        texts = ["One. Two. Three", "Four", "Five! Six", "Four"]
        assert translate_many(texts, "ro", "fr") == [expected(t, "en", "fr") for t in texts]

    def test_random_mixes_of_lines_stay_aligned(self, translate_many, ro_fr):
        pool = ["Hello world", "One. Two. Three", "Wait! What? No", "Yes", "", " ", "A b c d"]
        rng = random.Random(7)
        for _ in range(200):
            texts = [rng.choice(pool) for _ in range(rng.randint(0, 12))]
            assert translate_many(texts, "ro", "fr") == [expected(t, "en", "fr") for t in texts]

    def test_three_hops_run_in_order(self, translate_many, fake_argos):
        fake_argos.chain("ro", "en", "de", "fr")
        texts = ["Hello world", "", "Good morning", "Hello world"]
        assert translate_many(texts, "ro", "fr") == [expected(t, "en", "de", "fr") for t in texts]
        for pair in [("ro", "en"), ("en", "de"), ("de", "fr")]:
            assert len(fake_argos.model(*pair).batches) == 1

    def test_a_failure_in_the_second_hop_gives_back_the_original_lines(self, translate_many, ro_fr):
        # Not the half-translated output of the first hop.
        ro_fr.model("en", "fr").error = RuntimeError("out of memory")
        texts = ["Hello world", "", "Good morning"]
        assert translate_many(texts, "ro", "fr") == texts

    def test_the_hops_are_not_run_when_the_first_one_fails(self, translate_many, ro_fr):
        ro_fr.model("ro", "en").error = RuntimeError("out of memory")
        texts = ["Hello world"]
        assert translate_many(texts, "ro", "fr") == texts
        assert ro_fr.model("en", "fr").batches == []


class TestOtherKindsOfTranslation:
    def test_a_package_translation_without_the_cache_wrapper_is_batched_too(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr", cached=False)
        texts = ["Hello world", "Good morning"]
        assert translate_many(texts, "en", "fr") == [expected(t, "fr") for t in texts]
        assert len(fake_argos.model("en", "fr").batches) == 1

    def test_an_unknown_kind_is_translated_text_by_text_and_stays_aligned(
        self, translate_many, fake_argos
    ):
        remote = fake_argos.remote("en", "fr")
        texts = ["Hello world", "", "Good morning", "Hello world"]
        assert translate_many(texts, "en", "fr") == [expected(t, "fr") for t in texts]
        # Blank lines are not sent, and a repeated line is sent once.
        assert remote.calls == ["Hello world", "Good morning"]

    def test_a_pivot_can_mix_a_batched_hop_with_an_unknown_kind(self, translate_many, fake_argos):
        remote = fake_argos.remote("en", "fr")
        fake_argos.pivot(fake_argos.pair("ro", "en"), remote)
        texts = ["Hello world", "", "Good morning", "Hello world"]
        assert translate_many(texts, "ro", "fr") == [expected(t, "en", "fr") for t in texts]
        assert sent(fake_argos.model("ro", "en")) == ["Hello world", "Good morning"]
        assert remote.calls == ["en/Hello en/world", "en/Good en/morning"]

    def test_an_identity_translation_hands_back_the_texts(self, translator_module, fake_argos):
        # Argos only makes an identity translation from a language to itself, which
        # translate_many never asks for, so this branch is reachable only directly.
        identity = IdentityTranslation(fake_argos.language("en"))
        texts = ["Hello world", "Good morning"]
        out = translator_module._batch_translation(identity, texts)
        assert out == texts
        assert out is not texts


class TestDecoding:
    def test_a_target_prefix_is_sent_for_every_sentence_and_removed_from_the_result(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr", target_prefix="__fr__")
        out = translate_many(["Hello world", "Good morning"], "en", "fr")
        _, kwargs = fake_argos.model("en", "fr").batches[0]
        assert kwargs["target_prefix"] == [["__fr__"], ["__fr__"]]
        assert out == ["fr/Hello fr/world", "fr/Good fr/morning"]

    def test_the_space_a_tokenizer_puts_in_front_is_removed(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr", leading_space=True)
        assert translate_many(["Hello world"], "en", "fr") == ["fr/Hello fr/world"]


class TestWhenTranslationIsNotPossible:
    TEXTS = ["Hello world", "", "Good morning", "Hello world"]

    def test_no_installed_languages_gives_back_the_input_lines(self, translate_many):
        assert translate_many(self.TEXTS, "en", "fr") == self.TEXTS

    def test_a_missing_target_language_gives_back_the_input_lines(self, translate_many, fake_argos):
        fake_argos.language("en")
        assert translate_many(self.TEXTS, "en", "fr") == self.TEXTS

    def test_two_installed_languages_with_no_path_between_them_give_back_the_input_lines(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr")
        fake_argos.language("ja")
        assert translate_many(self.TEXTS, "en", "ja") == self.TEXTS
        assert fake_argos.model("en", "fr").batches == []

    def test_a_model_error_gives_back_the_input_lines_and_is_logged(
        self, translate_many, fake_argos, caplog
    ):
        fake_argos.pair("en", "fr")
        fake_argos.model("en", "fr").error = RuntimeError("boom")
        with caplog.at_level(logging.ERROR, logger="langscope.translator"):
            out = translate_many(self.TEXTS, "en", "fr")
        assert out == self.TEXTS
        failures = [r for r in caplog.records if "Batch translation failed" in r.getMessage()]
        assert len(failures) == 1 and failures[0].exc_info

    def test_an_error_while_listing_the_languages_gives_back_the_input_lines(
        self, translate_many, fake_argos, monkeypatch
    ):
        def broken():
            raise OSError("the data directory is gone")

        translate_module = fake_argos.modules["argostranslate.translate"]
        monkeypatch.setattr(translate_module, "get_installed_languages", broken)
        assert translate_many(self.TEXTS, "en", "fr") == self.TEXTS

    def test_a_line_the_model_returns_nothing_for_keeps_its_own_text(
        self, translate_many, fake_argos
    ):
        fake_argos.pair("en", "fr")
        fake_argos.model("en", "fr").empty_for = {"Bad input"}
        out = translate_many(["Hello", "Bad input", "World"], "en", "fr")
        assert out == ["fr/Hello", "Bad input", "fr/World"]

    def test_a_line_with_no_sentences_keeps_its_text_and_shifts_no_other_result(
        self, translate_many, fake_argos
    ):
        pack = fake_argos.pair("en", "fr")
        pack.underlying.sentencizer.no_sentences = {"... skipped"}
        texts = ["Hello. There", "... skipped", "World. Again"]
        assert translate_many(texts, "en", "fr") == [
            "fr/Hello. fr/There",
            "... skipped",
            "fr/World. fr/Again",
        ]

    def test_lines_that_all_have_no_sentences_never_reach_the_model(
        self, translate_many, fake_argos
    ):
        pack = fake_argos.pair("en", "fr")
        pack.underlying.sentencizer.no_sentences = {"... one", "... two"}
        texts = ["... one", "... two"]
        assert translate_many(texts, "en", "fr") == texts
        assert fake_argos.model("en", "fr").batches == []


class TestModelLoading:
    def test_the_model_is_loaded_on_first_use_with_argos_settings(self, translate_many, fake_argos):
        pack = fake_argos.pair("en", "fr", loaded=False)
        assert translate_many(["Hello world"], "en", "fr") == ["fr/Hello fr/world"]
        model_path = str(pack.underlying.pkg.package_path / "model")
        assert fake_argos.model_loads == [
            (
                model_path,
                {"device": "cpu", "inter_threads": 1, "intra_threads": 0, "compute_type": "auto"},
            )
        ]
        assert pack.underlying.translator is fake_argos.model("en", "fr")

    def test_a_loaded_model_is_not_loaded_again(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr", loaded=False)
        translate_many(["Hello world"], "en", "fr")
        translate_many(["Good morning"], "en", "fr")
        assert len(fake_argos.model_loads) == 1

    def test_a_model_that_is_already_loaded_is_used_as_it_is(self, translate_many, fake_argos):
        fake_argos.pair("en", "fr")
        translate_many(["Hello world"], "en", "fr")
        assert fake_argos.model_loads == []

    def test_a_failed_cuda_load_falls_back_to_the_cpu(self, translate_many, fake_argos):
        fake_argos.settings.device = "cuda"
        fake_argos.settings.compute_type = "float16"
        fake_argos.model_load_errors["cuda"] = RuntimeError("CUDA out of memory")
        fake_argos.pair("en", "fr", loaded=False)

        assert translate_many(["Hello world"], "en", "fr") == ["fr/Hello fr/world"]

        loads = [kwargs for _, kwargs in fake_argos.model_loads]
        assert [kw["device"] for kw in loads] == ["cuda", "cpu"]
        assert loads[1]["compute_type"] == "auto"  # float16 does not run on a CPU
        assert fake_argos.settings.device == "cpu"

    def test_after_a_cuda_failure_the_next_model_goes_straight_to_the_cpu(
        self, translate_many, fake_argos
    ):
        fake_argos.settings.device = "cuda"
        fake_argos.model_load_errors["cuda"] = RuntimeError("CUDA out of memory")
        fake_argos.pivot(
            fake_argos.pair("ro", "en", loaded=False), fake_argos.pair("en", "fr", loaded=False)
        )

        assert translate_many(["Hello world"], "ro", "fr") == ["fr/en/Hello fr/en/world"]
        assert [kw["device"] for _, kw in fake_argos.model_loads] == ["cuda", "cpu", "cpu"]

    def test_a_failed_cpu_load_gives_back_the_input_lines(self, translate_many, fake_argos):
        fake_argos.model_load_errors["cpu"] = RuntimeError("the model file is corrupt")
        fake_argos.pair("en", "fr", loaded=False)
        texts = ["Hello world", "", "Good morning"]
        assert translate_many(texts, "en", "fr") == texts
        assert len(fake_argos.model_loads) == 1  # no fallback from the CPU


class TestPreloadModel:
    """preload_model loads the model and runs a tiny batch, so that the first real translation
    does not pay the load cost."""

    def test_it_loads_the_model_and_reports_success(self, translator_module, fake_argos, caplog):
        pack = fake_argos.pair("en", "fr", loaded=False)
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            assert translator_module.preload_model("en", "fr") is True
        assert pack.underlying.translator is fake_argos.model("en", "fr")
        assert sent(fake_argos.model("en", "fr")) == ["Hello world"]
        (record,) = [r for r in caplog.records if "warmed up" in r.getMessage()]
        assert record.levelno == logging.INFO
        assert "en->fr" in record.getMessage()

    def test_a_pair_that_is_not_installed_is_not_ready(self, translator_module, fake_argos):
        assert translator_module.preload_model("en", "fr") is False

    def test_the_same_language_needs_no_model(self, translator_module, fake_argos):
        assert translator_module.preload_model("en", "en") is True
        assert fake_argos.language_lookups == 0

    def test_a_failure_is_reported_as_not_ready_and_logged_as_a_warning(
        self, translator_module, fake_argos, caplog
    ):
        fake_argos.pair("en", "fr")
        fake_argos.model("en", "fr").error = RuntimeError("out of memory")
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            assert translator_module.preload_model("en", "fr") is False
        (record,) = [r for r in caplog.records if "Warm-up skipped" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "out of memory" in record.getMessage()


class TestLogMessages:
    def test_a_missing_translation_path_is_a_warning_that_names_the_pair(
        self, translate_many, fake_argos, caplog
    ):
        fake_argos.language("en")
        fake_argos.language("ja")
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translate_many(["Hello world"], "en", "ja")
        (record,) = [r for r in caplog.records if "No translation path" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "en->ja" in record.getMessage()

    def test_a_batch_failure_names_the_pair_and_keeps_the_traceback(
        self, translate_many, fake_argos, caplog
    ):
        fake_argos.pair("en", "fr")
        fake_argos.model("en", "fr").error = RuntimeError("boom")
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translate_many(["Hello world"], "en", "fr")
        (record,) = [r for r in caplog.records if "Batch translation failed" in r.getMessage()]
        assert record.levelno == logging.ERROR
        assert "en->fr" in record.getMessage()
        assert record.exc_info and "boom" in str(record.exc_info[1])

    def test_a_cuda_load_failure_is_a_warning_before_the_cpu_fallback(
        self, translate_many, fake_argos, caplog
    ):
        fake_argos.settings.device = "cuda"
        fake_argos.model_load_errors["cuda"] = RuntimeError("CUDA out of memory")
        fake_argos.pair("en", "fr", loaded=False)
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translate_many(["Hello world"], "en", "fr")
        (record,) = [r for r in caplog.records if "falling back to CPU" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "CUDA out of memory" in record.getMessage()

    def test_the_device_choice_is_logged_not_printed(
        self, translator_module, fake_argos, caplog, capsys, monkeypatch
    ):
        monkeypatch.delenv("ARGOS_DEVICE_TYPE", raising=False)
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translator_module.init_translation_backend()
        assert any("Device: CPU" in r.getMessage() for r in caplog.records)
        assert capsys.readouterr().out == ""
