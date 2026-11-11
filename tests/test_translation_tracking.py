"""
Region tracking in TranslationWorker: how each OCR batch is matched to the regions already being
tracked (_update, _find_match), how regions age and disappear (miss counting, _prune,
DISPLAY_GRACE_MISSES), and how a language change resets all of it.

The tests call the worker's methods directly on one thread, with a fake translator and a fake
clock, so they need neither timing nor a running QThread.
"""

import logging

import pytest

from langscope import translation_worker
from langscope.models import RawSegment
from langscope.translation_providers import BaseTranslator
from langscope.translation_worker import (
    DISPLAY_GRACE_MISSES,
    EMA_ALPHA,
    IOU_THRESHOLD,
    MAX_MISSES,
    REGION_TTL,
    TEXT_SIM_THRESHOLD,
    TranslationWorker,
)
from tests.helpers import bbox

SENTENCE = "Hello brave new world"


class RecordingTranslator(BaseTranslator):
    """Translates "text" to "[<to>] text" and remembers every translate_many call.

    Set `error` to make translate_many raise, and put a text into `answers` to make the
    translator answer that text with a fixed value instead."""

    def __init__(self):
        self.calls = []  # (texts, from_code, to_code)
        self.error = None
        self.answers = {}

    def translate(self, text, from_code, to_code):
        return f"[{to_code}] {text}"

    def translate_many(self, texts, from_code, to_code):
        self.calls.append((list(texts), from_code, to_code))
        if self.error is not None:
            raise self.error
        return [
            self.answers[t] if t in self.answers else self.translate(t, from_code, to_code)
            for t in texts
        ]

    @property
    def translated_texts(self):
        return [text for texts, _, _ in self.calls for text in texts]


def seg(text=SENTENCE, x=10, y=10, width=190, height=30, score=0.9):
    return RawSegment(bbox=bbox(x, y, x + width, y + height), text=text, score=score)


@pytest.fixture
def translator():
    return RecordingTranslator()


@pytest.fixture
def worker(translator, clock, no_language_detection):
    w = TranslationWorker(translator=translator)
    w.set_languages("en", "fr")
    return w


def only_region(worker):
    (region,) = worker._regions.values()
    return region


def published(worker):
    worker._publish()
    return worker.latest_regions


class TestTuningConstants:
    def test_keep_their_documented_values(self):
        # These shape how text is tracked. Change them on purpose, and update this test.
        assert REGION_TTL == 8.0
        assert MAX_MISSES == 5
        assert DISPLAY_GRACE_MISSES == 0
        assert IOU_THRESHOLD == 0.30
        assert TEXT_SIM_THRESHOLD == 0.55
        assert EMA_ALPHA == 0.35


class TestNewRegions:
    def test_a_detection_creates_a_translated_region(self, worker, translator, clock):
        worker._update([seg(score=0.8)])
        region = only_region(worker)
        assert region.original_text == SENTENCE
        assert region.translated_text == f"[fr] {SENTENCE}"
        assert region.confidence == pytest.approx(0.8)
        assert region.miss_count == 0
        assert region.last_seen == clock.now
        assert region.smoothed_bbox == region.bbox == bbox(10, 10, 200, 40)
        assert translator.calls == [([SENTENCE], "en", "fr")]

    def test_separate_detections_get_separate_regions_translated_in_one_call(
        self, worker, translator
    ):
        worker._update([seg("First line of text", y=10), seg("Second line of text", y=100)])
        assert len(worker._regions) == 2
        assert translator.calls == [(["First line of text", "Second line of text"], "en", "fr")]

    def test_every_region_gets_its_own_id(self, worker):
        worker._update([seg("First line of text", y=10), seg("Second line of text", y=100)])
        assert len({r.region_id for r in worker._regions.values()}) == 2

    @pytest.mark.parametrize("text", ["12:45", "555-123-4567", "ok", "1234"])
    def test_numbers_and_timestamps_are_shown_untranslated(self, worker, translator, text):
        worker._update([seg(text)])
        region = only_region(worker)
        assert region.translated_text == text
        assert not region.snapshot().has_translation
        assert translator.calls == []

    def test_text_already_in_the_target_language_is_not_translated(
        self, worker, translator, monkeypatch
    ):
        monkeypatch.setattr(translation_worker, "_already_target", lambda text, to_code: True)
        worker._update([seg("Bonjour tout le monde")])
        assert only_region(worker).translated_text == "Bonjour tout le monde"
        assert translator.calls == []

    def test_the_target_language_is_what_is_checked(self, worker, monkeypatch):
        seen = []
        monkeypatch.setattr(
            translation_worker, "_already_target", lambda text, to_code: seen.append(to_code)
        )
        worker._update([seg()])
        assert seen == ["fr"]

    def test_a_same_language_pair_translates_nothing(self, worker, translator):
        worker.set_languages("fr", "fr")
        worker._update([seg("Bonjour tout le monde")])
        assert only_region(worker).translated_text == "Bonjour tout le monde"
        assert translator.calls == []

    def test_an_empty_batch_creates_nothing_and_calls_nothing(self, worker, translator):
        worker._update([])
        assert worker._regions == {} and translator.calls == []


class TestMatching:
    def test_a_repeat_sighting_updates_the_same_region(self, worker, clock):
        worker._update([seg(score=0.7)])
        first = only_region(worker)
        clock.advance(2.0)
        worker._update([seg(x=14, score=0.95)])

        assert only_region(worker) is first
        assert first.confidence == pytest.approx(0.95)
        assert first.last_seen == clock.now
        assert first.miss_count == 0

    def test_a_repeat_sighting_smooths_the_position(self, worker):
        worker._update([seg(x=10)])
        region = only_region(worker)
        worker._update([seg(x=50)])  # moved 40 px to the right, still overlapping
        expected = 10 + EMA_ALPHA * 40
        assert region.smoothed_bbox[0][0] == pytest.approx(expected)
        assert region.smoothed_bbox[1][0] == pytest.approx(200 + EMA_ALPHA * 40)
        assert region.bbox[0][0] == 10  # the raw first position is never touched

    def test_the_next_sighting_is_matched_against_the_smoothed_position(self, worker):
        worker._update([seg(x=10)])
        region = only_region(worker)
        for _ in range(8):
            worker._update([seg(x=70)])
        # A detection at the smoothed position keeps matching; the region has followed the text.
        assert region.smoothed_bbox[0][0] > 50
        assert len(worker._regions) == 1

    def test_unchanged_text_is_not_translated_again(self, worker, translator):
        worker._update([seg()])
        worker._update([seg(x=12)])
        worker._update([seg(x=14)])
        assert len(translator.calls) == 1
        # It does not even look the text up in the cache again.
        assert (worker.cache_hits, worker.cache_misses) == (0, 1)

    def test_changed_text_in_the_same_place_is_translated_again(self, worker, translator):
        worker._update([seg("Hello brave new world")])
        region = only_region(worker)
        worker._update([seg("Goodbye cruel old world")])

        assert only_region(worker) is region  # same place: the same region, new text
        assert region.original_text == "Goodbye cruel old world"
        assert region.translated_text == "[fr] Goodbye cruel old world"
        assert translator.translated_texts == ["Hello brave new world", "Goodbye cruel old world"]

    def test_a_detection_that_barely_overlaps_is_a_new_region(self, worker):
        worker._update([seg(x=10, width=100)])  # x 10..110
        old = only_region(worker)
        # x 90..190: 20 px overlap of 180 px total, an IoU of about 0.11
        worker._update([seg(x=90, width=100)])
        assert len(worker._regions) == 2
        assert old.miss_count == 1

    def test_a_detection_just_above_the_overlap_threshold_matches(self, worker):
        worker._update([seg(x=0, width=100)])  # x 0..100
        # shifted by 53: intersection 47, union 153, IoU 0.307
        worker._update([seg(x=53, width=100)])
        assert len(worker._regions) == 1

    def test_a_detection_just_below_the_overlap_threshold_does_not(self, worker):
        worker._update([seg(x=0, width=100)])
        # shifted by 55: intersection 45, union 155, IoU 0.290
        worker._update([seg(x=55, width=100)])
        assert len(worker._regions) == 2

    def test_a_detection_at_exactly_the_overlap_threshold_matches(self, worker):
        first, second = seg(x=0, width=130), seg(x=70, width=130)
        # intersection 60, union 200: an IoU of exactly 0.30
        assert translation_worker._iou(first.bbox, second.bbox) == IOU_THRESHOLD
        worker._update([first])
        worker._update([second])
        assert len(worker._regions) == 1

    @pytest.mark.parametrize(
        "new_text, same_line",
        [
            ("abcdeg", True),  # nearly the same text: similarity 0.83
            ("xyzwuv", False),  # unrelated text: similarity 0
        ],
    )
    def test_at_a_middling_overlap_the_text_decides(self, worker, new_text, same_line):
        # A match is judged on 0.6 * overlap + 0.4 * text similarity, which has to reach 0.40.
        # These two boxes have an IoU of 0.50, which on its own gives 0.30.
        worker._update([seg("abcdef", x=0, width=120)])
        original = only_region(worker)
        worker._update([seg(new_text, x=40, width=120)])  # x 40..160: intersection 80 of 160

        assert (len(worker._regions) == 1) is same_line
        assert (original.miss_count == 0) is same_line
        assert (original.original_text == new_text) is same_line

    def test_a_nearly_identical_place_matches_even_when_the_text_is_unrelated(self, worker):
        # An IoU of 0.80 gives 0.48 on its own, enough without any text similarity: the line
        # was rewritten where it stood.
        worker._update([seg("abcdef", x=0, width=180)])
        region = only_region(worker)
        worker._update([seg("xyzwuv", x=20, width=180)])
        assert only_region(worker) is region
        assert region.original_text == "xyzwuv"

    def test_a_region_is_claimed_by_at_most_one_detection_per_batch(self, worker):
        worker._update([seg("Some line of text here")])
        existing = only_region(worker)
        # Two detections both sit on top of the existing region. Only one may take it over,
        # otherwise they would overwrite each other and one line would never be shown.
        worker._update([seg("Some line of text here"), seg("Some line of text here", x=12)])
        assert len(worker._regions) == 2
        assert existing in worker._regions.values()
        assert all(r.miss_count == 0 for r in worker._regions.values())

    def test_the_best_overlapping_region_wins(self, worker):
        # Two regions that overlap each other (created in one batch, so neither claims the other).
        worker._update([seg("Alpha line of text", x=10), seg("Beta line of text", x=100)])
        alpha, beta = (worker._regions[rid] for rid in worker._regions)
        assert (alpha.original_text, beta.original_text) == (
            "Alpha line of text",
            "Beta line of text",
        )
        # A detection of Beta's text that overlaps both: Beta overlaps much more and says the
        # same thing, so it, not Alpha, takes it over.
        worker._update([seg("Beta line of text", x=95, width=190)])
        assert beta.miss_count == 0 and alpha.miss_count == 1

    def test_text_similarity_breaks_a_tie_between_equally_overlapping_regions(self, worker):
        worker._update([seg("Cats and dogs", y=10), seg("Bread and butter", y=10, x=0, width=190)])
        regions = list(worker._regions.values())
        by_text = {r.original_text: r for r in regions}
        worker._update([seg("Bread and butter", y=10, x=5)])
        assert by_text["Bread and butter"].miss_count == 0
        assert by_text["Cats and dogs"].miss_count == 1

    def test_a_dead_region_is_never_matched(self, worker):
        worker._update([seg()])
        old = only_region(worker)
        old.miss_count = MAX_MISSES + 1
        worker._update([seg()])
        assert len(worker._regions) == 2
        assert old.miss_count == MAX_MISSES + 2

    def test_a_region_at_exactly_max_misses_can_still_be_matched(self, worker):
        worker._update([seg()])
        old = only_region(worker)
        old.miss_count = MAX_MISSES
        worker._update([seg()])
        assert only_region(worker) is old and old.miss_count == 0


class TestTextChangesInPlace:
    """A region that is matched again with different text: what its new text is shown as."""

    def test_text_that_becomes_a_number_is_shown_as_it_is(self, worker, translator):
        worker._update([seg("Hello brave new world")])
        region = only_region(worker)
        worker._update([seg("12:45")])

        assert only_region(worker) is region
        assert region.original_text == "12:45"
        assert region.translated_text == "12:45"  # not the old translation
        assert not region.snapshot().has_translation
        assert translator.translated_texts == ["Hello brave new world"]

    def test_text_that_becomes_target_language_text_is_shown_as_it_is(
        self, worker, translator, monkeypatch
    ):
        worker._update([seg("Hello brave new world")])
        region = only_region(worker)
        monkeypatch.setattr(translation_worker, "_already_target", lambda text, to_code: True)
        worker._update([seg("Bonjour tout le monde")])

        assert region.translated_text == "Bonjour tout le monde"
        assert translator.translated_texts == ["Hello brave new world"]

    def test_a_same_language_pair_never_translates_a_change(self, worker, translator):
        worker.set_languages("fr", "fr")
        worker._update([seg("Bonjour tout le monde")])
        region = only_region(worker)
        worker._update([seg("Au revoir tout le monde")])

        assert region.translated_text == "Au revoir tout le monde"
        assert translator.calls == []


class TestTranslatorAnswers:
    @pytest.mark.parametrize("answer", ["", "   ", "\n", None])
    def test_a_blank_answer_keeps_the_original_text(self, worker, translator, answer):
        translator.answers[SENTENCE] = answer
        worker._update([seg()])
        assert only_region(worker).translated_text == SENTENCE

    def test_whitespace_around_an_answer_is_trimmed(self, worker, translator):
        translator.answers[SENTENCE] = "  Bonjour tout le monde \n"
        worker._update([seg()])
        assert only_region(worker).translated_text == "Bonjour tout le monde"

    def test_a_translator_that_raises_leaves_the_original_text_on_screen(
        self, worker, translator, caplog
    ):
        translator.error = RuntimeError("out of memory")
        with caplog.at_level(logging.ERROR):
            worker._update([seg()])
        assert only_region(worker).translated_text == SENTENCE
        assert any("Translation error" in record.getMessage() for record in caplog.records)

    def test_the_next_batch_after_a_failure_is_translated_normally(self, worker, translator):
        translator.error = RuntimeError("out of memory")
        worker._update([seg("First line of text", y=10)])
        translator.error = None
        worker._update([seg("Second line of text", y=200)])
        by_text = {r.original_text: r for r in worker._regions.values()}
        assert by_text["Second line of text"].translated_text == "[fr] Second line of text"

    def test_a_text_whose_translation_failed_is_tried_again(self, worker, translator):
        translator.error = RuntimeError("out of memory")
        worker._update([seg(SENTENCE, y=10)])
        assert len(worker._tx_cache) == 0
        translator.error = None
        worker._update([seg(SENTENCE, y=300)])  # the same text, in a new region
        newest = list(worker._regions.values())[-1]
        assert newest.translated_text == f"[fr] {SENTENCE}"


class TestMissCounting:
    def test_regions_missed_by_a_batch_count_one_miss_each(self, worker):
        worker._update([seg()])
        region = only_region(worker)
        for expected in (1, 2, 3):
            worker._update([])
            assert region.miss_count == expected

    def test_a_detection_elsewhere_still_counts_as_a_miss_for_the_old_region(self, worker):
        worker._update([seg("First line of text", y=10)])
        first = only_region(worker)
        worker._update([seg("Second line of text", y=200)])
        assert first.miss_count == 1

    def test_a_reappearing_region_keeps_its_identity_and_clears_its_misses(self, worker):
        worker._update([seg()])
        region = only_region(worker)
        region_id = region.region_id
        for _ in range(3):
            worker._update([])
        assert region.miss_count == 3
        worker._update([seg()])
        assert only_region(worker).region_id == region_id
        assert region.miss_count == 0

    def test_a_region_missed_for_too_long_comes_back_as_a_new_region(self, worker):
        worker._update([seg()])
        old_id = only_region(worker).region_id
        for _ in range(MAX_MISSES + 1):
            worker._update([])
        worker._update([seg()])
        assert old_id not in {r.region_id for r in worker._regions.values() if r.miss_count == 0}
        assert sum(r.miss_count == 0 for r in worker._regions.values()) == 1


class TestDisplayGrace:
    def test_a_missed_region_is_hidden_by_default_but_still_tracked(self, worker):
        worker._update([seg()])
        assert len(published(worker)) == 1
        worker._update([])
        assert published(worker) == ()
        assert len(worker._regions) == 1  # kept, so it can re-match instantly

    def test_a_hidden_region_reappears_with_the_same_id(self, worker):
        worker._update([seg()])
        shown_id = published(worker)[0].region_id
        worker._update([])
        worker._update([seg()])
        assert published(worker)[0].region_id == shown_id

    def test_the_grace_setting_keeps_a_missed_region_visible_for_that_many_passes(
        self, worker, monkeypatch
    ):
        monkeypatch.setattr(translation_worker, "DISPLAY_GRACE_MISSES", 2)
        worker._update([seg()])
        for _ in range(2):
            worker._update([])
            assert len(published(worker)) == 1
        worker._update([])
        assert published(worker) == ()

    def test_only_regions_seen_in_the_latest_pass_are_published_by_default(self, worker):
        worker._update([seg("First line of text", y=10), seg("Second line of text", y=100)])
        worker._update([seg("Second line of text", y=100)])
        assert [r.original_text for r in published(worker)] == ["Second line of text"]


class TestPruning:
    def test_a_region_missed_max_misses_times_is_kept(self, worker):
        worker._update([seg()])
        for _ in range(MAX_MISSES):
            worker._update([])
        worker._prune()
        assert len(worker._regions) == 1

    def test_a_region_missed_more_than_max_misses_times_is_pruned(self, worker):
        worker._update([seg()])
        for _ in range(MAX_MISSES + 1):
            worker._update([])
        worker._prune()
        assert worker._regions == {}

    def test_a_missed_region_is_pruned_once_it_has_not_been_seen_for_the_ttl(self, worker, clock):
        worker._update([seg()])
        worker._update([])  # missed once, so the age limit applies to it
        clock.advance(REGION_TTL + 0.5)
        worker._prune()
        assert worker._regions == {}

    def test_a_missed_region_younger_than_the_ttl_is_kept(self, worker, clock):
        worker._update([seg()])
        worker._update([])
        clock.advance(REGION_TTL - 0.5)
        worker._prune()
        assert len(worker._regions) == 1

    def test_exactly_the_ttl_is_still_kept(self, worker, clock):
        worker._update([seg()])
        worker._update([])
        clock.advance(REGION_TTL)
        worker._prune()
        assert len(worker._regions) == 1
        clock.advance(0.001)
        worker._prune()
        assert worker._regions == {}

    def test_a_region_from_the_latest_pass_is_never_aged_out(self, worker, clock):
        """A static screen runs no OCR, so a visible region can be older than the TTL."""
        worker._update([seg()])
        clock.advance(10 * REGION_TTL)
        worker._prune()
        assert len(worker._regions) == 1

    def test_prune_removes_only_the_dead_regions(self, worker, clock):
        worker._update([seg("Stays on screen", y=10), seg("Goes away soon", y=100)])
        worker._update([seg("Stays on screen", y=10)])
        clock.advance(REGION_TTL + 1)
        worker._update([seg("Stays on screen", y=10)])  # seen again just now
        worker._prune()
        assert [r.original_text for r in worker._regions.values()] == ["Stays on screen"]

    def test_pruning_does_not_change_what_is_published(self, worker, clock):
        worker._update([seg()])
        before = published(worker)
        worker._prune()
        assert published(worker) == before

    def test_dead_regions_are_removed_while_batches_keep_arriving(
        self, translator, clock, no_language_detection, monkeypatch
    ):
        monkeypatch.setattr(translation_worker, "MAX_PENDING_BATCHES", 100)
        worker = TranslationWorker(translator=translator)
        worker.set_languages("en", "fr")
        for i in range(MAX_MISSES + 10):
            # Each batch holds one new line far from the last, so every region goes unseen
            # after its own batch.
            worker._mailbox.post("segments", [seg(f"Line number {i} of text", y=10 + 80 * i)])
        worker._mailbox.post("stop")

        worker.run()  # the real loop, on this thread: messages are waiting, so it never idles

        assert len(worker._regions) <= MAX_MISSES + 1


class TestCacheHits:
    def test_text_seen_before_is_served_from_the_cache(self, worker, translator):
        worker._update([seg(SENTENCE, y=10)])
        # The same text appears in a new place, so it is a new region, but the cache has it.
        worker._update([seg(SENTENCE, y=300)])
        assert translator.translated_texts == [SENTENCE]
        assert len(worker._regions) == 2
        assert all(r.translated_text == f"[fr] {SENTENCE}" for r in worker._regions.values())
        assert (worker.cache_hits, worker.cache_misses) == (1, 1)

    def test_new_text_is_a_miss(self, worker):
        worker._update([seg("First line of text", y=10), seg("Second line of text", y=100)])
        assert (worker.cache_hits, worker.cache_misses) == (0, 2)


class TestLanguageChange:
    def populate(self, worker):
        worker._update([seg("First line of text", y=10), seg("Second line of text", y=100)])
        worker._update([seg("First line of text", y=10)])
        worker._publish()

    def test_it_forgets_all_tracked_regions_and_the_published_snapshot(self, worker):
        self.populate(worker)
        assert len(worker._regions) == 2 and len(worker.latest_regions) == 1
        worker.set_languages("en", "de")
        assert worker._regions == {}
        assert worker.latest_regions == ()

    def test_it_empties_the_translation_cache(self, worker):
        self.populate(worker)
        worker.set_languages("en", "de")
        assert len(worker._tx_cache) == 0

    def test_old_text_is_translated_again_into_the_new_language(self, worker, translator):
        worker._update([seg(SENTENCE)])
        worker.set_languages("en", "de")
        worker._update([seg(SENTENCE)])
        assert translator.calls == [([SENTENCE], "en", "fr"), ([SENTENCE], "en", "de")]
        assert only_region(worker).translated_text == f"[de] {SENTENCE}"

    def test_the_same_text_is_a_brand_new_region_after_the_change(self, worker):
        worker._update([seg(SENTENCE)])
        old_id = only_region(worker).region_id
        worker.set_languages("en", "de")
        worker._update([seg(SENTENCE)])
        assert only_region(worker).region_id != old_id
        assert only_region(worker).miss_count == 0

    def test_it_clears_the_translators_own_cache(self, worker, translator):
        cleared = []
        translator.clear_cache = lambda: cleared.append(True)
        worker.set_languages("en", "de")
        assert cleared == [True]
