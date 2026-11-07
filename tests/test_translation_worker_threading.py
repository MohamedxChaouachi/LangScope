"""
Thread safety of TranslationWorker.

The worker's tracking state (regions, translation cache, language pair) is touched only by the
worker thread while it runs. The GUI and OCR threads talk to it through a mailbox, and it
publishes results as immutable snapshots, so the render timer never sees an object the worker
is still changing.
"""

import dataclasses
import threading

import pytest

from langscope.models import RawSegment
from langscope.translation_providers import BaseTranslator
from langscope.translation_worker import TranslationWorker
from tests.helpers import bbox

SENTENCE = "Hello brave new world"


def segment(text=SENTENCE, x=10):
    return RawSegment(bbox=bbox(x, 10, x + 190, 40), text=text, score=0.9)


class GatedTranslator(BaseTranslator):
    """Translates "text" to "[<to>] text". Can hold the worker open inside translate_many."""

    def __init__(self):
        self.calls = []  # (texts, from_code, to_code) for every translate_many call
        self.cleared_on = []  # the thread id each clear_cache() call ran on
        self.entered = threading.Event()  # set when a translate_many call has started
        self.release = threading.Event()  # translate_many waits for this
        self.release.set()

    def translate(self, text, from_code, to_code):
        return f"[{to_code}] {text}"

    def translate_many(self, texts, from_code, to_code):
        self.calls.append((list(texts), from_code, to_code))
        self.entered.set()
        assert self.release.wait(10), "test never released the translator"
        return [self.translate(t, from_code, to_code) for t in texts]

    def clear_cache(self):
        self.cleared_on.append(threading.get_ident())


@pytest.fixture
def worker(no_language_detection):
    translator = GatedTranslator()
    w = TranslationWorker(translator=translator)
    w.translator = translator
    yield w
    translator.release.set()  # never leave the worker blocked inside the translator
    w.stop()


def shown_translation(worker):
    regions = worker.latest_regions
    return regions[0].translated_text if regions else None


class TestLanguageChanges:
    def test_applied_on_the_worker_thread(self, qtbot, worker):
        worker.start()
        worker.set_languages("en", "de")  # called from the GUI thread, worker running

        qtbot.waitUntil(lambda: bool(worker.translator.cleared_on), timeout=5000)
        # The cache was cleared on the worker's own thread, never on the caller's. Clearing
        # on the caller's thread is what let the GUI empty dictionaries the worker was reading.
        assert threading.get_ident() not in worker.translator.cleared_on

    def test_before_start_takes_effect_immediately(self, worker):
        worker.set_languages("fr", "de")
        worker._update([segment("Bonjour tout le monde")])
        assert worker.translator.calls == [(["Bonjour tout le monde"], "fr", "de")]

    def test_change_during_a_batch_does_not_poison_the_cache(self, qtbot, worker):
        """A batch still translating when the language changes must not leave its old-language
        result in the cache, or the new language would keep serving it."""
        worker.translator.release.clear()
        worker.set_languages("en", "fr")
        worker.start()

        worker.enqueue([segment()])
        assert worker.translator.entered.wait(5), "worker never reached the translator"
        worker.set_languages("en", "de")  # GUI thread, while the batch is in flight
        worker.enqueue([segment()])  # the same text again, now to be translated into German
        worker.translator.release.set()

        qtbot.waitUntil(lambda: shown_translation(worker) is not None, timeout=5000)
        assert shown_translation(worker) == f"[de] {SENTENCE}"

    def test_batches_are_handled_in_the_order_they_were_posted(self, qtbot, worker):
        worker.translator.release.clear()
        worker.set_languages("en", "fr")
        worker.start()

        worker.enqueue([segment("First sentence here")])  # in flight when the rest is posted
        assert worker.translator.entered.wait(5)
        worker.enqueue([segment("Second sentence here", x=300)])
        worker.set_languages("en", "de")
        worker.enqueue([segment("Third sentence here", x=600)])
        worker.translator.release.set()

        qtbot.waitUntil(lambda: len(worker.translator.calls) == 3, timeout=5000)
        # Batches posted before the change use the old pair, batches posted after it the new one.
        assert worker.translator.calls == [
            (["First sentence here"], "en", "fr"),
            (["Second sentence here"], "en", "fr"),
            (["Third sentence here"], "en", "de"),
        ]

    def test_change_posted_just_before_stop_is_not_lost(self, worker):
        worker.set_languages("en", "fr")
        worker.start()
        worker.set_languages("en", "de")
        worker.stop()
        worker._update([segment()])
        assert worker.translator.calls[-1][1:] == ("en", "de")


class TestSnapshots:
    def test_latest_regions_is_an_immutable_snapshot(self, worker):
        from langscope.models import RegionSnapshot

        worker.set_languages("en", "fr")
        worker._update([segment()])
        worker._publish()

        snapshot = worker.latest_regions
        assert isinstance(snapshot, tuple)
        assert all(isinstance(region, RegionSnapshot) for region in snapshot)
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot[0].translated_text = "changed by the GUI"

    def test_later_tracking_updates_do_not_change_a_published_snapshot(self, worker):
        worker.set_languages("en", "fr")
        worker._update([segment()])
        worker._publish()
        published = worker.latest_regions
        text, box = published[0].translated_text, published[0].smoothed_bbox

        # The worker keeps tracking: the text moves and the label changes, in place.
        worker._update([segment(x=40)])  # overlaps enough to be matched as the same region
        (live,) = worker._regions.values()
        live.translated_text = "something else"

        assert published[0].translated_text == text
        assert published[0].smoothed_bbox == box
        assert box != tuple(tuple(pt) for pt in live.smoothed_bbox)

    def test_snapshot_carries_what_the_ui_draws(self, worker):
        worker.set_languages("en", "fr")
        worker._update([segment()])
        worker._publish()
        (region,) = worker.latest_regions
        assert region.original_text == SENTENCE
        assert region.translated_text == f"[fr] {SENTENCE}"
        assert region.has_translation
        assert region.confidence == pytest.approx(0.9)
        assert len(region.smoothed_bbox) == 4 and len(region.smoothed_bbox[0]) == 2
        assert region.region_id


class TestMailbox:
    def make(self, max_batches=2):
        from langscope.translation_worker import _Mailbox

        return _Mailbox(max_batches)

    def test_is_first_in_first_out(self):
        box = self.make()
        box.post("languages", ("en", "fr"))
        box.post("segments", [1])
        assert box.get(timeout=0) == ("languages", ("en", "fr"))
        assert box.get(timeout=0) == ("segments", [1])
        assert box.get(timeout=0) is None

    def test_drops_the_oldest_batch_when_too_many_are_waiting(self):
        box = self.make(max_batches=2)
        for n in (1, 2, 3):
            box.post("segments", [n])
        assert [box.get(timeout=0)[1] for _ in range(2)] == [[2], [3]]

    def test_never_drops_a_control_message(self):
        box = self.make(max_batches=2)
        box.post("segments", [1])
        box.post("languages", ("en", "de"))
        box.post("segments", [2])
        box.post("segments", [3])  # over the limit: the oldest batch goes, not the language change
        kinds = [box.get(timeout=0)[0] for _ in range(3)]
        assert kinds == ["languages", "segments", "segments"]

    def test_get_wakes_up_when_a_message_is_posted(self):
        box = self.make()
        threading.Timer(0.05, box.post, args=("segments", [7])).start()
        assert box.get(timeout=5) == ("segments", [7])

    def test_has_pending_and_drain(self):
        box = self.make()
        assert not box.has_pending("languages")
        box.post("languages", ("en", "fr"))
        assert box.has_pending("languages")
        assert box.drain() == [("languages", ("en", "fr"))]
        assert not box.has_pending("languages")
