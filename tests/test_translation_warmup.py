"""
The translation worker warms the model up for the current language pair. When the language pack
is not installed yet that cannot succeed, so it backs off instead of retrying every second, and
tries again straight away when told that the pack has finished installing.
"""

import pytest

from langscope import translation_worker
from langscope.translation_providers import BaseTranslator
from langscope.translation_worker import TranslationWorker


class WarmUpTranslator(BaseTranslator):
    def __init__(self, ready=False):
        self.ready = ready
        self.attempts = []  # (from, to) for each warm_up call

    def translate(self, text, from_code, to_code):
        return text

    def warm_up(self, from_code, to_code):
        self.attempts.append((from_code, to_code))
        return self.ready


@pytest.fixture
def translator():
    return WarmUpTranslator()


@pytest.fixture
def worker(translator, no_language_detection):
    w = TranslationWorker(translator=translator)
    w.set_languages("en", "fr")
    return w


def poll(worker, clock, seconds, step=1.0):
    """Run the worker's once-per-loop warm-up check every *step* seconds for *seconds*."""
    for _ in range(int(seconds / step)):
        worker._warm_up_if_needed()
        clock.now += step


class TestBackoff:
    def test_a_missing_pack_is_not_retried_every_second(self, worker, translator, clock):
        poll(worker, clock, 100)
        # Attempts at 0, 1, 3, 7, 15, 31 and 63 s. Retrying every second made it 100.
        assert len(translator.attempts) == 7

    def test_the_delay_doubles_up_to_a_cap(self, worker, translator, clock, monkeypatch):
        monkeypatch.setattr(translation_worker, "WARMUP_RETRY_MAX", 8.0)
        times = []
        original = translator.warm_up
        translator.warm_up = lambda f, t: (times.append(clock.now), original(f, t))[1]
        poll(worker, clock, 60)
        gaps = [b - a for a, b in zip(times, times[1:])]
        # Attempts at 0, 1, 3, 7, 15, 23, ... : the gap doubles, then stays at the cap.
        assert gaps == [1, 2, 4, 8, 8, 8, 8, 8, 8]

    def test_the_first_retry_comes_after_one_second_as_before(self, worker, translator, clock):
        worker._warm_up_if_needed()
        clock.now += 0.5
        worker._warm_up_if_needed()
        assert len(translator.attempts) == 1
        clock.now += 0.5
        worker._warm_up_if_needed()
        assert len(translator.attempts) == 2

    def test_a_successful_warm_up_is_not_repeated(self, worker, translator, clock):
        translator.ready = True
        poll(worker, clock, 30)
        assert translator.attempts == [("en", "fr")]

    def test_an_exception_counts_as_a_failed_attempt(self, worker, translator, clock):
        def boom(f, t):
            translator.attempts.append((f, t))
            raise RuntimeError("model not found")

        translator.warm_up = boom
        poll(worker, clock, 100)
        assert len(translator.attempts) == 7

    def test_same_language_never_warms_up(self, translator, clock, no_language_detection):
        w = TranslationWorker(translator=translator)
        w.set_languages("fr", "fr")
        poll(w, clock, 10)
        assert translator.attempts == []


class TestRetryTriggers:
    def test_a_language_change_resets_the_backoff(self, worker, translator, clock):
        poll(worker, clock, 40)  # now waiting 16+ s between attempts
        before = len(translator.attempts)
        worker.set_languages("en", "de")
        worker._warm_up_if_needed()
        assert translator.attempts[before:] == [("en", "de")]

    def test_retry_after_package_setup_tries_immediately(self, worker, translator, clock):
        poll(worker, clock, 40)
        before = len(translator.attempts)
        worker._warm_up_if_needed()
        assert len(translator.attempts) == before  # still backing off

        translator.ready = True  # the pack finished installing
        worker.retry_warm_up()
        worker._warm_up_if_needed()  # no waiting for the backoff to expire
        assert len(translator.attempts) == before + 1
        worker._warm_up_if_needed()
        assert len(translator.attempts) == before + 1  # warmed up: done

    def test_retry_reaches_a_running_worker_without_waiting(self, qtbot, translator, monkeypatch):
        monkeypatch.setattr(translation_worker, "WARMUP_RETRY_MIN", 30.0)
        monkeypatch.setattr(translation_worker, "_already_target", lambda text, to_code: False)
        worker = TranslationWorker(translator=translator)
        worker.set_languages("en", "fr")
        worker.start()
        try:
            qtbot.waitUntil(lambda: len(translator.attempts) == 1, timeout=5000)
            qtbot.wait(300)
            assert len(translator.attempts) == 1  # backing off for 30 s

            translator.ready = True
            worker.retry_warm_up()  # from the GUI thread, as main.py does when setup finishes
            qtbot.waitUntil(lambda: len(translator.attempts) == 2, timeout=5000)
            assert worker._warmed_pair == ("en", "fr")
        finally:
            worker.stop()
