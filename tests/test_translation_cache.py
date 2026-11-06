"""
TranslationWorker keeps one translation cache: bounded, least recently used first out.

There used to be two: the worker's own, which grew without limit, and a second in translator.py
that could only ever see texts the worker's cache had already missed.
"""

import pytest

import translation_worker
from models import RawSegment, TranslationRegion
from tests.helpers import bbox
from translation_providers import BaseTranslator
from translation_worker import TranslationWorker, _thash


class CountingTranslator(BaseTranslator):
    def __init__(self):
        self.batches = []  # the texts of every translate_many call

    def translate(self, text, from_code, to_code):
        return f"[{to_code}] {text}"

    def translate_many(self, texts, from_code, to_code):
        self.batches.append(list(texts))
        return [self.translate(t, from_code, to_code) for t in texts]

    @property
    def translated(self):
        return [text for batch in self.batches for text in batch]


@pytest.fixture
def worker():
    return TranslationWorker(translator=CountingTranslator())


def translate(worker, *texts):
    """Run texts through the worker's batch step, each in a region of its own."""
    items = []
    regions = []
    for i, text in enumerate(texts):
        region = TranslationRegion(
            bbox=bbox(0, i * 50, 100, i * 50 + 20), original_text=text, translated_text=text
        )
        regions.append(region)
        items.append((_thash(text), RawSegment(bbox=region.bbox, text=text, score=0.9), region))
    worker._batch_translate(items, "en", "fr")
    return regions


def cached_texts(worker, *texts):
    return [t for t in texts if _thash(t) in worker._tx_cache]


class TestCache:
    def test_a_repeated_text_is_translated_once(self, worker):
        translate(worker, "Hello there")
        regions = translate(worker, "Hello there")
        assert worker._translator.translated == ["Hello there"]
        assert regions[0].translated_text == "[fr] Hello there"

    def test_identical_texts_in_one_batch_are_translated_once(self, worker):
        regions = translate(worker, "Same words", "Same words")
        assert worker._translator.translated == ["Same words"]
        assert [r.translated_text for r in regions] == ["[fr] Same words"] * 2

    def test_counts_hits_and_misses(self, worker):
        translate(worker, "one", "two")
        translate(worker, "one", "three")
        assert (worker.cache_hits, worker.cache_misses) == (1, 3)

    def test_is_bounded(self, worker, monkeypatch):
        monkeypatch.setattr(translation_worker, "TX_CACHE_MAX", 3)
        translate(worker, "a one", "b two", "c three", "d four", "e five")
        assert len(worker._tx_cache) == 3
        assert cached_texts(worker, "a one", "b two", "c three", "d four", "e five") == [
            "c three",
            "d four",
            "e five",
        ]

    def test_evicts_the_least_recently_used_entry(self, worker, monkeypatch):
        monkeypatch.setattr(translation_worker, "TX_CACHE_MAX", 3)
        translate(worker, "a one", "b two", "c three")
        translate(worker, "a one")  # a hit: "a one" is now the most recently used
        translate(worker, "d four")  # over the limit: "b two" is the oldest unused entry
        assert cached_texts(worker, "a one", "b two", "c three", "d four") == [
            "a one",
            "c three",
            "d four",
        ]

    def test_an_evicted_text_is_translated_again(self, worker, monkeypatch):
        monkeypatch.setattr(translation_worker, "TX_CACHE_MAX", 1)
        translate(worker, "first text")
        translate(worker, "second text")
        translate(worker, "first text")
        assert worker._translator.translated == ["first text", "second text", "first text"]

    def test_the_default_limit_is_the_old_translator_cache_size(self):
        assert translation_worker.TX_CACHE_MAX == 512

    def test_a_language_change_empties_it(self, worker):
        translate(worker, "Hello there")
        worker.set_languages("en", "de")
        assert len(worker._tx_cache) == 0
