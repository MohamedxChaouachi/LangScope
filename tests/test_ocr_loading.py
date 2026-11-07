"""
The OCR model is loaded on the OCR worker thread, never on the GUI thread.

EasyOCR's reader takes seconds to build and, for a language it has not seen, downloads about
200 MB first. Building it in OCRWorker.__init__ or in set_language() froze the window.
"""

import logging
import re
import sys
import threading
import time
import types

import numpy as np
import pytest

from langscope.ocr_providers import BaseOCR
from langscope.ocr_worker import OCRWorker

MAIN_THREAD = threading.get_ident()


class FakeCapture:
    latest_frame = None


class FakeOCR(BaseOCR):
    """Records which thread each call ran on."""

    def __init__(self, fail_load=False):
        self.fail_load = fail_load
        self.language_calls = []  # (language, thread id)
        self.load_calls = []  # thread id
        self.read_calls = 0

    def read(self, frame):
        self.read_calls += 1
        return []

    def set_language(self, from_code):
        self.language_calls.append((from_code, threading.get_ident()))

    def load(self):
        self.load_calls.append(threading.get_ident())
        if self.fail_load:
            raise RuntimeError("download failed")


@pytest.fixture
def started(qtbot):
    """Start workers and always stop them, even if the test fails."""
    workers = []

    def start(ocr=None, capture=None):
        worker = OCRWorker(capture or FakeCapture(), ocr=ocr or FakeOCR())
        statuses = []
        worker.status_changed.connect(statuses.append)
        workers.append(worker)
        return worker, statuses

    yield start
    for worker in workers:
        worker._stop_requested.set()
        worker.wait(5000)


class TestOcrWorkerLoading:
    def test_creating_the_worker_and_choosing_a_language_loads_nothing(self, started):
        ocr = FakeOCR()
        worker, _ = started(ocr)
        worker.set_languages("fr", "en")
        assert ocr.language_calls == [] and ocr.load_calls == []

    def test_model_loads_on_the_worker_thread_and_reports_status(self, qtbot, started):
        ocr = FakeOCR()
        worker, statuses = started(ocr)
        worker.start()
        qtbot.waitUntil(lambda: any("ready" in s for s in statuses), timeout=5000)

        assert len(ocr.load_calls) == 1 and ocr.load_calls[0] != MAIN_THREAD
        assert "Loading OCR model" in statuses[0]
        assert "ready" in statuses[-1]

    def test_a_language_change_loads_the_new_model_on_the_worker_thread(self, qtbot, started):
        ocr = FakeOCR()
        worker, statuses = started(ocr)
        worker.start()
        qtbot.waitUntil(lambda: len(ocr.load_calls) == 1, timeout=5000)

        worker.set_languages("fr", "en")  # from the GUI thread
        # Signals reach the GUI thread through its event loop, so wait for the status itself,
        # not just for the load call that precedes it.
        qtbot.waitUntil(lambda: sum("ready" in s for s in statuses) == 2, timeout=5000)

        assert len(ocr.load_calls) == 2
        assert ocr.language_calls[-1][0] == "fr"
        assert all(thread != MAIN_THREAD for _, thread in ocr.language_calls)
        assert all(thread != MAIN_THREAD for thread in ocr.load_calls)
        assert sum("Loading OCR model" in s for s in statuses) == 2

    def test_ocr_starts_once_the_model_is_ready(self, qtbot, started):
        ocr = FakeOCR()
        capture = FakeCapture()
        capture.latest_frame = np.full((40, 40, 3), 200, np.uint8)
        worker, _ = started(ocr, capture)
        worker.start()
        qtbot.waitUntil(lambda: ocr.read_calls >= 1, timeout=5000)
        assert len(ocr.load_calls) == 1

    def test_changing_only_the_target_language_still_runs_ocr_on_a_static_screen(
        self, qtbot, started
    ):
        # The translator drops its regions on any language change and needs fresh segments to
        # repopulate them. With an unchanged screen, both change gates would skip the frame.
        ocr = FakeOCR()
        capture = FakeCapture()
        capture.latest_frame = np.full((40, 40, 3), 200, np.uint8)
        worker, _ = started(ocr, capture)
        worker.set_languages("en", "fr")
        worker.start()
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)
        time.sleep(0.4)  # a few more passes: the unchanged frame must be skipped
        assert ocr.read_calls == 1

        worker.set_languages("en", "de")  # same OCR language, different target
        qtbot.waitUntil(lambda: ocr.read_calls == 2, timeout=5000)

    def test_unchanged_frames_are_counted_as_skipped(self, qtbot, started):
        ocr = FakeOCR()
        capture = FakeCapture()
        capture.latest_frame = np.full((40, 40, 3), 200, np.uint8)
        worker, _ = started(ocr, capture)
        worker.start()
        qtbot.waitUntil(lambda: worker.frames_skipped >= 3, timeout=5000)
        assert ocr.read_calls == 1 and worker.ocr_calls == 1

    def test_a_failed_load_is_reported_and_ocr_is_not_attempted(self, qtbot, started):
        ocr = FakeOCR(fail_load=True)
        capture = FakeCapture()
        capture.latest_frame = np.full((40, 40, 3), 200, np.uint8)
        worker, statuses = started(ocr, capture)
        worker.start()
        qtbot.waitUntil(lambda: any("failed" in s for s in statuses), timeout=5000)
        time.sleep(0.5)  # several loop iterations: it must not retry the load every time

        assert len(ocr.load_calls) == 1
        assert ocr.read_calls == 0

    def test_a_failed_load_is_retried_after_a_language_change(self, qtbot, started):
        ocr = FakeOCR(fail_load=True)
        worker, statuses = started(ocr)
        worker.start()
        qtbot.waitUntil(lambda: any("failed" in s for s in statuses), timeout=5000)

        ocr.fail_load = False
        worker.set_languages("de", "en")
        qtbot.waitUntil(lambda: any("ready" in s for s in statuses), timeout=5000)
        assert len(ocr.load_calls) == 2


class TestStopBeforeTheThreadRuns:
    def test_a_stop_that_arrives_first_is_not_undone(self, started):
        """run() used to set its running flag back to True, undoing a stop() that came in
        before the thread began, and stop() then waited forever for a thread that never ends."""
        gate = threading.Event()

        class GatedStart(OCRWorker):
            def run(self):
                gate.wait(5)  # the thread exists but has not reached its loop yet
                super().run()

        worker = GatedStart(FakeCapture(), ocr=FakeOCR())
        worker.start()
        stopper = threading.Thread(target=worker.stop)
        stopper.start()
        deadline = time.monotonic() + 5
        while not worker._stop_requested.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)  # until stop() has really been requested
        assert worker._stop_requested.is_set()
        gate.set()  # only now may the thread reach its loop
        stopper.join(5)
        try:
            assert not stopper.is_alive(), "stop() never returned: the thread ignored the stop"
            assert not worker.isRunning()
        finally:
            worker._stop_requested.set()  # do not leave the thread spinning if it failed
            worker.wait(5000)


class _FakeEasyOcr(types.ModuleType):
    """A stand-in for the easyocr package that counts the readers built and where."""

    def __init__(self):
        super().__init__("easyocr")
        self.readers = []  # (langs, thread id)
        outer = self

        class Reader:
            def __init__(self, langs, gpu=False, verbose=False):
                outer.readers.append((tuple(langs), threading.get_ident()))

            def readtext(self, img, detail=1, batch_size=1):
                return []

        self.Reader = Reader


@pytest.fixture
def fake_easyocr(monkeypatch):
    module = _FakeEasyOcr()
    monkeypatch.setitem(sys.modules, "easyocr", module)
    return module


class TestEasyOCRProviderIsLazy:
    def make(self, from_code="en"):
        from langscope.ocr_providers import EasyOCRProvider

        return EasyOCRProvider(from_code)

    def test_construction_and_set_language_build_no_reader(self, fake_easyocr):
        provider = self.make()
        provider.set_language("fr")
        assert fake_easyocr.readers == []

    def test_load_builds_the_reader_for_the_current_language_once(self, fake_easyocr):
        provider = self.make()
        provider.set_language("ja")
        provider.load()
        provider.load()
        assert [langs for langs, _ in fake_easyocr.readers] == [("ja",)]

    def test_read_loads_lazily_for_callers_that_never_call_load(self, fake_easyocr):
        # bench_ocr.py works this way; its warm-up call absorbs the load.
        provider = self.make("en")
        provider.read(np.zeros((30, 30, 3), np.uint8))
        assert [langs for langs, _ in fake_easyocr.readers] == [("en",)]

    def test_the_reader_is_built_on_the_calling_thread(self, fake_easyocr):
        provider = self.make()
        built_on = []
        thread = threading.Thread(target=lambda: (provider.load(), built_on.append(1)))
        thread.start()
        thread.join()
        assert fake_easyocr.readers[0][1] != MAIN_THREAD

    def test_switching_language_loads_a_new_reader_and_switching_back_reuses_one(
        self, fake_easyocr
    ):
        provider = self.make("en")
        provider.load()
        provider.set_language("ja")
        provider.load()
        provider.set_language("en")
        provider.load()
        assert [langs for langs, _ in fake_easyocr.readers] == [("en",), ("ja",)]


class TestGpuOutOfMemory:
    """On the GPU, a frame that runs out of VRAM is retried with smaller recognition batches
    instead of being dropped."""

    class Reader:
        def __init__(self, fail_until_batch):
            self.fail_until_batch = fail_until_batch
            self.batch_sizes = []

        def readtext(self, img, detail=1, batch_size=1):
            self.batch_sizes.append(batch_size)
            # Not "out of memory", so the code under test does not swallow it as a retry.
            assert len(self.batch_sizes) < 12, "the retry loop does not end"
            if batch_size > self.fail_until_batch:
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
            return []

    @pytest.fixture
    def gpu(self, monkeypatch):
        from langscope import ocr_providers

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: torch.cleared.append(1))
        torch.cleared = []
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setattr(ocr_providers.BACKEND, "use_gpu", True)
        monkeypatch.setattr(ocr_providers.BACKEND, "batch_size", 16)
        return torch

    def provider(self):
        from langscope.ocr_providers import EasyOCRProvider

        return EasyOCRProvider("en")

    def test_the_batch_is_halved_until_the_frame_fits(self, gpu, caplog):
        provider = self.provider()
        reader = self.Reader(fail_until_batch=4)
        with caplog.at_level(logging.WARNING, logger="langscope.ocr_providers"):
            provider._readtext(reader, np.zeros((30, 30), np.uint8))
        assert reader.batch_sizes == [16, 8, 4]
        assert provider._batch_size == 4  # and it stays small for the next frames
        assert len(gpu.cleared) == 2
        assert [r.getMessage() for r in caplog.records if "out of memory" in r.getMessage()] == [
            "[OCR] GPU out of memory — retrying with batch_size=8.",
            "[OCR] GPU out of memory — retrying with batch_size=4.",
        ]

    def test_it_gives_up_when_even_one_box_does_not_fit(self, gpu):
        provider = self.provider()
        reader = self.Reader(fail_until_batch=0)
        with pytest.raises(RuntimeError, match="out of memory"):
            provider._readtext(reader, np.zeros((30, 30), np.uint8))
        assert reader.batch_sizes == [16, 8, 4, 2, 1]

    def test_other_errors_are_not_retried(self, gpu):
        class Broken:
            calls = 0

            def readtext(self, img, detail=1, batch_size=1):
                Broken.calls += 1
                raise ValueError("bad image")

        with pytest.raises(ValueError):
            self.provider()._readtext(Broken(), np.zeros((30, 30), np.uint8))
        assert Broken.calls == 1

    def test_on_the_cpu_an_out_of_memory_error_is_not_retried(self, monkeypatch):
        from langscope import ocr_providers

        # The batch size is above 1 here so that it is the missing GPU, not a batch of one,
        # that rules out a retry. (Emptying the CUDA cache means nothing without a GPU.)
        monkeypatch.setattr(ocr_providers.BACKEND, "use_gpu", False)
        monkeypatch.setattr(ocr_providers.BACKEND, "batch_size", 16)
        reader = self.Reader(fail_until_batch=0)
        with pytest.raises(RuntimeError, match="out of memory"):
            self.provider()._readtext(reader, np.zeros((30, 30), np.uint8))
        assert reader.batch_sizes == [16]


class TestOcrLogMessages:
    LOGGER = "langscope.ocr_providers"

    @pytest.fixture(autouse=True)
    def cpu_backend(self, monkeypatch):
        from langscope import ocr_providers

        monkeypatch.setattr(ocr_providers.BACKEND, "use_gpu", False)

    def messages(self, caplog, level=logging.INFO):
        return [r.getMessage() for r in caplog.records if r.levelno >= level]

    def test_loading_a_reader_logs_what_is_loaded_and_how_long_it_took(self, fake_easyocr, caplog):
        from langscope.ocr_providers import EasyOCRProvider

        provider = EasyOCRProvider("fr")
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            provider.load()
        loading, ready = self.messages(caplog)
        assert loading == "[OCR] Loading EasyOCR reader langs=['fr', 'en'] gpu=False fp16=False…"
        assert re.fullmatch(r"\[OCR\] Reader ready in \d+ ms\.", ready)

    def test_the_load_time_is_reported_in_milliseconds(self, fake_easyocr, caplog, monkeypatch):
        from langscope import ocr_providers
        from langscope.ocr_providers import EasyOCRProvider

        ticks = iter([100.0, 102.5])  # the load starts at t=100 s and ends 2.5 s later
        clock = types.SimpleNamespace(perf_counter=lambda: next(ticks))
        monkeypatch.setattr(ocr_providers, "time", clock)
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            EasyOCRProvider("en").load()
        assert "[OCR] Reader ready in 2500 ms." in self.messages(caplog)

    def test_a_cached_reader_logs_nothing(self, fake_easyocr, caplog):
        from langscope.ocr_providers import EasyOCRProvider

        provider = EasyOCRProvider("en")
        provider.load()
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            provider.load()
        assert self.messages(caplog) == []

    def test_fp16_is_logged_when_applied_and_when_it_fails(self, caplog):
        from langscope.ocr_providers import EasyOCRProvider

        class Net:
            def half(self):
                return self

        good = types.SimpleNamespace(detector=Net(), recognizer=Net())

        class BrokenNet:
            def half(self):
                raise RuntimeError("boom")

        bad = types.SimpleNamespace(detector=BrokenNet(), recognizer=None)
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            EasyOCRProvider._apply_fp16(good)
            EasyOCRProvider._apply_fp16(bad)
        applied, failed = [r for r in caplog.records]
        assert applied.levelno == logging.INFO
        assert applied.getMessage() == "[OCR] fp16 applied to detector + recognizer."
        assert failed.levelno == logging.WARNING
        assert failed.getMessage() == "[OCR] fp16 cast failed (boom) — staying fp32."

    def test_confidence_statistics_are_logged_and_the_window_starts_over(self, caplog):
        from langscope.ocr_providers import EasyOCRProvider

        provider = EasyOCRProvider("en")
        for score in (0.4, 0.8):
            provider._update_stats(score)
        provider._call_count = 20
        with caplog.at_level(logging.DEBUG, logger=self.LOGGER):
            provider._log_stats()
            provider._log_stats()  # the window was reset by the first call
        first, second, empty = self.messages(caplog, logging.DEBUG)
        assert first == (
            "[OCR] Confidence stats (last 20 calls): "
            "detections=2  avg=0.60  min=0.40  max=0.80  threshold=0.35"
        )
        assert second == "[OCR] Stats — calls=20  dets=2  avg=0.60  min=0.40  max=0.80"
        assert empty == "[OCR] Stats: 0 detections in last 20 calls"
