"""
OCRWorker's loop applies FrameChangeGate: a frame is only read by OCR when the screen has
changed enough, and the frames that were skipped are counted.

FrameChangeGate has its own tests (test_frame_change_gate.py). These run the real worker thread
against a fake capture and a fake OCR, to check that the loop obeys the gate.
"""

import pytest

from ocr_worker import OCRWorker
from tests.test_change_detection import MIN_CHANGED_PX, blank, change_pixels
from tests.test_ocr_loading import FakeCapture, FakeOCR


@pytest.fixture
def running(qtbot):
    """Start a worker on the given capture, and always stop it, even if the test fails."""
    workers = []

    def start(capture):
        ocr = FakeOCR()
        worker = OCRWorker(capture, ocr=ocr)
        workers.append(worker)
        worker.start()
        return worker, ocr

    yield start
    for worker in workers:
        worker._stop_requested.set()
        worker.wait(5000)


@pytest.fixture
def capture():
    return FakeCapture()


class CountingCapture:
    """A capture that counts the frames it hands out. The loop takes one frame per pass."""

    def __init__(self, frame):
        self.frame = frame
        self.handed_out = 0

    @property
    def latest_frame(self):
        if self.frame is not None:
            self.handed_out += 1
        return self.frame


def wait_for_skips(qtbot, worker, count):
    """Wait until the loop has skipped `count` more frames than it had when this was called."""
    target = worker.frames_skipped + count
    qtbot.waitUntil(lambda: worker.frames_skipped >= target, timeout=5000)


class TestWorkerAppliesTheGate:
    def test_a_static_screen_is_read_once_and_the_rest_are_skipped(self, qtbot, running, capture):
        capture.latest_frame = blank()
        worker, ocr = running(capture)
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)
        wait_for_skips(qtbot, worker, 3)
        assert ocr.read_calls == 1
        assert worker.ocr_calls == 1

    def test_a_new_array_with_the_same_contents_is_skipped_too(self, qtbot, running, capture):
        # The capture thread builds a new array on every grab.
        capture.latest_frame = blank()
        worker, ocr = running(capture)
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)
        for _ in range(3):
            capture.latest_frame = blank()
            wait_for_skips(qtbot, worker, 1)
        assert ocr.read_calls == 1

    def test_a_tiny_change_is_skipped_and_a_meaningful_one_is_read(self, qtbot, running, capture):
        base = blank()
        capture.latest_frame = base
        worker, ocr = running(capture)
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)

        capture.latest_frame = change_pixels(base, MIN_CHANGED_PX - 1, 100)
        wait_for_skips(qtbot, worker, 2)  # at least one of these is the changed frame
        assert ocr.read_calls == 1

        capture.latest_frame = change_pixels(base, MIN_CHANGED_PX, 100)
        qtbot.waitUntil(lambda: ocr.read_calls == 2, timeout=5000)

    def test_a_read_frame_is_not_read_again(self, qtbot, running, capture):
        base = blank()
        capture.latest_frame = base
        worker, ocr = running(capture)
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)
        capture.latest_frame = change_pixels(base, MIN_CHANGED_PX, 100)
        qtbot.waitUntil(lambda: ocr.read_calls == 2, timeout=5000)
        wait_for_skips(qtbot, worker, 3)
        assert ocr.read_calls == 2

    def test_every_frame_taken_from_the_capture_is_either_read_or_counted_as_skipped(
        self, qtbot, running
    ):
        capture = CountingCapture(blank())
        worker, ocr = running(capture)
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)
        wait_for_skips(qtbot, worker, 3)
        capture.frame = change_pixels(blank(), MIN_CHANGED_PX, 100)
        qtbot.waitUntil(lambda: ocr.read_calls == 2, timeout=5000)
        wait_for_skips(qtbot, worker, 2)

        worker.stop()  # a pass in progress finishes first, so the counts are exact from here

        assert worker.ocr_calls == ocr.read_calls == 2
        assert capture.handed_out == worker.ocr_calls + worker.frames_skipped

    def test_no_frame_yet_is_not_a_skip(self, qtbot, running, capture):
        worker, ocr = running(capture)  # latest_frame is None
        qtbot.waitUntil(lambda: worker.isRunning(), timeout=5000)
        qtbot.wait(300)  # a few loop passes
        assert (ocr.read_calls, worker.frames_skipped) == (0, 0)

    def test_the_first_frame_after_a_restart_is_read_again(self, qtbot, running, capture):
        capture.latest_frame = blank()
        worker, ocr = running(capture)
        qtbot.waitUntil(lambda: ocr.read_calls == 1, timeout=5000)

        worker.stop()
        worker.start()

        # The screen has not changed, but the region may have: the first frame is always read.
        qtbot.waitUntil(lambda: ocr.read_calls == 2, timeout=5000)
