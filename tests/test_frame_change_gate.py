"""
FrameChangeGate: the decision "run OCR on this frame, or skip it?".

It remembers the last frame that was sent to OCR and compares each new frame with that one.
"""

import numpy as np
import pytest

import ocr_worker
from ocr_worker import FrameChangeGate
from tests.test_change_detection import MIN_CHANGED_PX, PIXEL_DELTA, blank, change_pixels


@pytest.fixture
def gate():
    return FrameChangeGate()


class TestDecisions:
    def test_the_first_frame_is_always_ocrd(self, gate):
        assert gate.should_ocr(blank()) is True

    def test_an_identical_frame_is_skipped(self, gate):
        frame = blank()
        assert gate.should_ocr(frame) is True
        assert gate.should_ocr(frame) is False
        assert gate.should_ocr(frame) is False

    def test_an_equal_frame_in_a_new_array_is_skipped(self, gate):
        # The capture thread produces a new array on every grab.
        gate.should_ocr(blank())
        assert gate.should_ocr(blank()) is False

    def test_an_identical_frame_is_not_compared_pixel_by_pixel(self, gate, monkeypatch):
        # The hash is the cheap check and comes first; most grabs of a static screen end there.
        compared = []
        monkeypatch.setattr(
            ocr_worker, "_content_changed", lambda frame, prev: compared.append(1) or True
        )
        gate.should_ocr(blank())  # nothing to compare with yet
        gate.should_ocr(blank())
        gate.should_ocr(blank())
        assert compared == []
        assert gate.should_ocr(blank(50)) is True  # a different hash is compared
        assert compared == [1]

    def test_a_tiny_change_is_skipped(self, gate):
        base = blank()
        gate.should_ocr(base)
        # The hash sees it (it moves a sampled pixel), but too few pixels changed to matter.
        assert gate.should_ocr(change_pixels(base, MIN_CHANGED_PX - 1, 100)) is False

    def test_a_faint_change_everywhere_is_skipped(self, gate):
        gate.should_ocr(blank(100))
        assert gate.should_ocr(blank(100 + PIXEL_DELTA)) is False

    def test_a_meaningful_change_triggers_ocr(self, gate):
        base = blank()
        gate.should_ocr(base)
        assert gate.should_ocr(change_pixels(base, MIN_CHANGED_PX, 100)) is True

    def test_after_a_change_the_new_frame_is_the_reference(self, gate):
        base = blank()
        changed = change_pixels(base, MIN_CHANGED_PX, 100)
        gate.should_ocr(base)
        assert gate.should_ocr(changed) is True
        assert gate.should_ocr(changed) is False

    def test_a_frame_of_a_different_size_is_ocrd(self, gate):
        # Selecting a new region changes the frame shape; there is nothing to compare against.
        gate.should_ocr(blank())
        assert gate.should_ocr(blank(shape=(60, 80, 3))) is True


class TestReferenceFrame:
    def test_the_reference_is_the_last_frame_ocrd_not_the_last_one_seen(self, gate):
        """Many tiny changes, each too small on its own, add up against the frame that OCR last
        saw. Comparing with the previous frame instead would let slow changes through unseen."""
        base = blank()
        gate.should_ocr(base)
        step = MIN_CHANGED_PX // 3  # a third of what is needed, per frame
        frames = [change_pixels(base, step * n, 100) for n in (1, 2, 3)]
        assert gate.should_ocr(frames[0]) is False  # 4 pixels vs the reference
        assert gate.should_ocr(frames[1]) is False  # 8 pixels vs the reference
        assert gate.should_ocr(frames[2]) is True  # 12 pixels vs the reference

    def test_a_skipped_frame_does_not_become_the_reference(self, gate):
        base = blank()
        gate.should_ocr(base)
        almost = change_pixels(base, MIN_CHANGED_PX - 1, 100)
        assert gate.should_ocr(almost) is False
        # If `almost` had been adopted, `base` would now be one pixel short of a change from it.
        # It is still compared with `base` itself, and a full change from there triggers.
        assert gate.should_ocr(change_pixels(base, MIN_CHANGED_PX, 100)) is True

    def test_the_reference_is_a_private_copy(self, gate):
        """The capture thread may reuse and overwrite a buffer after handing it out."""
        frame = blank()
        gate.should_ocr(frame)
        frame[:] = 255  # overwritten in place after the gate has seen it
        # A frame that differs from the original by too little to matter, but not from the
        # overwritten buffer. Its hash differs from the original's, so it is compared pixel by
        # pixel with the reference, which must still hold the original contents.
        almost = change_pixels(blank(), MIN_CHANGED_PX - 1, 100)
        assert gate.should_ocr(almost) is False


class TestReset:
    def test_a_reset_makes_the_same_frame_run_again(self, gate):
        frame = blank()
        gate.should_ocr(frame)
        assert gate.should_ocr(frame) is False
        gate.reset()
        assert gate.should_ocr(frame) is True
        assert gate.should_ocr(frame) is False

    def test_a_reset_forgets_the_reference_for_changes_too(self, gate):
        base = blank()
        gate.should_ocr(base)
        gate.reset()
        # Too small a change to trigger against `base`, but there is no `base` any more.
        assert gate.should_ocr(change_pixels(base, 1, 100)) is True


class TestStaticScreenAfterReset:
    def test_both_checks_are_reset_not_just_the_hash(self, gate):
        """A reset that only cleared the hash would leave the pixel comparison in place, and a
        static screen would still be skipped: the second check sees no change from the frame
        OCR last ran on."""
        frame = blank()
        gate.should_ocr(frame)
        gate.reset()
        assert gate.should_ocr(np.array(frame)) is True
