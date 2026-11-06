"""
Change detection in ocr_worker.py: how the OCR worker decides that the screen has changed
enough to be worth running OCR again.

Two checks guard OCR. A cheap hash of a coarse sample of the frame (_hash_frame) rules out an
identical frame. Then _content_changed counts pixels that changed noticeably, measured at half
resolution, so that compression noise and a blinking caret don't trigger OCR but a changed line
of text does.
"""

import numpy as np
import pytest

import ocr_worker
from ocr_worker import MIN_CHANGED_PX, PIXEL_DELTA, _content_changed, _hash_frame

HEIGHT, WIDTH = 100, 120
HALF_COLUMNS = WIDTH // 2


def blank(value=100, shape=(HEIGHT, WIDTH, 3)):
    return np.full(shape, value, np.uint8)


def change_pixels(frame, count, delta):
    """A copy of *frame* in which *count* pixels of the half-resolution grid differ by *delta*.

    _content_changed compares every second row and column, so these are the pixels it can see.
    They are laid out row by row on that grid.
    """
    out = frame.copy()
    for i in range(count):
        y, x = 2 * (i // HALF_COLUMNS), 2 * (i % HALF_COLUMNS)
        out[y, x] = np.clip(frame[y, x].astype(int) + delta, 0, 255)
    return out


class TestTuningConstants:
    def test_keep_their_documented_values(self):
        # These decide how sensitive OCR is. Change them on purpose, and update this test.
        assert (ocr_worker.OCR_FPS, PIXEL_DELTA, MIN_CHANGED_PX) == (8.0, 40, 12)


class TestContentChanged:
    def test_an_identical_frame_has_not_changed(self):
        frame = blank()
        assert _content_changed(frame, frame.copy()) is False

    def test_enough_noticeably_changed_pixels_count_as_a_change(self):
        before = blank()
        assert _content_changed(change_pixels(before, MIN_CHANGED_PX, 100), before) is True

    def test_one_pixel_fewer_than_the_minimum_does_not(self):
        before = blank()
        assert _content_changed(change_pixels(before, MIN_CHANGED_PX - 1, 100), before) is False

    def test_a_change_of_exactly_the_per_pixel_delta_is_ignored(self):
        before = blank()
        everywhere = np.clip(before.astype(int) + PIXEL_DELTA, 0, 255).astype(np.uint8)
        assert _content_changed(everywhere, before) is False

    def test_a_change_just_over_the_per_pixel_delta_counts(self):
        before = blank()
        everywhere = np.clip(before.astype(int) + PIXEL_DELTA + 1, 0, 255).astype(np.uint8)
        assert _content_changed(everywhere, before) is True

    def test_a_faint_change_across_the_whole_frame_is_ignored(self):
        # Compression noise or a gentle gradient: every pixel moves, none by much.
        before = blank()
        assert _content_changed(blank(130), before) is False

    def test_darker_and_brighter_both_count(self):
        before = blank(150)
        assert _content_changed(change_pixels(before, MIN_CHANGED_PX, +100), before) is True
        assert _content_changed(change_pixels(before, MIN_CHANGED_PX, -100), before) is True

    def test_uint8_wraparound_does_not_hide_a_change(self):
        # In uint8, 10 - 250 wraps to 16, which would look like a small change.
        dark, bright = blank(10), blank(250)
        assert _content_changed(bright, dark) is True
        assert _content_changed(dark, bright) is True

    def test_a_single_channel_is_enough(self):
        before = blank()
        after = before.copy()
        after[::2, ::2, 1] += PIXEL_DELTA + 20  # only the green channel
        assert _content_changed(after, before) is True

    def test_is_measured_at_half_resolution(self):
        before = blank()
        # An aligned 2x2 block holds a single pixel of the half-resolution grid...
        small = before.copy()
        small[10:12, 10:12] = 255
        assert _content_changed(small, before) is False
        # ...and a 4x4 block, which is 16 pixels at full resolution, holds only 4 of them.
        medium = before.copy()
        medium[10:14, 10:14] = 255
        assert _content_changed(medium, before) is False
        # A 6x8 block is the 12 grid pixels needed.
        block = before.copy()
        block[10:16, 10:18] = 255
        assert _content_changed(block, before) is True


class TestHashFrame:
    def test_equal_frames_hash_equal(self):
        assert _hash_frame(blank()) == _hash_frame(blank())

    def test_is_an_md5_hex_digest(self):
        digest = _hash_frame(blank())
        assert len(digest) == 32 and int(digest, 16) >= 0

    def test_a_change_at_a_sampled_pixel_changes_the_hash(self):
        # Every fourth row and column is hashed; (0, 0) is one of them.
        changed = blank()
        changed[0, 0] = 0
        assert _hash_frame(changed) != _hash_frame(blank())

    @pytest.mark.parametrize("y, x", [(0, 4), (4, 0), (4, 4), (8, 12), (96, 116)])
    def test_every_fourth_row_and_column_is_sampled(self, y, x):
        changed = blank()
        changed[y, x] = 0
        assert _hash_frame(changed) != _hash_frame(blank())

    @pytest.mark.parametrize("y, x", [(1, 1), (2, 2), (3, 3), (0, 2), (2, 0), (5, 7)])
    def test_a_pixel_between_the_sampled_ones_is_not_looked_at(self, y, x):
        # The hash is a cheap pre-check on 1/16 of the pixels. A real change to the screen
        # touches far more than that, and _content_changed looks at it properly.
        changed = blank()
        changed[y, x] = 0
        assert _hash_frame(changed) == _hash_frame(blank())

    def test_a_non_contiguous_view_hashes_like_its_copy(self):
        # CaptureManager hands over np.array(bgra)[..., :3], a view with gaps between pixels.
        bgra = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
        bgra[..., 0] = 7
        view = bgra[..., :3]
        assert not view.flags["C_CONTIGUOUS"]
        assert _hash_frame(view) == _hash_frame(np.ascontiguousarray(view))

    @pytest.mark.parametrize("shape", [(40, 40, 3), (HEIGHT, WIDTH, 3), (101, 77, 3)])
    def test_frames_of_different_sizes_do_not_collide(self, shape):
        assert _hash_frame(blank(shape=shape)) != _hash_frame(blank(shape=(8, 8, 3)))
