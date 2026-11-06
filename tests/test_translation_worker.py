"""
Tests for the pure, Qt-free helpers in translation_worker.py:

    _iou        — bbox Intersection-over-Union
    _text_sim   — difflib SequenceMatcher ratio
    _numeric    — numeric / timestamp / phone filter (uses _TIMESTAMP_RE)

Nothing is mocked: these functions have no external dependencies beyond
numpy (used internally by _iou) and difflib/re from the stdlib.
"""

import pytest

from translation_worker import _TIMESTAMP_RE, _iou, _numeric, _text_sim


def _square(x1, y1, x2, y2):
    """Build a 4-point bbox (clockwise) from an axis-aligned rectangle."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ── _iou ──────────────────────────────────────────────────────────────────────


class TestIou:
    def test_identical_boxes(self):
        box = _square(0, 0, 10, 10)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _square(0, 0, 10, 10)
        b = _square(20, 20, 30, 30)
        assert _iou(a, b) == 0.0

    def test_touching_edges_no_area(self):
        # Boxes share an edge but have zero intersection area.
        a = _square(0, 0, 10, 10)
        b = _square(10, 0, 20, 10)
        assert _iou(a, b) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes overlapping in a 5x10 strip.
        # inter = 50, union = 100 + 100 - 50 = 150 -> 1/3
        a = _square(0, 0, 10, 10)
        b = _square(5, 0, 15, 10)
        assert _iou(a, b) == pytest.approx(50.0 / 150.0)

    def test_contained_box(self):
        # Small box fully inside a large one: iou = area_small / area_large.
        outer = _square(0, 0, 10, 10)
        inner = _square(2, 2, 4, 4)  # area 4, outer area 100
        assert _iou(outer, inner) == pytest.approx(4.0 / 100.0)

    def test_zero_area_bbox(self):
        # A degenerate (zero-area) bbox has no overlap area -> 0.0,
        # and must not raise (guarded against division by zero).
        degenerate = _square(5, 5, 5, 5)
        other = _square(0, 0, 10, 10)
        assert _iou(degenerate, other) == 0.0

    def test_two_zero_area_bboxes(self):
        a = _square(0, 0, 0, 0)
        b = _square(0, 0, 0, 0)
        # No intersection area -> 0.0, no exception.
        assert _iou(a, b) == 0.0

    def test_symmetric(self):
        a = _square(0, 0, 10, 10)
        b = _square(3, 3, 13, 13)
        assert _iou(a, b) == pytest.approx(_iou(b, a))

    def test_result_in_unit_range(self):
        a = _square(0, 0, 8, 6)
        b = _square(2, 1, 9, 7)
        val = _iou(a, b)
        assert 0.0 <= val <= 1.0


# ── _text_sim ──────────────────────────────────────────────────────────────────


class TestTextSim:
    def test_identical_text(self):
        assert _text_sim("hello world", "hello world") == pytest.approx(1.0)

    def test_case_insensitive(self):
        # Comparison lowercases both sides, so case differences are ignored.
        assert _text_sim("Hello", "hello") == pytest.approx(1.0)

    def test_completely_different(self):
        assert _text_sim("abc", "xyz") == 0.0

    def test_both_empty_strings(self):
        # SequenceMatcher treats two empty sequences as a perfect match.
        assert _text_sim("", "") == pytest.approx(1.0)

    def test_one_empty_string(self):
        assert _text_sim("", "nonempty") == 0.0

    def test_partial_similarity_in_range(self):
        val = _text_sim("kitten", "sitting")
        assert 0.0 < val < 1.0

    def test_symmetric_enough(self):
        # Ratio is symmetric for these inputs.
        assert _text_sim("hello there", "hello world") == pytest.approx(
            _text_sim("hello world", "hello there")
        )


# ── _numeric (and the timestamp regex) ─────────────────────────────────────────


class TestNumeric:
    def test_empty_string_is_numeric(self):
        # Empty string has fewer than _MIN_ALPHA letters -> filtered out.
        assert _numeric("") is True

    def test_plain_word_is_not_numeric(self):
        assert _numeric("Hello") is False

    def test_sentence_is_not_numeric(self):
        assert _numeric("The quick brown fox") is False

    def test_short_letters_filtered(self):
        # Fewer than 3 alpha chars -> treated as numeric/noise.
        assert _numeric("ok") is True

    def test_pure_number(self):
        assert _numeric("12345") is True

    def test_timestamp_hh_mm(self):
        assert _numeric("14:32") is True

    def test_timestamp_with_seconds_like(self):
        assert _numeric("9.05") is True

    def test_timestamp_am_pm(self):
        assert _numeric("11:59 PM") is True

    def test_phone_number(self):
        assert _numeric("+1 (555) 123-4567") is True

    def test_phone_number_plain(self):
        assert _numeric("555-123-4567") is True

    def test_date_slash(self):
        assert _numeric("12/31/2025") is True

    def test_weekday_prefixed_time(self):
        assert _numeric("Mon 14:32") is True

    def test_real_sentence_with_number_not_filtered(self):
        # Enough alpha content and not a bare timestamp/phone.
        assert _numeric("Meet me at the cafe") is False


class TestTimestampRegex:
    def test_matches_bare_time(self):
        assert _TIMESTAMP_RE.match("14:32") is not None

    def test_matches_phone(self):
        assert _TIMESTAMP_RE.match("+1 (555) 123-4567") is not None

    def test_does_not_match_word(self):
        assert _TIMESTAMP_RE.match("hello") is None
