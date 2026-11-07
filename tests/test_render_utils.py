"""
render_utils._wrap_lines: the greedy word-wrap behind every translated label.

Most tests use a fake font in which every character, spaces included, is 10 px wide, so the
width of any line is easy to work out by hand: "Hello world" is 11 characters, 110 px.
"""

import random

import pytest
from PyQt6.QtGui import QFont, QFontMetrics

from langscope.render_utils import _wrap_lines


class FakeMetrics:
    """The one QFontMetrics method _wrap_lines uses."""

    def horizontalAdvance(self, text):
        return 10 * len(text)


@pytest.fixture
def fm():
    return FakeMetrics()


def wrap(text, max_width, fm):
    return _wrap_lines(text, fm, max_width)


class TestWhereLinesBreak:
    def test_text_that_fits_stays_on_one_line(self, fm):
        assert wrap("Hello world", 200, fm) == ["Hello world"]

    def test_a_line_exactly_as_wide_as_the_limit_still_fits(self, fm):
        assert wrap("Hello world", 110, fm) == ["Hello world"]

    def test_one_pixel_less_wraps_the_last_word(self, fm):
        assert wrap("Hello world", 109, fm) == ["Hello", "world"]

    def test_lines_are_filled_as_far_as_they_go(self, fm):
        # 80 px is 8 characters: "aa bb cc" just fits, and " dd" would make it 11.
        assert wrap("aa bb cc dd ee", 80, fm) == ["aa bb cc", "dd ee"]

    def test_every_word_on_its_own_line_when_nothing_fits_beside_another(self, fm):
        assert wrap("one two six", 30, fm) == ["one", "two", "six"]

    def test_many_lines_keep_their_order(self, fm):
        words = [f"w{i:02d}" for i in range(12)]  # 3 characters each
        # 70 px holds two words (3 + 1 + 3 = 7 characters), not three.
        assert wrap(" ".join(words), 70, fm) == [
            "w00 w01",
            "w02 w03",
            "w04 w05",
            "w06 w07",
            "w08 w09",
            "w10 w11",
        ]


class TestWordsThatAreTooLong:
    def test_a_word_wider_than_the_limit_gets_a_line_to_itself_unbroken(self, fm):
        assert wrap("short extraordinarily short", 50, fm) == [
            "short",
            "extraordinarily",
            "short",
        ]

    def test_a_too_long_first_word_does_not_leave_an_empty_line_before_it(self, fm):
        assert wrap("extraordinarily ok", 50, fm) == ["extraordinarily", "ok"]

    def test_a_too_long_last_word_is_kept(self, fm):
        assert wrap("ok extraordinarily", 50, fm) == ["ok", "extraordinarily"]

    def test_a_single_too_long_word_is_one_line(self, fm):
        assert wrap("extraordinarily", 50, fm) == ["extraordinarily"]

    def test_the_word_after_a_too_long_word_starts_a_new_line(self, fm):
        assert wrap("extraordinarily ok fine", 50, fm) == ["extraordinarily", "ok", "fine"]

    @pytest.mark.parametrize("limit", [0, -1, -100])
    def test_a_limit_below_one_word_puts_every_word_on_its_own_line(self, fm, limit):
        assert wrap("one two three", limit, fm) == ["one", "two", "three"]


class TestWhitespace:
    def test_any_run_of_whitespace_becomes_a_single_space(self, fm):
        assert wrap("a   b\n\tc\r\nd", 1000, fm) == ["a b c d"]

    def test_leading_and_trailing_whitespace_is_dropped(self, fm):
        assert wrap("  Hello world \n", 1000, fm) == ["Hello world"]

    def test_wrapped_lines_carry_no_whitespace_at_either_end(self, fm):
        for line in wrap("  aa   bb  cc   dd  ee  ", 50, fm):
            assert line == line.strip()

    @pytest.mark.parametrize("text", ["", "   ", "\n", " \t "])
    def test_text_without_words_is_returned_as_the_single_line(self, fm, text):
        # Never an empty list: the caller sizes its background from len(lines).
        assert wrap(text, 100, fm) == [text]


class TestWhatEveryWrapMustSatisfy:
    WORDS = ["a", "to", "the", "word", "wraps", "somewhat", "extraordinarily", "I", "ok"]

    def make_cases(self):
        rng = random.Random(20240607)  # fixed, so a failure can be reproduced
        for _ in range(500):
            words = [rng.choice(self.WORDS) for _ in range(rng.randint(1, 25))]
            separator = rng.choice([" ", "  ", " \n", "\t"])
            yield separator.join(words), rng.randint(-10, 220)

    def test_no_word_is_lost_added_or_reordered(self, fm):
        for text, limit in self.make_cases():
            assert " ".join(wrap(text, limit, fm)) == " ".join(text.split()), (text, limit)

    def test_a_line_is_either_within_the_limit_or_a_single_word(self, fm):
        for text, limit in self.make_cases():
            for line in wrap(text, limit, fm):
                assert fm.horizontalAdvance(line) <= limit or " " not in line, (text, limit, line)

    def test_a_line_ends_only_where_the_next_word_would_not_have_fit(self, fm):
        for text, limit in self.make_cases():
            lines = wrap(text, limit, fm)
            for line, following in zip(lines, lines[1:]):
                first_of_next = following.split(" ")[0]
                assert fm.horizontalAdvance(line + " " + first_of_next) > limit, (text, limit)


class TestWithRealFontMetrics:
    def test_lines_fit_the_measured_width(self, qapp):
        fm = QFontMetrics(QFont("Segoe UI", 14, QFont.Weight.Bold))
        text = " ".join(["Translated words that need to wrap"] * 5)
        limit = fm.horizontalAdvance("Translated words that need") + 3

        lines = _wrap_lines(text, fm, limit)

        assert len(lines) > 1
        assert " ".join(lines) == text
        for line in lines:
            assert fm.horizontalAdvance(line) <= limit or " " not in line
