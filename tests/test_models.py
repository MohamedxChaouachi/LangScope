"""
Tests for TranslationRegion.apply_smoothing in models.py.

apply_smoothing runs an Exponential Moving Average on each bbox corner:

    smoothed = alpha * new + (1 - alpha) * old

Nothing is mocked; TranslationRegion has no external dependencies.
"""

import pytest

from models import TranslationRegion


def _square(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _flat(bbox):
    """Flatten a 4-point bbox into a 1-D list so pytest.approx can compare it."""
    return [coord for pt in bbox for coord in pt]


def _make_region(bbox):
    return TranslationRegion(
        bbox=bbox,
        original_text="hello",
        translated_text="bonjour",
    )


class TestApplySmoothing:
    def test_smoothed_bbox_initialised_to_bbox(self):
        bbox = _square(0, 0, 10, 10)
        region = _make_region(bbox)
        assert region.smoothed_bbox == bbox
        # Must be a deep copy, not the same list objects.
        assert region.smoothed_bbox is not bbox
        assert region.smoothed_bbox[0] is not bbox[0]

    def test_same_position_is_stable(self):
        # Feeding the identical bbox back must not move the smoothed bbox.
        bbox = _square(0, 0, 10, 10)
        region = _make_region(bbox)
        region.apply_smoothing(bbox)
        assert _flat(region.smoothed_bbox) == pytest.approx(_flat(bbox))

    def test_ema_formula_single_step(self):
        region = _make_region(_square(0, 0, 0, 0))
        # old = 0, new = 100, alpha default 0.35 -> 35.0
        region.apply_smoothing(_square(100, 100, 100, 100))
        for pt in region.smoothed_bbox:
            assert pt[0] == pytest.approx(35.0)
            assert pt[1] == pytest.approx(35.0)

    def test_custom_alpha(self):
        region = _make_region(_square(0, 0, 0, 0))
        # alpha = 0.5, old = 0, new = 10 -> 5.0
        region.apply_smoothing(_square(10, 10, 10, 10), alpha=0.5)
        for pt in region.smoothed_bbox:
            assert pt[0] == pytest.approx(5.0)
            assert pt[1] == pytest.approx(5.0)

    def test_alpha_one_snaps_to_new(self):
        region = _make_region(_square(0, 0, 0, 0))
        new = _square(20, 30, 40, 50)
        region.apply_smoothing(new, alpha=1.0)
        assert _flat(region.smoothed_bbox) == pytest.approx(_flat(new))

    def test_alpha_zero_freezes(self):
        old = _square(1, 2, 3, 4)
        region = _make_region(old)
        region.apply_smoothing(_square(99, 99, 99, 99), alpha=0.0)
        assert _flat(region.smoothed_bbox) == pytest.approx(_flat(old))

    def test_convergence_over_many_steps(self):
        region = _make_region(_square(0, 0, 0, 0))
        target = _square(100, 100, 100, 100)
        for _ in range(50):
            region.apply_smoothing(target)
        # After many EMA steps the smoothed bbox approaches the target.
        for pt in region.smoothed_bbox:
            assert pt[0] == pytest.approx(100.0, abs=1e-3)
            assert pt[1] == pytest.approx(100.0, abs=1e-3)

    def test_extra_new_corner_is_appended(self):
        # If the new bbox has more points than the current smoothed bbox,
        # the extra corner is appended verbatim.
        region = _make_region(_square(0, 0, 10, 10))
        five_point = _square(0, 0, 10, 10) + [[5, 5]]
        region.apply_smoothing(five_point)
        assert len(region.smoothed_bbox) == 5
        assert region.smoothed_bbox[4] == [5, 5]

    def test_zero_area_bbox_smoothing(self):
        # A degenerate zero-area detection must smooth without error.
        region = _make_region(_square(10, 10, 10, 10))
        region.apply_smoothing(_square(10, 10, 10, 10))
        for pt in region.smoothed_bbox:
            assert pt[0] == pytest.approx(10)
            assert pt[1] == pytest.approx(10)


class TestSnapshot:
    def test_copies_what_is_drawn(self):
        region = _make_region(_square(0, 0, 10, 20))
        region.confidence = 0.8
        snap = region.snapshot()
        assert snap.region_id == region.region_id
        assert snap.original_text == "hello"
        assert snap.translated_text == "bonjour"
        assert snap.confidence == pytest.approx(0.8)
        assert snap.smoothed_bbox == ((0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0))
        assert snap.has_translation

    def test_is_frozen(self):
        snap = _make_region(_square(0, 0, 10, 10)).snapshot()
        with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError
            snap.translated_text = "changed"

    def test_does_not_follow_later_changes_to_the_region(self):
        region = _make_region(_square(0, 0, 10, 10))
        snap = region.snapshot()
        region.apply_smoothing(_square(100, 100, 110, 110), alpha=1.0)
        region.translated_text = "salut"
        assert snap.smoothed_bbox[0] == (0.0, 0.0)
        assert snap.translated_text == "bonjour"

    def test_untranslated_text_has_no_translation(self):
        region = TranslationRegion(
            bbox=_square(0, 0, 10, 10), original_text="same", translated_text="same"
        )
        assert not region.snapshot().has_translation
