"""
screen_utils: conversions between Qt's logical pixels and the native pixels that mss and dxcam
use, on screens with a devicePixelRatio of 1.0, 1.25 and 2.0.

The tests use fake screens, so they do not depend on the monitors of the machine they run on.
A real QScreen reports its top-left corner in native desktop coordinates and its size in
logical pixels, which is what screen_utils is written for and what FakeScreen reproduces.
"""

import pytest
from PyQt6.QtCore import QRect

import screen_utils
from screen_utils import (
    local_rect_to_native,
    native_region_to_logical,
    screen_at_native_point,
    screen_index,
    screen_native_rect,
)

DPRS = [1.0, 1.25, 2.0]


class FakeScreen:
    """The two QScreen methods screen_utils reads."""

    def __init__(self, x, y, logical_width, logical_height, dpr):
        self._geometry = (x, y, logical_width, logical_height)
        self._dpr = dpr

    def geometry(self):
        return QRect(*self._geometry)

    def devicePixelRatio(self):
        return self._dpr


def screen(x=0, y=0, native_width=1920, native_height=1080, dpr=1.0):
    """A screen with this many physical pixels, as Qt reports it at the given scaling."""
    return FakeScreen(x, y, round(native_width / dpr), round(native_height / dpr), dpr)


def as_tuple(rect):
    return rect.x(), rect.y(), rect.width(), rect.height()


def region(left, top, width, height):
    return {"left": left, "top": top, "width": width, "height": height}


@pytest.fixture
def desktop(monkeypatch):
    """Make QGuiApplication.screens() return the given screens. The first one is the primary."""

    def install(*screens):
        class FakeQGuiApplication:
            @staticmethod
            def screens():
                return list(screens)

            @staticmethod
            def primaryScreen():
                return screens[0]

        monkeypatch.setattr(screen_utils, "QGuiApplication", FakeQGuiApplication)
        return screens

    return install


class TestScreenNativeRect:
    @pytest.mark.parametrize("dpr", DPRS)
    def test_the_size_is_in_physical_pixels_and_the_origin_is_kept(self, dpr):
        rect = screen_native_rect(screen(x=-1920, y=120, dpr=dpr))
        assert as_tuple(rect) == (-1920, 120, 1920, 1080)

    def test_a_size_that_does_not_divide_evenly_is_rounded(self):
        # 1093 x 615 logical pixels at 125% are 1366.25 x 768.75 physical pixels.
        rect = screen_native_rect(FakeScreen(0, 0, 1093, 615, 1.25))
        assert (rect.width(), rect.height()) == (1366, 769)


class TestScreenAtNativePoint:
    """Two monitors side by side: a 1080p one at 100% and a 1440p one at 200%, to its right."""

    @pytest.fixture
    def screens(self, desktop):
        left = screen(x=0, native_width=1920, native_height=1080, dpr=1.0)
        right = screen(x=1920, native_width=2560, native_height=1440, dpr=2.0)
        desktop(left, right)
        return left, right

    @pytest.mark.parametrize("point", [(0, 0), (960, 540), (1919, 1079)])
    def test_points_inside_the_left_screen(self, screens, point):
        assert screen_at_native_point(*point) is screens[0]

    @pytest.mark.parametrize("point", [(1920, 0), (3000, 700), (4479, 1439)])
    def test_points_inside_the_right_screen(self, screens, point):
        # The right screen is 1280 logical pixels wide, but 2560 native pixels.
        assert screen_at_native_point(*point) is screens[1]

    def test_the_last_native_pixel_of_a_scaled_screen_belongs_to_it(self, screens):
        assert screen_at_native_point(4479, 1439) is screens[1]
        assert screen_at_native_point(4480, 1439) is screens[0]  # off the desktop: the primary

    def test_with_no_screens_at_all_it_says_so(self, monkeypatch):
        class NoScreens:
            @staticmethod
            def screens():
                return []

            @staticmethod
            def primaryScreen():
                return None

        monkeypatch.setattr(screen_utils, "QGuiApplication", NoScreens)
        with pytest.raises(RuntimeError, match="No screen"):
            screen_at_native_point(10, 10)

    def test_a_point_outside_every_screen_falls_back_to_the_primary(self, screens):
        assert screen_at_native_point(9000, 9000) is screens[0]
        assert screen_at_native_point(-1, -1) is screens[0]

    def test_a_screen_to_the_left_has_negative_coordinates(self, desktop):
        primary = screen(x=0, native_width=1920, native_height=1080)
        left = screen(x=-1280, y=-100, native_width=1280, native_height=1024)
        desktop(primary, left)
        assert screen_at_native_point(-1, 500) is left
        assert screen_at_native_point(-1280, -100) is left
        assert screen_at_native_point(0, 500) is primary


class TestScreenIndex:
    def test_screens_are_numbered_from_one(self, desktop):
        a, b, c = desktop(screen(), screen(x=1920), screen(x=3840))
        assert [screen_index(s) for s in (a, b, c)] == [1, 2, 3]

    def test_a_screen_that_is_not_attached_is_zero(self, desktop):
        desktop(screen(), screen(x=1920))
        assert screen_index(screen(x=5000)) == 0


class TestLocalRectToNative:
    """A rect dragged on a screen-sized overlay: logical, relative to the screen's top-left."""

    LOCAL = QRect(100, 40, 400, 80)  # left, top, width, height

    @pytest.mark.parametrize(
        "dpr, expected",  # expected is (top, left, width, height)
        [
            (1.0, (40, 100, 400, 80)),
            (1.25, (50, 125, 500, 100)),
            (2.0, (80, 200, 800, 160)),
        ],
    )
    def test_position_and_size_are_scaled(self, dpr, expected):
        assert local_rect_to_native(screen(dpr=dpr), self.LOCAL) == expected

    @pytest.mark.parametrize(
        "dpr, expected",
        [
            (1.0, (-160, 2020, 400, 80)),
            (1.25, (-150, 2045, 500, 100)),
            (2.0, (-120, 2120, 800, 160)),
        ],
    )
    def test_the_screens_native_origin_is_added_and_not_scaled(self, dpr, expected):
        offset_screen = screen(x=1920, y=-200, dpr=dpr)
        assert local_rect_to_native(offset_screen, self.LOCAL) == expected

    def test_fractional_results_are_rounded_to_the_nearest_pixel(self):
        # 3 * 1.25 = 3.75 and 7 * 1.25 = 8.75
        assert local_rect_to_native(screen(dpr=1.25), QRect(3, 3, 7, 7)) == (4, 4, 9, 9)

    @pytest.mark.parametrize("dpr", DPRS)
    def test_an_empty_rect_is_one_pixel_so_that_it_can_still_be_captured(self, dpr):
        _, _, width, height = local_rect_to_native(screen(dpr=dpr), QRect(10, 10, 0, 0))
        assert (width, height) == (1, 1)

    def test_the_result_is_ordered_top_left_width_height(self):
        # region_picked is emitted from this tuple: (top, left, width, height).
        top, left, width, height = local_rect_to_native(screen(), QRect(11, 22, 33, 44))
        assert (top, left, width, height) == (22, 11, 33, 44)


class TestNativeRegionToLogical:
    @pytest.mark.parametrize(
        "dpr, native",
        [
            (1.0, region(left=100, top=40, width=400, height=80)),
            (1.25, region(left=125, top=50, width=500, height=100)),
            (2.0, region(left=200, top=80, width=800, height=160)),
        ],
    )
    def test_a_native_region_becomes_the_same_logical_rect_at_every_scaling(
        self, desktop, dpr, native
    ):
        (only,) = desktop(screen(dpr=dpr))
        rect, ratio, found = native_region_to_logical(native)
        assert as_tuple(rect) == (100, 40, 400, 80)
        assert ratio == dpr
        assert found is only

    def test_a_region_on_the_second_screen_is_placed_relative_to_that_screen(self, desktop):
        first = screen(x=0, native_width=1920, native_height=1080, dpr=1.0)
        second = screen(x=1920, native_width=2560, native_height=1440, dpr=2.0)
        desktop(first, second)
        # 200 native pixels in from the second screen's left edge is 100 logical pixels.
        rect, ratio, found = native_region_to_logical(region(2120, 80, 800, 160))
        assert as_tuple(rect) == (2020, 40, 400, 80)
        assert ratio == 2.0
        assert found is second

    def test_a_region_on_the_first_screen_is_not_scaled_by_the_second(self, desktop):
        first = screen(x=0, native_width=1920, native_height=1080, dpr=1.0)
        desktop(first, screen(x=1920, native_width=2560, native_height=1440, dpr=2.0))
        rect, ratio, found = native_region_to_logical(region(100, 100, 400, 200))
        assert as_tuple(rect) == (100, 100, 400, 200)
        assert ratio == 1.0
        assert found is first

    @pytest.mark.parametrize(
        "left, width, on_second, expected",
        [
            # 420 native pixels on the first screen, 180 on the second; the centre, x=1800, is on
            # the first, and only the right edge is on the second.
            (1500, 600, False, (1500, 100, 600, 200)),
            # 120 pixels on the first screen, 280 on the second; the centre, x=2000, is on the
            # second, and only the left edge is on the first.
            (1800, 400, True, (1860, 50, 200, 100)),
        ],
    )
    def test_the_screen_holding_the_centre_decides_side_by_side(
        self, desktop, left, width, on_second, expected
    ):
        first = screen(x=0, native_width=1920, native_height=1080, dpr=1.0)
        second = screen(x=1920, native_width=2560, native_height=1440, dpr=2.0)
        desktop(first, second)
        rect, _, found = native_region_to_logical(region(left, 100, width, 200))
        assert found is (second if on_second else first)
        assert as_tuple(rect) == expected

    @pytest.mark.parametrize(
        "top, height, on_lower, expected",
        [
            # The centre, y=1050, is on the upper screen; only the bottom edge is on the lower.
            (800, 500, False, (100, 800, 400, 500)),
            # The centre, y=1100, is on the lower screen; only the top edge is on the upper.
            (900, 400, True, (50, 990, 200, 200)),
        ],
    )
    def test_the_screen_holding_the_centre_decides_when_stacked(
        self, desktop, top, height, on_lower, expected
    ):
        upper = screen(x=0, y=0, native_width=1920, native_height=1080, dpr=1.0)
        lower = screen(x=0, y=1080, native_width=3840, native_height=2160, dpr=2.0)
        desktop(upper, lower)
        rect, _, found = native_region_to_logical(region(100, top, 400, height))
        assert found is (lower if on_lower else upper)
        assert as_tuple(rect) == expected

    def test_a_region_off_every_screen_is_placed_on_the_primary(self, desktop):
        first = screen(x=0, dpr=1.25)
        desktop(first, screen(x=1920, dpr=2.0))
        rect, ratio, found = native_region_to_logical(region(9000, 9000, 500, 100))
        assert found is first
        assert ratio == 1.25
        assert as_tuple(rect) == (7200, 7200, 400, 80)

    @pytest.mark.parametrize("dpr", DPRS)
    def test_a_tiny_region_is_at_least_one_logical_pixel(self, desktop, dpr):
        desktop(screen(dpr=dpr))
        rect, _, _ = native_region_to_logical(region(10, 10, 1, 1))
        assert (rect.width(), rect.height()) == (1, 1)

    def test_a_screen_reporting_no_scaling_is_treated_as_unscaled(self, desktop):
        desktop(FakeScreen(0, 0, 1920, 1080, 0.0))
        rect, ratio, _ = native_region_to_logical(region(100, 40, 400, 80))
        assert ratio == 1.0
        assert as_tuple(rect) == (100, 40, 400, 80)

    def test_a_region_is_positioned_from_the_screens_native_origin(self, desktop):
        desktop(screen(x=-2560, y=-300, native_width=2560, native_height=1440, dpr=2.0))
        # 400 native pixels right of and 100 below the screen's top-left corner.
        rect, _, _ = native_region_to_logical(region(-2160, -200, 600, 200))
        assert as_tuple(rect) == (-2360, -250, 300, 100)


class TestRoundTrip:
    """A rectangle picked on screen is stored in native pixels, then drawn again from them."""

    RECTS = [
        QRect(x, y, w, h)
        for x in (0, 1, 7, 100, 333)
        for y in (0, 3, 50, 411)
        for w in (1, 2, 5, 150, 400)
        for h in (1, 3, 80, 301)
    ]

    @pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    @pytest.mark.parametrize("origin", [(0, 0), (1920, 0), (-3840, 300)])
    def test_the_same_rect_comes_back(self, desktop, dpr, origin):
        (picked_on,) = desktop(
            screen(x=origin[0], y=origin[1], native_width=3840, native_height=2160, dpr=dpr)
        )
        for local in self.RECTS:
            top, left, width, height = local_rect_to_native(picked_on, local)
            rect, ratio, found = native_region_to_logical(region(left, top, width, height))
            back = (rect.x() - origin[0], rect.y() - origin[1], rect.width(), rect.height())
            assert back == as_tuple(local), (dpr, origin, as_tuple(local))
            assert ratio == dpr
            assert found is picked_on

    def test_each_screen_of_a_mixed_dpi_desktop_gets_its_own_rect_back(self, desktop):
        first = screen(x=0, native_width=1920, native_height=1080, dpr=1.0)
        second = screen(x=1920, native_width=2560, native_height=1440, dpr=2.0)
        desktop(first, second)
        for picked_on in (first, second):
            origin = picked_on.geometry()
            local = QRect(40, 30, 321, 123)
            top, left, width, height = local_rect_to_native(picked_on, local)
            rect, _, found = native_region_to_logical(region(left, top, width, height))
            assert found is picked_on
            assert (rect.x() - origin.x(), rect.y() - origin.y()) == (40, 30)
            assert (rect.width(), rect.height()) == (321, 123)
