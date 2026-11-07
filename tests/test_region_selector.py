"""
The drag-to-select window that covers one screen (ui._ScreenSelectorWindow).

The window receives real QMouseEvent and QKeyEvent objects, offscreen. The offscreen screen has a
devicePixelRatio of 1.0, so a point in the window is the same pixel in native desktop terms.
"""

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent

from tests.helpers import render
from tests.test_screen_utils import FakeScreen
from ui import _ScreenSelectorWindow


@pytest.fixture
def window(qtbot):
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    if screen.devicePixelRatio() != 1.0:
        pytest.skip("these expectations are for a screen with a devicePixelRatio of 1.0")
    w = _ScreenSelectorWindow(screen, "Screen 1 of 1")
    qtbot.addWidget(w)
    w.show()
    return w


@pytest.fixture
def signals(window):
    """Collect what the window emits."""
    picked, cancelled = [], []
    window.region_picked.connect(lambda *args: picked.append(args))
    window.cancelled.connect(lambda: cancelled.append(True))
    return picked, cancelled


def mouse(kind, x, y, button=Qt.MouseButton.LeftButton):
    held = button if kind != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton
    return QMouseEvent(kind, QPointF(x, y), button, held, Qt.KeyboardModifier.NoModifier)


def drag(window, start, end):
    """Press at start, move to end, release there, as a user dragging with the left button."""
    window.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, *start))
    window.mouseMoveEvent(mouse(QEvent.Type.MouseMove, *end))
    window.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, *end))


class TestPickingARegion:
    def test_a_drag_emits_the_region_in_native_pixels(self, window, signals):
        picked, cancelled = signals
        drag(window, (100, 50), (300, 200))
        # (top, left, width, height). A drag from pixel 100 to pixel 300 includes both.
        assert picked == [(50, 100, 201, 151)]
        assert cancelled == []

    @pytest.mark.parametrize(
        "start, end",
        [
            ((100, 50), (300, 200)),  # down and to the right
            ((300, 200), (100, 50)),  # up and to the left
            ((100, 200), (300, 50)),  # up and to the right
            ((300, 50), (100, 200)),  # down and to the left
        ],
    )
    def test_a_drag_picks_the_same_region_in_every_direction(self, window, signals, start, end):
        picked, _ = signals
        drag(window, start, end)
        assert picked == [(50, 100, 201, 151)]

    def test_a_drag_includes_the_pixels_under_both_corners(self, window, signals):
        picked, _ = signals
        drag(window, (10, 10), (19, 19))  # from pixel 10 to pixel 19: ten pixels each way
        drag(window, (19, 19), (10, 10))
        assert picked == [(10, 10, 10, 10), (10, 10, 10, 10)]

    def test_a_drag_that_starts_at_the_top_left_pixel_of_the_screen_counts(self, window, signals):
        # A QPoint at (0, 0) is falsy, so truthiness checks on the drag origin lose this drag.
        picked, cancelled = signals
        drag(window, (0, 0), (120, 80))
        assert picked == [(0, 0, 121, 81)]
        assert cancelled == []

    def test_a_drag_from_the_top_edge_and_from_the_left_edge_count(self, window, signals):
        picked, _ = signals
        drag(window, (0, 40), (100, 100))
        drag(window, (40, 0), (100, 100))
        assert picked == [(40, 0, 101, 61), (0, 40, 61, 101)]

    def test_the_selection_is_cleared_once_the_region_is_picked(self, window, signals):
        picked, _ = signals
        drag(window, (10, 10), (100, 100))
        window.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, 100, 100))
        assert len(picked) == 1  # a stray release does not pick a second region


class TestOnAScaledScreen:
    def test_the_region_is_scaled_and_offset_into_native_pixels(self, window, signals):
        # A 200% screen whose top-left is at (1000, 500) in the native desktop.
        window._screen = FakeScreen(1000, 500, 960, 540, 2.0)
        picked, _ = signals
        drag(window, (100, 50), (300, 200))
        # The 201 x 151 logical pixels picked at (100, 50) are 402 x 302 physical pixels at
        # 1000 + 200 and 500 + 100.
        assert picked == [(600, 1200, 402, 302)]


class TestCancelling:
    @pytest.mark.parametrize("end", [(100, 50), (105, 50), (100, 55), (103, 103)])
    def test_a_drag_no_bigger_than_5_pixels_cancels(self, window, signals, end):
        picked, cancelled = signals
        drag(window, (100, 50), end)
        assert picked == []
        assert cancelled == [True]

    def test_a_drag_just_over_5_pixels_in_both_directions_picks(self, window, signals):
        picked, _ = signals
        drag(window, (100, 50), (105, 55))  # 6 x 6 pixels with both ends included
        assert picked == [(50, 100, 6, 6)]

    def test_a_right_click_cancels(self, window, signals):
        picked, cancelled = signals
        window.mousePressEvent(
            mouse(QEvent.Type.MouseButtonPress, 10, 10, Qt.MouseButton.RightButton)
        )
        assert cancelled == [True]
        assert picked == []

    def test_escape_cancels(self, window, signals):
        _, cancelled = signals
        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        )
        assert cancelled == [True]

    def test_other_keys_do_nothing(self, window, signals):
        _, cancelled = signals
        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        )
        assert cancelled == []

    def test_moving_the_mouse_without_pressing_does_nothing(self, window, signals):
        picked, cancelled = signals
        window.mouseMoveEvent(mouse(QEvent.Type.MouseMove, 50, 50))
        window.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, 50, 50))
        assert picked == [] and cancelled == []

    def test_a_release_with_the_right_button_does_not_pick(self, window, signals):
        picked, _ = signals
        window.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 10, 10))
        window.mouseReleaseEvent(
            mouse(QEvent.Type.MouseButtonRelease, 200, 200, Qt.MouseButton.RightButton)
        )
        assert picked == []


class TestPainting:
    def test_a_selection_looks_different_from_the_idle_window(self, window):
        idle = render(window)
        window.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 100, 50))
        window.mouseMoveEvent(mouse(QEvent.Type.MouseMove, 300, 200))
        selecting = render(window)
        assert selecting != idle

    def test_a_selection_from_the_top_left_pixel_is_painted_too(self, window):
        idle = render(window)
        window.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 0, 0))
        window.mouseMoveEvent(mouse(QEvent.Type.MouseMove, 300, 200))
        assert render(window) != idle

    def test_the_window_remembers_nothing_after_a_cancelled_drag(self, window):
        idle = render(window)
        drag(window, (100, 50), (101, 51))  # too small: cancelled
        assert render(window) == idle

    def test_the_origin_is_a_point(self, window):
        window.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 7, 9))
        assert window._origin == QPoint(7, 9)
