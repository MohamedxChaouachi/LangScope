"""
RenderGate decides, on each render-timer tick, whether there is anything new to draw, and
measures how many frames per second were actually drawn.
"""

import numpy as np
import pytest

from render_gate import RenderGate


class Clock:
    """A clock the test advances by hand."""

    def __init__(self, start=5000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def gate(clock):
    return RenderGate(clock=clock)


def new_frame():
    return np.zeros((4, 4, 3), np.uint8)


class TestWhatChanged:
    def test_nothing_to_draw_without_a_frame(self, gate):
        assert gate.tick(None, ()).render is False

    def test_the_first_frame_is_drawn(self, gate):
        assert gate.tick(new_frame(), ()).render is True

    def test_same_frame_and_same_regions_is_skipped(self, gate):
        frame, regions = new_frame(), ("a",)
        assert gate.tick(frame, regions).render is True
        assert gate.tick(frame, regions).render is False
        assert gate.tick(frame, regions).render is False

    def test_a_new_frame_is_drawn(self, gate):
        regions = ("a",)
        gate.tick(new_frame(), regions)
        assert gate.tick(new_frame(), regions).render is True

    def test_new_regions_are_drawn_even_on_the_same_frame(self, gate):
        frame = new_frame()
        gate.tick(frame, ("a",))
        assert gate.tick(frame, ("a", "b")).render is True

    def test_invalidate_forces_the_next_tick_to_draw(self, gate):
        frame, regions = new_frame(), ("a",)
        gate.tick(frame, regions)
        assert gate.tick(frame, regions).render is False
        gate.invalidate()
        assert gate.tick(frame, regions).render is True
        assert gate.tick(frame, regions).render is False

    def test_a_frame_that_equals_the_last_one_but_is_a_new_object_is_drawn(self, gate):
        # Identity is the test, not pixel equality: comparing pixels would cost as much as the
        # work being saved.
        gate.tick(new_frame(), ())
        assert gate.tick(new_frame(), ()).render is True


class TestFps:
    # Steps of 1/64 s are exact in binary floating point, so a window of 64 steps is exactly
    # one second and the clock never drifts.
    STEP = 1 / 64

    def run(self, gate, clock, ticks, frame_for):
        """Tick `ticks` times, advancing the clock a step after each; return all FPS readings."""
        readings = []
        for i in range(ticks):
            tick = gate.tick(frame_for(i), ("regions",))
            if tick.fps is not None:
                readings.append(tick.fps)
            clock.advance(self.STEP)
        return readings

    def test_no_reading_until_a_full_window_has_passed(self, gate, clock):
        assert self.run(gate, clock, 63, lambda i: new_frame()) == []

    def test_first_reading_is_right_even_though_the_clock_starts_large(self, gate, clock):
        """The window used to start at 0.0, so the first reading was 1 / (time since boot):
        about 0 FPS for the first second, whatever the real rate."""
        readings = self.run(gate, clock, 65, lambda i: new_frame())
        assert readings == [pytest.approx(64.0)]

    def test_counts_drawn_frames_not_ticks(self, gate, clock):
        shared = new_frame()
        # A new frame on every other tick; the ticks in between see the previous frame again.
        frames = {}

        def frame_for(i):
            if i % 2 == 0:
                frames["current"] = new_frame()
            return frames.get("current", shared)

        assert self.run(gate, clock, 65, frame_for) == [pytest.approx(32.0)]

    def test_reports_zero_on_a_static_screen_instead_of_a_stale_number(self, gate, clock):
        frame = new_frame()
        gate.tick(frame, ("regions",))
        clock.advance(self.STEP)
        readings = self.run(gate, clock, 200, lambda i: frame)
        assert readings and readings[-1] == pytest.approx(0.0)

    def test_each_window_starts_fresh(self, gate, clock):
        readings = self.run(gate, clock, 129, lambda i: new_frame())
        assert readings == [pytest.approx(64.0), pytest.approx(64.0)]
