"""
CaptureManager's DXCam loop, against fake dxcam and mss modules.

Covers how failures are logged (once, not twice), that the DXCam camera is released rather than
merely dereferenced, and which backend grabs which region.
"""

import logging
import sys
import threading
import time
import types

import numpy as np
import pytest

from langscope import capture_manager

PRIMARY = {"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": True}
SECOND = {"left": 1920, "top": 0, "width": 1920, "height": 1080}
VIRTUAL_DESKTOP = {"left": 0, "top": 0, "width": 3840, "height": 1080}


class Recorder:
    """Everything the fakes were asked to do, in order."""

    def __init__(self):
        self.events = []

    def names(self):
        return [e[0] for e in self.events]


class FakeCamera:
    def __init__(self, rec, grab_error=None, release_error=None):
        self.rec = rec
        self.grab_error = grab_error
        self.release_error = release_error

    def grab(self, region):
        self.rec.events.append(("dxcam.grab", region))
        if self.grab_error is not None:
            raise self.grab_error
        left, top, right, bottom = region
        return np.zeros((bottom - top, right - left, 3), np.uint8)

    def release(self):
        self.rec.events.append(("dxcam.release",))
        if self.release_error is not None:
            raise self.release_error


class FakeMss:
    monitors = [VIRTUAL_DESKTOP, PRIMARY, SECOND]

    def __init__(self, rec):
        self.rec = rec

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.rec.events.append(("mss.close",))
        return False

    def close(self):
        self.rec.events.append(("mss.close",))

    def grab(self, region):
        self.rec.events.append(("mss.grab", dict(region)))
        return np.zeros((region["height"], region["width"], 4), np.uint8)


@pytest.fixture
def rec():
    return Recorder()


@pytest.fixture
def install(monkeypatch, rec):
    """Put fake dxcam and mss modules in place and return a function that builds a manager."""

    def build(create_error=None, grab_error=None, release_error=None, with_dxcam=True):
        camera = FakeCamera(rec, grab_error, release_error)
        dxcam = types.ModuleType("dxcam")

        def create(output_color="RGB"):
            rec.events.append(("dxcam.create",))
            if create_error is not None:
                raise create_error
            return camera

        dxcam.create = create
        mss_module = types.ModuleType("mss")
        # Only the MSS class: mss.mss() is deprecated, and using it must fail here.
        mss_module.MSS = lambda: FakeMss(rec)
        monkeypatch.setitem(sys.modules, "mss", mss_module)
        if with_dxcam:
            monkeypatch.setitem(sys.modules, "dxcam", dxcam)
        else:
            monkeypatch.setitem(sys.modules, "dxcam", None)  # makes `import dxcam` fail
        manager = capture_manager.CaptureManager()
        manager.camera = camera
        return manager

    return build


@pytest.fixture
def run():
    """Start a manager, wait for a condition, and always stop it again."""
    managers = []

    def go(manager, region, until, timeout=5.0):
        managers.append(manager)
        manager.start(region)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if until():
                return
            time.sleep(0.005)
        raise AssertionError("timed out waiting for the capture loop")

    yield go
    for manager in managers:
        manager.stop()


IN_PRIMARY = {"top": 100, "left": 100, "width": 300, "height": 200}
ON_SECOND = {"top": 100, "left": 2100, "width": 300, "height": 200}


def count(rec, name):
    return rec.names().count(name)


class TestLogging:
    @pytest.mark.parametrize("with_dxcam, backend", [(True, "dxcam"), (False, "mss")])
    def test_the_capture_backend_is_logged(self, install, caplog, with_dxcam, backend):
        with caplog.at_level(logging.INFO, logger="langscope.capture"):
            install(with_dxcam=with_dxcam)
        assert f"[Capture] Backend: {backend}" in [r.getMessage() for r in caplog.records]

    def test_a_dxcam_init_failure_is_logged_once(self, install, run, rec, caplog):
        manager = install(create_error=RuntimeError("no GPU adapter"))
        with caplog.at_level(logging.DEBUG, logger="langscope.capture"):
            run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        problems = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(problems) == 1, [r.getMessage() for r in problems]
        assert "no GPU adapter" in problems[0].getMessage()
        assert manager._backend == "mss"

    def test_a_dxcam_error_is_logged_once_with_its_traceback(self, install, run, rec, caplog):
        manager = install(grab_error=RuntimeError("access lost"))
        with caplog.at_level(logging.DEBUG, logger="langscope.capture"):
            run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        problems = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(problems) == 1, [r.getMessage() for r in problems]
        assert problems[0].exc_info is not None
        assert "access lost" in problems[0].getMessage()


class TestCameraRelease:
    def test_the_camera_is_released_when_capture_stops(self, install, run, rec):
        manager = install()
        run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        manager.stop()
        assert count(rec, "dxcam.release") == 1

    def test_the_camera_is_released_before_falling_back_to_mss(self, install, run, rec):
        """The fallback used to run inside the except block, so the camera was only let go after
        the whole MSS session ended."""
        manager = install(grab_error=RuntimeError("access lost"))
        run(manager, IN_PRIMARY, until=lambda: "mss.grab" in rec.names())
        names = rec.names()
        assert names.index("dxcam.release") < names.index("mss.grab")

    def test_nothing_is_released_if_the_camera_was_never_created(self, install, run, rec):
        manager = install(create_error=RuntimeError("no GPU adapter"))
        run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        manager.stop()
        assert count(rec, "dxcam.release") == 0

    def test_a_failing_release_does_not_kill_the_thread(self, install, run, rec, caplog):
        errors = []
        original_hook = threading.excepthook
        threading.excepthook = lambda args: errors.append(args.exc_value)
        try:
            manager = install(release_error=OSError("device lost"))
            with caplog.at_level(logging.DEBUG, logger="langscope.capture"):
                run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
                manager.stop()
                time.sleep(0.1)
        finally:
            threading.excepthook = original_hook
        assert errors == []
        assert any("device lost" in r.getMessage() for r in caplog.records)


class TestRouting:
    def test_a_region_on_the_primary_monitor_uses_dxcam(self, install, run, rec):
        manager = install()
        run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        assert "dxcam.grab" in rec.names() and "mss.grab" not in rec.names()
        assert manager.latest_frame.shape == (200, 300, 3)

    def test_a_region_on_another_monitor_uses_mss(self, install, run, rec):
        manager = install()
        run(manager, ON_SECOND, until=lambda: manager.latest_frame is not None)
        assert "mss.grab" in rec.names() and "dxcam.grab" not in rec.names()
        assert manager.latest_frame.shape == (200, 300, 3)

    @pytest.mark.parametrize("region", [IN_PRIMARY, ON_SECOND])
    def test_mss_is_given_the_region_as_a_plain_dict_with_the_four_keys(
        self, install, run, rec, region
    ):
        manager = install(with_dxcam=False)
        run(manager, region, until=lambda: "mss.grab" in rec.names())
        (grabbed, *_) = [event[1] for event in rec.events if event[0] == "mss.grab"]
        assert grabbed == region

    def test_a_dxcam_backend_without_dxcam_falls_back_to_mss(self, install, run, rec, caplog):
        # Not reachable through CaptureManager.__init__, which only picks the dxcam backend
        # when it imported; this checks the guard in the loop itself.
        manager = install(with_dxcam=False)
        manager._backend = "dxcam"
        with caplog.at_level(logging.WARNING, logger="langscope.capture"):
            run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        assert manager._backend == "mss"
        assert "mss.grab" in rec.names() and "dxcam.create" not in rec.names()
        assert any("dxcam is not installed" in r.getMessage() for r in caplog.records)

    def test_without_dxcam_everything_uses_mss(self, install, run, rec):
        manager = install(with_dxcam=False)
        run(manager, IN_PRIMARY, until=lambda: manager.latest_frame is not None)
        assert manager._backend == "mss"
        assert "mss.grab" in rec.names() and "dxcam.create" not in rec.names()
