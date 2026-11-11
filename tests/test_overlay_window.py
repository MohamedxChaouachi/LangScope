"""OverlayWindow draws each region where the tracker currently has it."""

from langscope.models import TranslationRegion
from langscope.overlay_window import OverlayWindow
from tests.helpers import bbox, painted_columns, render

CAPTURE_RECT = {"top": 0, "left": 0, "width": 600, "height": 120}


def test_overlay_draws_at_the_smoothed_position(qtbot):
    region = TranslationRegion(
        bbox=bbox(10, 20, 150, 60), original_text="hello", translated_text="bonjour"
    )
    # The text moves 300 px to the right. Tracking updates smoothed_bbox on every sighting but
    # never touches bbox, which stays where the region was first seen.
    region.apply_smoothing(bbox(310, 20, 450, 60), alpha=1.0)
    assert region.bbox[0][0] == 10

    overlay = OverlayWindow()
    qtbot.addWidget(overlay)
    overlay.set_active(True)
    overlay.update_regions([region.snapshot()], CAPTURE_RECT)

    columns = painted_columns(render(overlay))
    assert columns is not None, "nothing was painted"
    first, last = columns
    assert first >= 300 and last <= 460, f"painted x={first}..{last}, expected about 310..450"


def test_overlay_paints_regions_in_the_far_third_on_a_scaled_display(qtbot):
    """On a 150% display the 600 px wide window covers 900 native pixels.

    The painter is scaled by 1/dpr so it draws in native pixels, so the clipping surface it
    hands to the text renderer must be native too. With the logical size, anything past native
    x=600 was clipped away."""
    # Native x 700..880 is inside the window (logical 467..587) but past the logical width.
    region = TranslationRegion(
        bbox=bbox(700, 20, 880, 60), original_text="hello", translated_text="bonjour"
    )
    overlay = OverlayWindow()
    qtbot.addWidget(overlay)
    overlay.set_active(True)
    overlay.update_regions([region.snapshot()], CAPTURE_RECT)
    overlay._dpr = 1.5  # what native_region_to_logical reports for a 150% screen
    overlay.update_regions([region.snapshot()], CAPTURE_RECT)

    columns = painted_columns(render(overlay))
    assert columns is not None, "a region in the right third was not painted"
    first, last = columns
    assert first >= 460 and last <= 590, f"painted x={first}..{last}, expected about 467..587"
