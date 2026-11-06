"""OverlayWindow draws each region where the tracker currently has it."""

from models import TranslationRegion
from overlay_window import OverlayWindow
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
    overlay.update_regions([region], CAPTURE_RECT)

    columns = painted_columns(render(overlay))
    assert columns is not None, "nothing was painted"
    first, last = columns
    assert first >= 300 and last <= 460, f"painted x={first}..{last}, expected about 310..450"
