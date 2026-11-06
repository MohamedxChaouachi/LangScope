"""LiveFeedWidget.update_frame keeps its own copy of the frame."""

import numpy as np
from PyQt6.QtGui import QColor

from ui import LiveFeedWidget


def test_the_pixmap_does_not_share_memory_with_the_frame(qtbot):
    """update_frame wraps the numpy buffer in a QImage without copying and QPixmap.fromImage
    then copies it. A second copy in between is redundant, but only because the pixmap really
    is independent of the buffer, which is what this checks."""
    feed = LiveFeedWidget()
    qtbot.addWidget(feed)
    frame = np.zeros((6, 8, 3), np.uint8)
    frame[:, :] = (10, 20, 30)  # BGR

    feed.update_frame(frame, [], 8, 6)
    frame[:, :] = 255  # the capture thread reuses and overwrites its buffers

    pixel = feed._pixmap.toImage().pixelColor(0, 0)
    assert pixel == QColor(30, 20, 10)  # RGB: the BGR input, converted
    assert (feed._pixmap.width(), feed._pixmap.height()) == (8, 6)
