"""
main.py — Application entry point.

Reads the configuration, sets up logging and the OCR and translation backends, builds the
application and runs it. Everything that connects the pieces lives in controller.py.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

import ocr_providers
import translator
from config import configure_logging, load_config
from controller import build_controller
from ui import APP_ICON_PATH


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application and return its exit code."""
    configure_logging(load_config())

    # Probe the GPU and set up Argos once, here, after logging is configured so that what they
    # report is shown. Importing the modules does neither.
    ocr_providers.init_ocr_backend()
    translator.init_translation_backend()

    app = QApplication(list(sys.argv if argv is None else argv))

    # App-wide window/taskbar icon (path is script-relative, packaging-safe).
    if os.path.exists(APP_ICON_PATH):
        app.setWindowIcon(QIcon(APP_ICON_PATH))

    controller = build_controller()
    app.aboutToQuit.connect(controller.shutdown)
    controller.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
