"""
Test configuration for LangScope.

Ensures the project root (the directory containing translation_worker.py and
models.py) is importable when pytest is run from anywhere, and makes Qt run without
a display.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Qt's offscreen platform renders into memory, so widget tests need no display (CI has none).
# This must be set before the first QApplication is created. Set QT_QPA_PLATFORM yourself
# (for example to "windows") to watch the tests run in real windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
