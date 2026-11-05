"""
Test configuration for LangScope.

Ensures the project root (the directory containing translation_worker.py and
models.py) is importable when pytest is run from anywhere.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
