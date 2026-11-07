"""
check_wheel.py — Check that a built wheel contains what the application needs.

    python -m build --wheel
    python scripts/check_wheel.py dist/langscope-*.whl

A wheel that imports fine from the source tree can still be missing its package data or its
entry point, which only shows once somebody installs it. Exit status 0 means the wheel is complete.
"""

from __future__ import annotations

import sys
import zipfile

REQUIRED_FILES = [
    "langscope/__init__.py",
    "langscope/__main__.py",
    "langscope/main.py",
    "langscope/controller.py",
    "langscope/assets/langscope.png",
    "langscope/assets/langscope.ico",
    "langscope/assets/check.png",
]
ENTRY_POINT = "langscope = langscope.main:main"


def problems_in(wheel_path: str) -> list[str]:
    problems: list[str] = []
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        problems += [
            f"missing from the wheel: {name}" for name in REQUIRED_FILES if name not in names
        ]

        entry_points = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
        if not entry_points:
            problems.append("no entry_points.txt: the langscope command would not be installed")
        elif ENTRY_POINT not in wheel.read(entry_points[0]).decode("utf-8"):
            problems.append(f"entry_points.txt does not contain {ENTRY_POINT!r}")

        if not any(n.rsplit("/", 1)[-1] == "LICENSE" and ".dist-info/" in n for n in names):
            problems.append("the LICENSE file is not in the wheel's metadata")
        if any(n.startswith(("tests/", "scripts/")) for n in names):
            problems.append("tests or scripts ended up inside the wheel")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    problems = problems_in(argv[1])
    for problem in problems:
        print(f"FAIL  {problem}")
    if not problems:
        print(f"OK    {argv[1]} has the package, its assets, the entry point and the license")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
