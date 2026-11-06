"""
A stand-in for argostranslate and ctranslate2.

translator.py imports both at module level and creates the user's Argos data directory as soon
as it is imported. That makes it impossible to import in CI, which installs no ML packages, and
unsafe to import in a test on a developer machine. FakeArgos puts small fake modules into
sys.modules, backed by plain Python state that a test can read and change, so the real
translator.py can be imported and exercised against them.

Use the `fake_argos` and `translator_module` fixtures from conftest.py rather than this module
directly.
"""

from __future__ import annotations

import pathlib
import sys
import types


class FakePackage:
    """Stands in for argostranslate.package.Package and AvailablePackage."""

    def __init__(self, owner: FakeArgos, from_code: str, to_code: str):
        self._owner = owner
        self.from_code = from_code
        self.to_code = to_code
        self.type = "translate"
        self.downloads = 0
        self.download_error: Exception | None = None

    def download(self) -> pathlib.Path:
        self.downloads += 1
        if self.download_error is not None:
            raise self.download_error
        path = pathlib.Path(f"{self.from_code}_{self.to_code}.argosmodel")
        self._owner._downloaded[path] = self
        return path


class FakeArgos:
    """What is installed, what the package index offers, and what the code under test did."""

    def __init__(self, data_dir: pathlib.Path):
        self.installed: list[FakePackage] = []
        self.available: list[FakePackage] = []
        self.index_updates = 0
        self.index_error: Exception | None = None
        self.installs: list[tuple[str, str]] = []  # pairs installed, in order
        self._downloaded: dict[pathlib.Path, FakePackage] = {}
        self._build_modules(data_dir)

    # ── Arranging a scenario ─────────────────────────────────────────────────

    def add_installed(self, from_code: str, to_code: str) -> FakePackage:
        package = FakePackage(self, from_code, to_code)
        self.installed.append(package)
        return package

    def add_available(self, from_code: str, to_code: str) -> FakePackage:
        package = FakePackage(self, from_code, to_code)
        self.available.append(package)
        return package

    @property
    def installed_pairs(self) -> set[tuple[str, str]]:
        return {(p.from_code, p.to_code) for p in self.installed}

    @property
    def downloaded_pairs(self) -> list[tuple[str, str]]:
        return [(p.from_code, p.to_code) for p in self.available if p.downloads]

    # ── The fake modules ─────────────────────────────────────────────────────

    def _build_modules(self, data_dir: pathlib.Path) -> None:
        root = types.ModuleType("argostranslate")
        root.__path__ = []  # marks it as a package

        settings = types.ModuleType("argostranslate.settings")
        settings.data_dir = data_dir
        settings.package_data_dir = str(data_dir / "packages")
        settings.device = "cpu"
        settings.compute_type = "auto"
        settings.inter_threads = 1
        settings.intra_threads = 0
        settings.batch_size = 32
        settings.beam_size = 4

        package = types.ModuleType("argostranslate.package")
        package.get_installed_packages = lambda: list(self.installed)
        package.get_available_packages = lambda: list(self.available)
        package.update_package_index = self._update_package_index
        package.install_from_path = self._install_from_path

        translate = types.ModuleType("argostranslate.translate")
        translate.get_installed_languages = lambda: []

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 0
        ctranslate2.get_supported_compute_types = lambda device: {"default"}

        root.settings, root.package, root.translate = settings, package, translate
        self.settings = settings
        self.modules = {
            "argostranslate": root,
            "argostranslate.settings": settings,
            "argostranslate.package": package,
            "argostranslate.translate": translate,
            "ctranslate2": ctranslate2,
        }

    def _update_package_index(self) -> None:
        self.index_updates += 1
        if self.index_error is not None:
            raise self.index_error

    def _install_from_path(self, path: pathlib.Path) -> None:
        package = self._downloaded[path]
        self.installed.append(package)
        self.installs.append((package.from_code, package.to_code))

    def install(self, monkeypatch) -> None:
        """Make `import argostranslate` and `import ctranslate2` resolve to the fakes."""
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)
