"""
The packaging metadata in pyproject.toml, and the files that must agree with it.

Nothing here builds or installs anything: it reads the files in the repository. The built
wheel itself is checked in CI by scripts/check_wheel.py.
"""

import ast
import pathlib
import re
import tomllib

import pytest

import langscope

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_DIR = pathlib.Path(langscope.__file__).parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]

# Packages the CI environment must not contain: the tests skip what needs them.
ML_PACKAGES = {"torch", "torchvision", "easyocr", "argostranslate", "ctranslate2"}


def requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[ ;]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def read_requirements(path: pathlib.Path) -> list[str]:
    lines = (
        line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
    )
    return [line for line in lines if line]


CI_REQUIREMENTS = read_requirements(ROOT / "requirements-ci.txt")


class TestMetadata:
    def test_the_project_is_called_langscope(self):
        assert PROJECT["name"] == "langscope"

    def test_the_license_is_mit_and_its_file_is_shipped(self):
        assert PROJECT["license"] == "MIT"
        assert PROJECT["license-files"] == ["LICENSE"]
        assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")

    def test_the_author_has_no_email(self):
        assert PROJECT["authors"] == [{"name": "Chaouachi Mohamed"}]

    def test_python_3_11_is_the_floor_for_the_package_and_for_ruff(self):
        assert PROJECT["requires-python"] == ">=3.11"
        assert PYPROJECT["tool"]["ruff"]["target-version"] == "py311"

    def test_the_command_runs_main_in_the_main_module(self):
        assert PROJECT["scripts"] == {"langscope": "langscope.main:main"}
        tree = ast.parse((PACKAGE_DIR / "main.py").read_text(encoding="utf-8"))
        assert "main" in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}

    def test_the_module_can_be_run_with_python_dash_m(self):
        assert (PACKAGE_DIR / "__main__.py").is_file()

    def test_dxcam_and_dev_are_the_optional_dependencies(self):
        assert set(PROJECT["optional-dependencies"]) == {"dxcam", "dev"}

    def test_the_readme_named_in_the_metadata_exists(self):
        assert (ROOT / PROJECT["readme"]).is_file()


class TestDependencies:
    def test_torch_is_not_listed_because_it_needs_a_custom_index(self):
        names = {requirement_name(r) for r in PROJECT["dependencies"]}
        assert "torch" not in names and "torchvision" not in names

    def test_the_imports_of_the_package_are_declared(self):
        declared = {requirement_name(r) for r in PROJECT["dependencies"]}
        needed = {"pyqt6", "qtawesome", "mss", "numpy", "opencv-python-headless", "easyocr"}
        needed |= {"langdetect", "argostranslate"}
        assert needed <= declared

    def test_argostranslate_is_bounded_on_both_sides(self):
        (argos,) = [r for r in PROJECT["dependencies"] if requirement_name(r) == "argostranslate"]
        assert ">=1.10.0" in argos and "<1.12" in argos

    def test_every_dependency_is_used(self):
        # The reverse of the test above: no dependency is left over after a refactor.
        imported = set()
        for path in PACKAGE_DIR.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    imported.add(node.module.split(".")[0])
        import_names = {
            "pyqt6": "PyQt6",
            "qtawesome": "qtawesome",
            "mss": "mss",
            "numpy": "numpy",
            "opencv-python-headless": "cv2",
            "easyocr": "easyocr",
            "langdetect": "langdetect",
            "argostranslate": "argostranslate",
        }
        for requirement in PROJECT["dependencies"]:
            name = requirement_name(requirement)
            assert import_names[name] in imported, f"{name} is declared but never imported"


class TestCiRequirements:
    """requirements-ci.txt is what CI installs. It must not drift from pyproject.toml."""

    def test_the_tool_pins_are_the_dev_extra(self):
        dev = set(PROJECT["optional-dependencies"]["dev"])
        assert dev <= set(CI_REQUIREMENTS), sorted(dev - set(CI_REQUIREMENTS))

    def test_everything_else_is_a_runtime_dependency_with_the_same_specifier(self):
        dev = set(PROJECT["optional-dependencies"]["dev"])
        runtime = {requirement_name(r): r for r in PROJECT["dependencies"]}
        for requirement in CI_REQUIREMENTS:
            if requirement in dev:
                continue
            name = requirement_name(requirement)
            assert name in runtime, f"{requirement} is in requirements-ci.txt but not a dependency"
            assert runtime[name] == requirement, (runtime[name], requirement)

    def test_the_ml_stack_is_not_installed_in_ci(self):
        names = {requirement_name(r) for r in CI_REQUIREMENTS}
        assert not names & ML_PACKAGES

    @pytest.mark.parametrize("old_file", ["requirements.txt", "requirements-dev.txt"])
    def test_the_old_requirements_files_are_gone(self, old_file):
        assert not (ROOT / old_file).exists()


class TestPackageData:
    @pytest.mark.parametrize("name", ["langscope.png", "langscope.ico", "check.png"])
    def test_the_assets_are_inside_the_package(self, name):
        assert (PACKAGE_DIR / "assets" / name).is_file()

    def test_the_package_data_pattern_covers_them(self):
        assert PYPROJECT["tool"]["setuptools"]["package-data"] == {"langscope": ["assets/*"]}

    def test_the_ui_looks_for_its_icon_in_that_folder(self):
        from langscope.ui import APP_ICON_PATH

        assert pathlib.Path(APP_ICON_PATH) == PACKAGE_DIR / "assets" / "langscope.png"
