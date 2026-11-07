"""
The contract between translator.py and Argos Translate / CTranslate2.

translator.py does not only use Argos' public API. translate_many reaches into the objects Argos
builds (PackageTranslation.translator, pkg.tokenizer, CompositeTranslation.t1 and t2, ...) so
that it can send every sentence of every line to the model in one batch. None of that is
promised to stay the same, so these tests read translator.py itself, work out everything it
needs, and check that the installed packages still provide it. If one fails after an upgrade,
change translator.py (see _batch_package and _batch_translation) and the version range in
pyproject.toml together.

They need the real packages and skip, with a reason, where those are not installed (as in CI).
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import pathlib
import re
import textwrap
import tomllib

import pytest

argos_translate = pytest.importorskip(
    "argostranslate.translate", reason="the Argos contract needs argostranslate installed"
)
ctranslate2 = pytest.importorskip(
    "ctranslate2", reason="the Argos contract needs ctranslate2 installed"
)
argos_package = importlib.import_module("argostranslate.package")
argos_sbd = importlib.import_module("argostranslate.sbd")
argos_settings = importlib.import_module("argostranslate.settings")
argos_tokenizer = importlib.import_module("argostranslate.tokenizer")

import langscope  # noqa: E402  (after the importorskip calls)

PACKAGE_DIR = pathlib.Path(langscope.__file__).parent
TRANSLATOR_SOURCE = (PACKAGE_DIR / "translator.py").read_text(encoding="utf-8")
TREE = ast.parse(TRANSLATOR_SOURCE)
ROOT = PACKAGE_DIR.parent.parent

UPDATE_HINT = (
    "translator.py relies on Argos internals. Update _batch_package / _batch_translation in "
    "translator.py and the argostranslate range in pyproject.toml together."
)

# ── Reading translator.py ────────────────────────────────────────────────────


def attribute_chains(tree: ast.AST) -> list[tuple[str, list[str]]]:
    """Every longest `name.a.b.c` in the tree, as ("name", ["a", "b", "c"])."""
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    chains = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in inner:
            parts = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                chains.append((current.id, parts[::-1]))
    return chains


def imported_modules(tree: ast.AST) -> dict[str, str]:
    """{name the module is bound to: its full name} for the `import x.y [as z]` lines."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
    return bound


def imported_module_names(tree: ast.AST) -> list[str]:
    return [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]


def keywords_passed_to(tree: ast.AST, function_name: str) -> set[str]:
    """The keyword arguments of every call of `<anything>.function_name(...)`."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == function_name:
                found |= {k.arg for k in node.keywords if k.arg}
    return found


# ── Asking the installed classes what they provide ───────────────────────────


def assigned_on_self(cls: type) -> set[str]:
    """Names set as `self.x = ...` in any method of cls or of any class it inherits from.

    Not only __init__: Argos' Package fills in most of its attributes from a method that
    __init__ calls."""
    names: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        for member in vars(klass).values():
            function = member.fget if isinstance(member, property) else member
            if not inspect.isfunction(function):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                for target in targets:
                    for sub in ast.walk(target):
                        if (
                            isinstance(sub, ast.Attribute)
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id == "self"
                        ):
                            names.add(sub.attr)
    return names


def provides(cls: type, name: str) -> bool:
    """True if instances of cls have `name`: a method, a property, a field, or set on self."""
    if hasattr(cls, name):
        return True
    if any(name in getattr(klass, "__annotations__", {}) for klass in cls.__mro__):
        return True
    return name in assigned_on_self(cls)


def concrete_subclasses(base: type) -> list[type]:
    found = []
    for sub in base.__subclasses__():
        found.append(sub)
        found += concrete_subclasses(sub)
    return found


# ── The version ──────────────────────────────────────────────────────────────


class TestInstalledVersion:
    def test_it_is_inside_the_range_pyproject_declares(self):
        from packaging.requirements import Requirement

        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        (requirement,) = [
            Requirement(r)
            for r in project["dependencies"]
            if Requirement(r).name == "argostranslate"
        ]
        installed = importlib.metadata.version("argostranslate")
        assert installed in requirement.specifier, (
            f"argostranslate {installed} is installed but pyproject.toml allows "
            f"{requirement.specifier}. {UPDATE_HINT}"
        )


# ── Every Argos and CTranslate2 name translator.py uses ──────────────────────


@pytest.fixture(scope="module")
def modules():
    """The real modules, under the names translator.py binds them to."""
    for name in imported_module_names(TREE):
        importlib.import_module(name)
    return {bound: importlib.import_module(full) for bound, full in imported_modules(TREE).items()}


class TestNamesTranslatorUses:
    def test_translator_imports_what_this_test_expects(self):
        assert set(imported_module_names(TREE)) >= {
            "argostranslate.package",
            "argostranslate.settings",
            "argostranslate.translate",
            "ctranslate2",
        }

    def test_every_module_level_name_it_uses_exists(self, modules):
        problems = []
        checked = 0
        for root, attrs in attribute_chains(TREE):
            if root not in modules:
                continue
            obj = modules[root]
            path = root
            for attr in attrs:
                path += f".{attr}"
                if not hasattr(obj, attr):
                    problems.append(path)
                    break
                obj = getattr(obj, attr)
            checked += 1
        assert checked >= 20, "translator.py no longer looks like it uses Argos; has it moved?"
        assert not problems, f"missing: {sorted(set(problems))}. {UPDATE_HINT}"

    def test_the_settings_it_assigns_exist(self):
        for name in ("data_dir", "package_data_dir", "device", "compute_type"):
            assert hasattr(argos_settings, name), f"settings.{name}. {UPDATE_HINT}"

    def test_the_numeric_settings_are_numbers(self):
        for name in ("inter_threads", "intra_threads", "batch_size", "beam_size"):
            assert isinstance(getattr(argos_settings, name), int), name

    def test_data_dir_is_a_path_because_argos_divides_it(self):
        # init_translation_backend() sets data_dir to a Path, since Argos' own sbd.py builds
        # `settings.data_dir / "minisbd"`. It must be one already, or that fix is not enough.
        import pathlib as _pathlib

        assert isinstance(argos_settings.data_dir, _pathlib.Path)


# ── The objects Argos builds ─────────────────────────────────────────────────

# What translator.py reads from each kind of object, by the variable it keeps it in.
READ_FROM_VARIABLE = {
    "pt": {"pkg", "translator", "sentencizer"},  # a PackageTranslation
    "pkg": {"package_path", "tokenizer", "target_prefix", "from_code", "to_code", "download"},
    "p": {"from_code", "to_code"},  # a Package or an AvailablePackage
    "translation": {"underlying", "t1", "t2", "translate"},  # any kind of translation
    "lang": {"code"},
    "from_lang": {"code", "get_translation"},
    "to_lang": {"code"},
}
# And one level further down.
READ_FROM_ATTRIBUTE = {
    ("pt", "sentencizer"): {"split_sentences"},
    ("pt", "translator"): {"translate_batch"},
    ("pt", "pkg"): {"package_path", "tokenizer", "target_prefix"},
    ("pkg", "tokenizer"): {"encode", "decode"},
}


class TestObjectsTranslatorReads:
    def test_translator_reads_nothing_this_contract_does_not_know_about(self):
        """If translator.py starts reading a new attribute, this fails until it is covered."""
        unknown = []
        for root, attrs in attribute_chains(TREE):
            allowed = READ_FROM_VARIABLE.get(root)
            if allowed is None:
                continue
            if attrs[0] not in allowed:
                unknown.append(f"{root}.{attrs[0]}")
            deeper = READ_FROM_ATTRIBUTE.get((root, attrs[0]))
            if deeper is not None and len(attrs) > 1 and attrs[1] not in deeper:
                unknown.append(f"{root}.{attrs[0]}.{attrs[1]}")
        assert not unknown, (
            f"translator.py reads {sorted(set(unknown))}, which this test does not check. "
            "Add them to READ_FROM_VARIABLE / READ_FROM_ATTRIBUTE and to the checks below."
        )

    def test_the_variables_are_really_used_in_translator(self):
        used = {root for root, _ in attribute_chains(TREE)}
        assert {"pt", "pkg", "translation", "from_lang"} <= used

    def test_a_package_translation_has_a_pkg_a_model_and_a_sentence_splitter(self):
        for name in ("pkg", "translator", "sentencizer"):
            assert provides(argos_translate.PackageTranslation, name), f"{name}. {UPDATE_HINT}"

    def test_a_cached_translation_wraps_another(self):
        assert provides(argos_translate.CachedTranslation, "underlying"), UPDATE_HINT

    def test_a_composite_translation_has_its_two_halves(self):
        for name in ("t1", "t2"):
            assert provides(argos_translate.CompositeTranslation, name), f"{name}. {UPDATE_HINT}"

    def test_every_translation_can_translate_one_text(self):
        # The fallback for a kind of translation translator.py does not know how to batch.
        assert callable(argos_translate.ITranslation.translate)

    def test_a_language_has_a_code_and_can_find_its_translation(self):
        assert provides(argos_translate.Language, "code")
        assert callable(argos_translate.Language.get_translation)

    def test_an_installed_package_has_what_translate_many_reads(self):
        for name in ("package_path", "tokenizer", "target_prefix", "from_code", "to_code"):
            assert provides(argos_package.Package, name), f"Package.{name}. {UPDATE_HINT}"

    def test_an_available_package_can_be_downloaded(self):
        for name in ("from_code", "to_code", "download"):
            assert provides(argos_package.AvailablePackage, name), f"{name}. {UPDATE_HINT}"

    def test_the_package_functions_have_the_arguments_translator_passes(self):
        parameters = lambda f: list(inspect.signature(f).parameters)  # noqa: E731
        assert parameters(argos_package.update_package_index) == []
        assert parameters(argos_package.get_available_packages) == []
        assert parameters(argos_package.install_from_path)[:1] == ["path"]
        assert parameters(argos_translate.get_installed_languages) == []

    def test_every_sentence_splitter_has_split_sentences_of_one_text(self):
        splitters = concrete_subclasses(argos_sbd.ISentenceBoundaryDetectionModel)
        assert splitters, "no sentence splitters found"
        for cls in splitters:
            assert list(inspect.signature(cls.split_sentences).parameters) == ["self", "text"], cls

    def test_every_tokenizer_encodes_a_sentence_and_decodes_tokens(self):
        tokenizers = concrete_subclasses(argos_tokenizer.Tokenizer)
        assert tokenizers, "no tokenizers found"
        for cls in tokenizers:
            assert list(inspect.signature(cls.encode).parameters) == ["self", "sentence"], cls
            assert list(inspect.signature(cls.decode).parameters) == ["self", "tokens"], cls

    def test_the_four_kinds_of_translation_do_not_inherit_from_each_other(self):
        # _batch_translation tests isinstance in a fixed order. If one of these became a
        # subclass of another, the first test that matches would pick the wrong branch.
        kinds = [
            argos_translate.CachedTranslation,
            argos_translate.PackageTranslation,
            argos_translate.CompositeTranslation,
            argos_translate.IdentityTranslation,
        ]
        for a in kinds:
            for b in kinds:
                if a is not b:
                    assert not issubclass(a, b), (
                        f"{a.__name__} is now a {b.__name__}. {UPDATE_HINT}"
                    )


# ── CTranslate2 ──────────────────────────────────────────────────────────────


def signature_names(function) -> set[str]:
    """Parameter names from the pybind11 signature on the first line of the docstring."""
    first_line = (function.__doc__ or "").strip().splitlines()[0]
    return set(re.findall(r"[(,]\s*\*?\s*(\w+)\s*:", first_line)) | set(
        re.findall(r",\s*(\w+)\s*=", first_line)
    )


class TestCTranslate2:
    def test_translate_batch_takes_every_keyword_translator_passes(self):
        passed = keywords_passed_to(TREE, "translate_batch")
        assert passed >= {"target_prefix", "beam_size", "length_penalty"}, passed
        missing = passed - signature_names(ctranslate2.Translator.translate_batch)
        assert not missing, f"translate_batch no longer takes {sorted(missing)}. {UPDATE_HINT}"

    def test_the_translator_constructor_takes_every_keyword_translator_passes(self):
        passed = keywords_passed_to(TREE, "Translator")
        assert passed >= {"device", "compute_type"}, passed
        missing = passed - signature_names(ctranslate2.Translator.__init__)
        assert not missing, f"Translator() no longer takes {sorted(missing)}. {UPDATE_HINT}"

    def test_a_translation_result_has_hypotheses(self):
        # translate_many reads results[k].hypotheses[0].
        assert hasattr(ctranslate2.TranslationResult, "hypotheses")

    def test_the_device_probes_exist(self):
        assert callable(ctranslate2.get_cuda_device_count)
        assert callable(ctranslate2.get_supported_compute_types)


# ── Against a real language pack, when one is installed ──────────────────────


def installed_pairs() -> set[tuple[str, str]]:
    return {(p.from_code, p.to_code) for p in argos_package.get_installed_packages()}


@pytest.mark.skipif(
    not {("en", "fr"), ("fr", "en")} <= installed_pairs(),
    reason="needs the en-fr and fr-en language packs installed",
)
class TestAgainstARealPack:
    """translate_many promises the same translation as Argos gives one text at a time."""

    TEXTS = ["Hello world", "Good morning. How are you today?", "", "Open the settings window"]

    def argos_own(self, text: str, from_code: str, to_code: str) -> str:
        languages = argos_translate.get_installed_languages()
        source = next(lang for lang in languages if lang.code == from_code)
        target = next(lang for lang in languages if lang.code == to_code)
        translation = source.get_translation(target)
        assert translation is not None
        return translation.translate(text) if text.strip() else text

    @pytest.mark.parametrize("from_code, to_code", [("en", "fr"), ("fr", "en")])
    def test_the_batch_matches_argos_own_output(self, from_code, to_code):
        from langscope import translator

        texts = self.TEXTS if from_code == "en" else ["Bonjour tout le monde", "", "Merci beaucoup"]
        batched = translator.translate_many(texts, from_code, to_code)
        expected = [self.argos_own(t, from_code, to_code) for t in texts]
        assert [b.strip() for b in batched] == [e.strip() for e in expected]
