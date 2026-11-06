"""
A stand-in for argostranslate and ctranslate2.

translator.py imports both at module level and creates the user's Argos data directory as soon
as it is imported. That makes it impossible to import in CI, which installs no ML packages, and
unsafe to import in a test on a developer machine. FakeArgos puts small fake modules into
sys.modules, backed by plain Python state that a test can read and change, so the real
translator.py can be imported and exercised against them.

The fake "model" tags every word with the language it was translated into:

    "Hello world", translated into French   ->   "fr/Hello fr/world"

A text that went through English on the way to French reads "fr/en/Hello fr/en/world", so a
test can see both which input an output belongs to and which models it passed through.

Use the `fake_argos` and `translator_module` fixtures from conftest.py rather than this module
directly.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types


def tag_words(text: str, to_code: str) -> str:
    """What the fake model makes of *text*: every word tagged with the target language."""
    return " ".join(f"{to_code}/{word}" for word in text.split())


# ── The parts of a language pack ─────────────────────────────────────────────


class FakeTokenizer:
    """Words in, words out.

    With leading_space it decodes the way SentencePiece does, which puts a space in front of
    the first word."""

    def __init__(self, leading_space: bool = False):
        self.leading_space = leading_space

    def encode(self, sentence: str) -> list[str]:
        return sentence.split()

    def decode(self, tokens: list[str]) -> str:
        text = " ".join(tokens)
        return " " + text if self.leading_space and tokens else text


class FakeSentencizer:
    """Argos' sentence-boundary model: splits after sentence-ending punctuation and whitespace.

    It records the text of every call, so a test can tell when the model was consulted. Texts in
    no_sentences make it return an empty list."""

    def __init__(self):
        self.calls: list[str] = []
        self.no_sentences: set[str] = set()

    def split_sentences(self, text: str) -> list[str]:
        self.calls.append(text)
        if text in self.no_sentences:
            return []
        return [s for s in re.split(r"(?<=[.!?;:。！？])\s+", text) if s]


class FakeCT2Translator:
    """ctranslate2.Translator: tags the words of every sentence with the target language.

    Each translate_batch call is recorded as (tokenized sentences, keyword arguments). Like the
    real model, it starts a hypothesis with the target prefix when it is given one. Sentences in
    empty_for come back as an empty hypothesis, and a set error is raised instead of answering."""

    def __init__(self, to_code: str):
        self.to_code = to_code
        self.batches: list[tuple[list[list[str]], dict]] = []
        self.error: Exception | None = None
        self.empty_for: set[str] = set()

    def translate_batch(self, tokenized: list[list[str]], **kwargs):
        self.batches.append((tokenized, kwargs))
        if self.error is not None:
            raise self.error
        prefixes = kwargs.get("target_prefix")
        results = []
        for i, tokens in enumerate(tokenized):
            words = [f"{self.to_code}/{token}" for token in tokens]
            if " ".join(tokens) in self.empty_for:
                words = []
            if prefixes is not None:
                words = [*prefixes[i], *words]
            results.append(types.SimpleNamespace(hypotheses=[words]))
        return results


# ── Languages and translations (argostranslate.translate) ────────────────────


class FakeLanguage:
    """argostranslate.translate.Language."""

    def __init__(self, code: str):
        self.code = code
        self.name = code
        self.translations_from: list[ITranslation] = []

    def __str__(self) -> str:
        return self.name

    def get_translation(self, to: FakeLanguage) -> ITranslation | None:
        return next((t for t in self.translations_from if t.to_lang.code == to.code), None)


class ITranslation:
    """Base of the fake translations.

    Argos' own kinds must be translated in batches by translate_many, never one text at a time,
    so calling translate() on them is an error."""

    from_lang: FakeLanguage
    to_lang: FakeLanguage

    def translate(self, text: str) -> str:
        raise AssertionError(f"{type(self).__name__}.translate() called: texts should be batched")


class PackageTranslation(ITranslation):
    """A pack on disk. Its CTranslate2 model (translator) stays None until it is first used."""

    def __init__(self, from_lang: FakeLanguage, to_lang: FakeLanguage, pkg: FakePackage):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.pkg = pkg
        self.translator: FakeCT2Translator | None = None
        self.sentencizer = FakeSentencizer()


class CachedTranslation(ITranslation):
    """How Argos hands out a pack: wrapped, with a text cache that translate_many bypasses."""

    def __init__(self, underlying: ITranslation):
        self.underlying = underlying
        self.from_lang = underlying.from_lang
        self.to_lang = underlying.to_lang


class CompositeTranslation(ITranslation):
    """Two translations chained: the output of t1 is the input of t2."""

    def __init__(self, t1: ITranslation, t2: ITranslation):
        self.t1 = t1
        self.t2 = t2
        self.from_lang = t1.from_lang
        self.to_lang = t2.to_lang


class IdentityTranslation(ITranslation):
    """A language to itself."""

    def __init__(self, lang: FakeLanguage):
        self.from_lang = lang
        self.to_lang = lang


class RemoteTranslation(ITranslation):
    """A kind of translation translator.py does not know how to batch (Argos has several).

    It can only translate one text at a time, and records each text it was given."""

    def __init__(self, from_lang: FakeLanguage, to_lang: FakeLanguage):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return tag_words(text, self.to_lang.code)


# ── Packages ─────────────────────────────────────────────────────────────────


class FakePackage:
    """Stands in for argostranslate.package.Package and AvailablePackage."""

    def __init__(self, owner: FakeArgos, from_code: str, to_code: str, *, target_prefix: str = ""):
        self._owner = owner
        self.from_code = from_code
        self.to_code = to_code
        self.type = "translate"
        self.downloads = 0
        self.download_error: Exception | None = None
        # What the translation code reads from a pack it has installed:
        self.package_path = owner.data_dir / "packages" / f"{from_code}_{to_code}"
        self.tokenizer = FakeTokenizer()
        self.target_prefix = target_prefix

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
        self.data_dir = data_dir
        self.installed: list[FakePackage] = []
        self.available: list[FakePackage] = []
        self.index_updates = 0
        self.index_error: Exception | None = None
        self.installs: list[tuple[str, str]] = []  # pairs installed, in order
        self._downloaded: dict[pathlib.Path, FakePackage] = {}
        # What get_installed_languages() returns, and the models behind it:
        self.languages: dict[str, FakeLanguage] = {}
        self.language_lookups = 0  # how often the code asked which languages are installed
        self.models: dict[tuple[str, str], FakeCT2Translator] = {}
        self.model_loads: list[tuple[str, dict]] = []  # ctranslate2.Translator(path, **kwargs)
        self.model_load_errors: dict[str, Exception] = {}  # device name -> error loading on it
        self._models_by_path: dict[str, FakeCT2Translator] = {}
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

    def language(self, code: str) -> FakeLanguage:
        """The installed language with this code, created on first use."""
        if code not in self.languages:
            self.languages[code] = FakeLanguage(code)
        return self.languages[code]

    def pair(
        self,
        from_code: str,
        to_code: str,
        *,
        loaded: bool = True,
        cached: bool = True,
        target_prefix: str = "",
        leading_space: bool = False,
    ) -> ITranslation:
        """Make from_code -> to_code translatable with a direct pack, and return the translation.

        It is built the way Argos builds it: a PackageTranslation, wrapped in a CachedTranslation
        unless cached=False. With loaded=False the CTranslate2 model is not loaded yet, as it
        is not until a pack is first used, so translator.py has to load it.
        This is separate from `installed`, which only models the package manager."""
        package = FakePackage(self, from_code, to_code, target_prefix=target_prefix)
        package.tokenizer.leading_space = leading_space
        model = FakeCT2Translator(to_code)
        self.models[(from_code, to_code)] = model
        self._models_by_path[str(package.package_path / "model")] = model

        from_lang, to_lang = self.language(from_code), self.language(to_code)
        translation = PackageTranslation(from_lang, to_lang, package)
        if loaded:
            translation.translator = model
        result = CachedTranslation(translation) if cached else translation
        from_lang.translations_from.append(result)
        return result

    def remote(self, from_code: str, to_code: str) -> RemoteTranslation:
        """Make from_code -> to_code translatable by a kind of translation that cannot batch."""
        translation = RemoteTranslation(self.language(from_code), self.language(to_code))
        translation.from_lang.translations_from.append(translation)
        return translation

    def pivot(self, *hops: ITranslation) -> CompositeTranslation:
        """Make the first language translatable into the last by chaining hops.

        Chains nest to the left, as they do when Argos builds pivot translations."""
        if len(hops) < 2:
            raise ValueError("a pivot needs at least two hops")
        translation = hops[0]
        for hop in hops[1:]:
            translation = CompositeTranslation(translation, hop)
        translation.from_lang.translations_from.append(translation)
        return translation

    def chain(self, *codes: str) -> CompositeTranslation:
        """pivot() over a direct pack for each step: chain("ro", "en", "fr") is ro -> en -> fr."""
        return self.pivot(*(self.pair(a, b) for a, b in zip(codes, codes[1:])))

    def model(self, from_code: str, to_code: str) -> FakeCT2Translator:
        return self.models[(from_code, to_code)]

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
        translate.get_installed_languages = self._get_installed_languages
        translate.ITranslation = ITranslation
        translate.PackageTranslation = PackageTranslation
        translate.CachedTranslation = CachedTranslation
        translate.CompositeTranslation = CompositeTranslation
        translate.IdentityTranslation = IdentityTranslation

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 0
        ctranslate2.get_supported_compute_types = lambda device: {"default"}
        ctranslate2.Translator = self._load_model

        root.settings, root.package, root.translate = settings, package, translate
        self.settings = settings
        self.modules = {
            "argostranslate": root,
            "argostranslate.settings": settings,
            "argostranslate.package": package,
            "argostranslate.translate": translate,
            "ctranslate2": ctranslate2,
        }

    def _get_installed_languages(self) -> list[FakeLanguage]:
        self.language_lookups += 1
        return list(self.languages.values())

    def _load_model(self, model_path: str, **kwargs) -> FakeCT2Translator:
        """ctranslate2.Translator(model_path, device=..., ...): hands back that pack's model."""
        self.model_loads.append((model_path, kwargs))
        error = self.model_load_errors.get(kwargs.get("device"))
        if error is not None:
            raise error
        return self._models_by_path[model_path]

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
