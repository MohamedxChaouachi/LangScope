"""
translation_providers.py — Translation abstraction layer.

BaseTranslator defines the interface.
ArgosTranslator wraps translator.py (offline Argos Translate).
Future engines (CTranslate2, DeepL…) implement BaseTranslator without touching
any other module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger("langscope.translation_providers")


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str, from_code: str, to_code: str) -> str: ...

    def translate_many(self, texts: list[str], from_code: str, to_code: str) -> list[str]:
        """Translate independent strings, returning a list aligned with *texts*.
        Default implementation calls translate() per item; engines that can
        batch should override."""
        return [self.translate(t, from_code, to_code) for t in texts]

    def warm_up(self, from_code: str, to_code: str) -> bool:
        """Pre-load models so the first real call is fast. Returns True when
        the pair is ready. No-op (True) by default."""
        return True

    def clear_cache(self) -> None:
        """Flush internal cache. No-op by default."""


class ArgosTranslator(BaseTranslator):
    """Offline translation via Argos Translate. Delegates to translator.py."""

    def translate(self, text: str, from_code: str, to_code: str) -> str:
        import translator as _t

        return _t.translate_text(text, from_code, to_code)

    def translate_many(self, texts: list[str], from_code: str, to_code: str) -> list[str]:
        import translator as _t

        return _t.translate_many(texts, from_code, to_code)

    def warm_up(self, from_code: str, to_code: str) -> bool:
        import translator as _t

        return _t.preload_model(from_code, to_code)


class LanguagePacks(ABC):
    """The offline language packs a translation engine needs, and how to get the missing ones."""

    @abstractmethod
    def is_installed(self, from_code: str, to_code: str) -> bool:
        """True if from_code -> to_code can be translated with what is already on disk."""

    @abstractmethod
    def install(self, from_code: str, to_code: str) -> bool:
        """Download and install what from_code -> to_code needs. Slow: call it off the GUI
        thread. Returns False, after logging why, if the pair cannot be made ready."""


class ArgosLanguagePacks(LanguagePacks):
    """Argos Translate language packs. Delegates to translator.py."""

    def is_installed(self, from_code: str, to_code: str) -> bool:
        import translator as _t

        return _t.is_package_installed(from_code, to_code)

    def install(self, from_code: str, to_code: str) -> bool:
        import translator as _t

        return _t.setup_translation_models(from_code, to_code)
