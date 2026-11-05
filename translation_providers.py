"""
translation_providers.py — Translation abstraction layer.

BaseTranslator defines the interface.
ArgosTranslator wraps translator.py (offline Argos Translate + FIFO cache).
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

    def clear_cache(self) -> None:
        import translator as _t
        _t.clear_translation_cache()
