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

    def clear_cache(self) -> None:
        """Flush internal cache. No-op by default."""


class ArgosTranslator(BaseTranslator):
    """Offline translation via Argos Translate. Delegates to translator.py."""

    def translate(self, text: str, from_code: str, to_code: str) -> str:
        import translator as _t
        return _t.translate_text(text, from_code, to_code)

    def clear_cache(self) -> None:
        import translator as _t
        _t.clear_translation_cache()
