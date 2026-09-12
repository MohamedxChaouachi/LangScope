"""
translator.py — Offline Translation via Argos Translate
Fully local inference — no internet required after first-time package install.

Package cache:
  Argos Translate uses a platform-specific data directory.  On Windows this
  ends up as C:\\Users\\<user>\\.local\\share\\argos-translate\\ (XDG-style),
  NOT under %%LOCALAPPDATA%%.  We pin the path explicitly at import time so
  that get_installed_packages() reliably finds already-downloaded models and
  never triggers a spurious re-download.

Optimizations added (Phase 1):
  • Translation cache — identical (text, from, to) tuples are never sent to
    Argos twice.  Cache is bounded to _CACHE_MAX entries and flushed on
    language-pair change.  Expected gain: eliminates 100% of repeated
    translation calls on static or slowly-changing screens.
  • Structured logging — DEBUG-level hits/misses, timing, and errors.
"""

from __future__ import annotations

import logging
import pathlib
import time
import threading
from typing import Any

# ── Pin argostranslate's data directory BEFORE importing submodules ────────────
# Without this, get_installed_packages() may return [] even when packages exist
# because argostranslate.settings resolves the path lazily and inconsistently
# across import order on Windows.
# pyrefly: ignore [missing-import]
import argostranslate.settings as _argo_settings

_DATA_DIR = pathlib.Path(_argo_settings.data_dir)      # e.g. ~/.local/share/argos-translate
_PACKAGES_DIR = _DATA_DIR / "packages"
_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

# Keep data_dir as a Path — argostranslate internals use the / operator on it
# (e.g. sbd.py: settings.data_dir / "minisbd").  Only package_data_dir is str.
_argo_settings.data_dir = _DATA_DIR
_argo_settings.package_data_dir = str(_PACKAGES_DIR)

log.info(f"[Translator] Package cache: {_PACKAGES_DIR}")

# pyrefly: ignore [missing-import]
import argostranslate.package
# pyrefly: ignore [missing-import]
import argostranslate.translate

log = logging.getLogger("langscope.translator")


# ─────────────────────────────────────────────────────────────────────────────
# Translation cache  (Goal 1)
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of (text, from_code, to_code) → translated entries to keep.
# Older entries are evicted in FIFO order when the limit is reached.
_CACHE_MAX = 512

# The cache itself: key = (text, from_code, to_code), value = translated str.
_translation_cache: dict[tuple[str, str, str], str] = {}

# Insertion-order list used for O(1) FIFO eviction.
_cache_order:  list[tuple[str, str, str]] = []

# Thread-safe access — translate_text may be called from the capture thread
# while clear_translation_cache is called from the main thread.
_cache_lock = threading.Lock()

# Lifetime counters for debugging / metrics.
_cache_hits:   int = 0
_cache_misses: int = 0


def _cache_put(key: tuple[str, str, str], value: str) -> None:
    """Insert *key→value* into the cache, evicting the oldest entry if full.
    Must be called while holding *_cache_lock*."""
    global _translation_cache, _cache_order
    if len(_translation_cache) >= _CACHE_MAX:
        oldest = _cache_order.pop(0)
        _translation_cache.pop(oldest, None)
    _translation_cache[key] = value
    _cache_order.append(key)


def clear_translation_cache() -> None:
    """Flush the entire translation cache.

    Should be called whenever the source or target language changes so that
    cached translations for the old pair are not accidentally reused.
    """
    global _translation_cache, _cache_order, _cache_hits, _cache_misses
    with _cache_lock:
        _translation_cache.clear()
        _cache_order.clear()
        _cache_hits   = 0
        _cache_misses = 0
    log.debug("[Translator] Translation cache cleared.")


def get_cache_stats() -> dict[str, Any]:
    """Return a snapshot of translation-cache statistics for debugging/metrics."""
    with _cache_lock:
        return {
            "size":   len(_translation_cache),
            "hits":   _cache_hits,
            "misses": _cache_misses,
            "max":    _CACHE_MAX,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_package_installed(from_code: str, to_code: str) -> bool:
    """Return True if the *from_code* → *to_code* package is already on disk."""
    installed = argostranslate.package.get_installed_packages()
    return any(
        p.from_code == from_code and p.to_code == to_code
        for p in installed
    )


def get_installed_pairs() -> list[tuple[str, str]]:
    """Return all (from_code, to_code) pairs that are cached locally."""
    return [
        (p.from_code, p.to_code)
        for p in argostranslate.package.get_installed_packages()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Setup — call once before the translation loop, or on language change
# ─────────────────────────────────────────────────────────────────────────────

def setup_translation_models(from_code: str, to_code: str) -> bool:
    """
    Ensure the *from_code* → *to_code* language pack is installed locally.

    * If the pack is already cached → returns True immediately (no network).
    * If not → fetches the package index, downloads and installs (~100 MB).
    * Same-language pairs (en→en) are a no-op and return True immediately.

    Args:
        from_code: BCP-47 source language code, e.g. ``"en"``.
        to_code:   BCP-47 target language code, e.g. ``"fr"``.

    Returns:
        True if the package is ready to use, False if setup failed.
    """
    pair = f"{from_code} → {to_code}"

    # Same language — nothing to translate, always "ready"
    if from_code == to_code:
        log.info(f"[Translator] Same language ({from_code}), skipping setup.")
        return True

    # ── Already cached on disk? ────────────────────────────────────────────
    if is_package_installed(from_code, to_code):
        log.info(f"[Translator] ✅ Package {pair} already cached — ready.")
        return True

    # ── Fetch package index ────────────────────────────────────────────────
    log.info(f"[Translator] Fetching Argos Translate package index…")
    try:
        argostranslate.package.update_package_index()
    except Exception as exc:
        log.warning(f"[Translator] ⚠  Could not update package index: {exc}")
        # Fall through — we may still find the package in a stale index

    # ── Locate the package ─────────────────────────────────────────────────
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if pkg is None:
        log.error(
            f"[Translator] ❌ No package found for {pair}.\n"
            "  Available pairs: https://www.argosopentech.com/argospm/index/"
        )
        return False

    # ── Download & install ─────────────────────────────────────────────────
    log.info(f"[Translator] ⬇  Downloading {pair} language pack (~100 MB)…")
    try:
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
        log.info(f"[Translator] ✅ Package {pair} installed and cached.")
        return True
    except Exception as exc:
        log.error(f"[Translator] ❌ Installation failed for {pair}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Translation  (with cache, Goal 1)
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(text: str, from_code: str, to_code: str) -> str:
    """
    Translate *text* from *from_code* to *to_code* using the locally cached
    Argos Translate model.

    Cache behaviour:
      • On a cache HIT  the translated string is returned instantly (no Argos
        call, no CPU/GPU cost).
      • On a cache MISS the result is translated, stored, and returned.

    Uses the explicit installed-languages API instead of the convenience
    wrapper, which can silently return None when duplicate package entries
    are present in get_installed_languages().

    Returns the translated string, or the original *text* on failure.
    """
    global _cache_hits, _cache_misses

    if not text.strip():
        return text

    if from_code == to_code:
        return text  # nothing to translate

    cache_key = (text, from_code, to_code)

    # ── Cache lookup ───────────────────────────────────────────────────────
    with _cache_lock:
        if cache_key in _translation_cache:
            _cache_hits += 1
            log.debug(
                "[Translator] CACHE HIT  (%s→%s) hits=%d misses=%d  %r",
                from_code, to_code, _cache_hits, _cache_misses, text[:40],
            )
            return _translation_cache[cache_key]
        _cache_misses += 1

    log.debug(
        "[Translator] CACHE MISS (%s→%s) hits=%d misses=%d  %r",
        from_code, to_code, _cache_hits, _cache_misses, text[:40],
    )
    log.debug(f"[Translator] Translating ({from_code}->{to_code}): {text[:50]!r}")

    # ── Argos Translate call ───────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        installed = argostranslate.translate.get_installed_languages()

        # De-duplicate: keep first occurrence of each language code
        seen: set[str] = set()
        unique_langs = []
        for lang in installed:
            if lang.code not in seen:
                seen.add(lang.code)
                unique_langs.append(lang)

        from_lang = next((l for l in unique_langs if l.code == from_code), None)
        to_lang   = next((l for l in unique_langs if l.code == to_code),   None)

        if from_lang is None or to_lang is None:
            log.warning(f"[Translator] ⚠  Language not found — from={from_lang}, to={to_lang}. "
                        f"Available: {[l.code for l in unique_langs]}")
            return text

        translation = from_lang.get_translation(to_lang)
        if translation is None:
            log.warning(f"[Translator] ⚠  No translation path for {from_code}->{to_code}")
            return text

        result = translation.translate(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log.debug(
            "[Translator] translate_text done in %.1f ms (%s→%s)",
            elapsed_ms, from_code, to_code,
        )
        log.debug(f"[Translator] OK Result: {result[:50]!r}")

        final = result.strip() if result and result.strip() else text

        # ── Store in cache ─────────────────────────────────────────────────
        with _cache_lock:
            _cache_put(cache_key, final)

        return final

    except Exception as exc:
        log.error(f"[Translator] ERR Exception ({from_code}->{to_code}): {exc}")
        return text
