"""
translator.py — Offline Translation via Argos Translate
Fully local inference — no internet required after first-time package install.

Package cache:
  Argos Translate uses a platform-specific data directory.  On Windows this
  ends up as C:\\Users\\<user>\\.local\\share\\argos-translate\\ (XDG-style),
  NOT under %%LOCALAPPDATA%%.  We pin the path explicitly at import time so
  that get_installed_packages() reliably finds already-downloaded models and
  never triggers a spurious re-download.
"""

from __future__ import annotations

import logging
import pathlib

# ── Pin argostranslate's data directory BEFORE importing submodules ────────────
# Without this, get_installed_packages() may return [] even when packages exist
# because argostranslate.settings resolves the path lazily and inconsistently
# across import order on Windows.
import argostranslate.settings as _argo_settings

_DATA_DIR = pathlib.Path(_argo_settings.data_dir)      # e.g. ~/.local/share/argos-translate
_PACKAGES_DIR = _DATA_DIR / "packages"
_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

# Keep data_dir as a Path — argostranslate internals use the / operator on it
# (e.g. sbd.py: settings.data_dir / "minisbd").  Only package_data_dir is str.
_argo_settings.data_dir = _DATA_DIR
_argo_settings.package_data_dir = str(_PACKAGES_DIR)

print(f"[Translator] Package cache: {_PACKAGES_DIR}")

# pyrefly: ignore [missing-import]
import argostranslate.package
# pyrefly: ignore [missing-import]
import argostranslate.translate

log = logging.getLogger(__name__)


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
        print(f"[Translator] Same language ({from_code}), skipping setup.")
        return True

    # ── Already cached on disk? ────────────────────────────────────────────
    if is_package_installed(from_code, to_code):
        print(f"[Translator] ✅ Package {pair} already cached — ready.")
        return True

    # ── Fetch package index ────────────────────────────────────────────────
    print(f"[Translator] Fetching Argos Translate package index…")
    try:
        argostranslate.package.update_package_index()
    except Exception as exc:
        print(f"[Translator] ⚠  Could not update package index: {exc}")
        # Fall through — we may still find the package in a stale index

    # ── Locate the package ─────────────────────────────────────────────────
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if pkg is None:
        print(
            f"[Translator] ❌ No package found for {pair}.\n"
            "  Available pairs: https://www.argosopentech.com/argospm/index/"
        )
        return False

    # ── Download & install ─────────────────────────────────────────────────
    print(f"[Translator] ⬇  Downloading {pair} language pack (~100 MB)…")
    try:
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
        print(f"[Translator] ✅ Package {pair} installed and cached.")
        return True
    except Exception as exc:
        print(f"[Translator] ❌ Installation failed for {pair}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(text: str, from_code: str, to_code: str) -> str:
    """
    Translate *text* from *from_code* to *to_code* using the locally cached
    Argos Translate model.

    Uses the explicit installed-languages API instead of the convenience
    wrapper, which can silently return None when duplicate package entries
    are present in get_installed_languages().

    Returns the translated string, or the original *text* on failure.
    """
    if not text.strip():
        return text

    if from_code == to_code:
        return text  # nothing to translate

    print(f"[Translator] Translating ({from_code}->{to_code}): {text[:50]!r}")
    try:
        # ── Use explicit language lookup (more robust than the wrapper) ────
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
            print(f"[Translator] ⚠  Language not found — from={from_lang}, to={to_lang}. "
                  f"Available: {[l.code for l in unique_langs]}")
            return text

        translation = from_lang.get_translation(to_lang)
        if translation is None:
            print(f"[Translator] ⚠  No translation path for {from_code}->{to_code}")
            return text

        result = translation.translate(text)
        print(f"[Translator] OK Result: {result[:50]!r}")
        return result.strip() if result and result.strip() else text

    except Exception as exc:
        print(f"[Translator] ERR Exception ({from_code}->{to_code}): {exc}")
        return text
