"""
translator.py — Offline Translation via Argos Translate
Fully local inference — no internet required after first-time package install.

Package cache:
  Argos Translate uses a platform-specific data directory.  On Windows this
  ends up as C:\\Users\\<user>\\.local\\share\\argos-translate\\ (XDG-style),
  NOT under %%LOCALAPPDATA%%.  We pin the path explicitly at import time so
  that get_installed_packages() reliably finds already-downloaded models and
  never triggers a spurious re-download.

Caching is not done here: TranslationWorker keeps the one (bounded) translation cache, above
the BaseTranslator interface, so every engine benefits and nothing is cached twice.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import threading
import time

log = logging.getLogger("langscope.translator")

# ── Pin argostranslate's data directory BEFORE importing submodules ────────────
# Without this, get_installed_packages() may return [] even when packages exist
# because argostranslate.settings resolves the path lazily and inconsistently
# across import order on Windows.
# pyrefly: ignore [missing-import]
import argostranslate.settings as _argo_settings

_DATA_DIR = pathlib.Path(_argo_settings.data_dir)  # e.g. ~/.local/share/argos-translate
_PACKAGES_DIR = _DATA_DIR / "packages"
_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

# Keep data_dir as a Path — argostranslate internals use the / operator on it
# (e.g. sbd.py: settings.data_dir / "minisbd").  Only package_data_dir is str.
_argo_settings.data_dir = _DATA_DIR
_argo_settings.package_data_dir = str(_PACKAGES_DIR)

log.info(f"[Translator] Package cache: {_PACKAGES_DIR}")


def _configure_inference_device() -> None:
    """
    Run CTranslate2 on the GPU when one is available.

    Argos defaults to device="cpu" regardless of hardware. Explicit
    ARGOS_DEVICE_TYPE / ARGOS_COMPUTE_TYPE environment variables always win.
    argostranslate.translate reads these settings lazily (when a model is
    first loaded), so assigning them here is early enough.
    """
    if os.environ.get("ARGOS_DEVICE_TYPE"):
        print(f"[Translator] Device (from env): {_argo_settings.device}")
        return
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            _argo_settings.device = "cuda"
            if not os.environ.get("ARGOS_COMPUTE_TYPE"):
                if "float16" in ctranslate2.get_supported_compute_types("cuda"):
                    _argo_settings.compute_type = "float16"
            print(f"[Translator] Device: CUDA ({_argo_settings.compute_type})")
            return
    except Exception as exc:
        print(f"[Translator] CUDA probe failed ({exc}).")
    print("[Translator] Device: CPU")


_configure_inference_device()

# pyrefly: ignore [missing-import]
import argostranslate.package

# pyrefly: ignore [missing-import]
import argostranslate.translate

# pyrefly: ignore [missing-import]
import ctranslate2

# ─────────────────────────────────────────────────────────────────────────────
# Installed language packs
# ─────────────────────────────────────────────────────────────────────────────


# Argos ships a pack between English and every other language, plus a few direct pairs such as
# es <-> pt. For any other pair it chains two packs through English at translation time, so
# setting such a pair up means having both packs.
PIVOT_LANGUAGE = "en"


def _pivot_legs(from_code: str, to_code: str) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """The two packs (from → English, English → to) that a pivoted pair needs.

    None when pivoting does not apply, because English is already one end of the pair."""
    if PIVOT_LANGUAGE in (from_code, to_code):
        return None
    return (from_code, PIVOT_LANGUAGE), (PIVOT_LANGUAGE, to_code)


def is_package_installed(from_code: str, to_code: str) -> bool:
    """Return True if *from_code* → *to_code* can be translated with what is on disk.

    That is the case when the direct pack is installed, or when both packs of the pivot
    through English are."""
    installed = set(get_installed_pairs())
    if (from_code, to_code) in installed:
        return True
    legs = _pivot_legs(from_code, to_code)
    return legs is not None and all(leg in installed for leg in legs)


def get_installed_pairs() -> list[tuple[str, str]]:
    """Return all (from_code, to_code) pairs that are cached locally."""
    return [(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()]


# ─────────────────────────────────────────────────────────────────────────────
# Setup — call once before the translation loop, or on language change
# ─────────────────────────────────────────────────────────────────────────────


def _install_package(pkg) -> bool:
    """Download and install one language pack. Returns False (after logging) if that fails."""
    pair = f"{pkg.from_code} → {pkg.to_code}"
    log.info("[Translator] ⬇  Downloading %s language pack (~100 MB)…", pair)
    try:
        argostranslate.package.install_from_path(pkg.download())
    except Exception as exc:
        log.error("[Translator] ❌ Installation failed for %s: %s", pair, exc)
        return False
    log.info("[Translator] ✅ Package %s installed and cached.", pair)
    return True


def setup_translation_models(from_code: str, to_code: str) -> bool:
    """
    Ensure *from_code* → *to_code* can be translated with locally installed packs.

    * If it can already → returns True immediately (no network).
    * If the package index offers a direct pack → downloads and installs it (~100 MB).
    * Otherwise → pivots through English: installs whichever of from → English and
      English → to is missing. Nothing is downloaded unless every missing pack is in the index.
    * Same-language pairs (en→en) are a no-op and return True immediately.

    Args:
        from_code: BCP-47 source language code, e.g. ``"en"``.
        to_code:   BCP-47 target language code, e.g. ``"fr"``.

    Returns:
        True if the pair is ready to use, False if setup failed.
    """
    pair = f"{from_code} → {to_code}"

    # Same language — nothing to translate, always "ready"
    if from_code == to_code:
        log.info("[Translator] Same language (%s), skipping setup.", from_code)
        return True

    # ── Already cached on disk? ────────────────────────────────────────────
    if is_package_installed(from_code, to_code):
        log.info("[Translator] ✅ Package %s already cached — ready.", pair)
        return True

    # ── Fetch package index ────────────────────────────────────────────────
    log.info("[Translator] Fetching Argos Translate package index…")
    try:
        argostranslate.package.update_package_index()
    except Exception as exc:
        log.warning("[Translator] ⚠  Could not update package index: %s", exc)
        # Fall through — we may still find the package in a stale index

    # ── Decide which packs are needed ──────────────────────────────────────
    available: dict[tuple[str, str], object] = {}
    for p in argostranslate.package.get_available_packages():
        available.setdefault((p.from_code, p.to_code), p)

    needed: list = []
    direct = available.get((from_code, to_code))
    if direct is not None:
        needed = [direct]
    else:
        legs = _pivot_legs(from_code, to_code)
        if legs is not None:
            installed = set(get_installed_pairs())
            missing = [leg for leg in legs if leg not in installed]
            if not missing:  # a pack finished installing since the check above
                return True
            if all(leg in available for leg in missing):
                needed = [available[leg] for leg in missing]
                log.info("[Translator] No direct %s pack — pivoting through English.", pair)
            else:
                unavailable = [leg for leg in missing if leg not in available]
                log.error(
                    "[Translator] ❌ No package found for %s: no direct pack, and the index lacks %s.",
                    pair,
                    ", ".join(f"{a} → {b}" for a, b in unavailable),
                )
                return False

    if not needed:
        log.error(
            "[Translator] ❌ No package found for %s.\n"
            "  Available pairs: https://www.argosopentech.com/argospm/index/",
            pair,
        )
        return False

    # ── Download & install ─────────────────────────────────────────────────
    return all(_install_package(pkg) for pkg in needed)


# ─────────────────────────────────────────────────────────────────────────────
# Translation  (with cache, Goal 1)
# ─────────────────────────────────────────────────────────────────────────────


def translate_text(text: str, from_code: str, to_code: str) -> str:
    """
    Translate *text* from *from_code* to *to_code* using the locally cached
    Argos Translate model. Results are not cached here; TranslationWorker does that.

    Uses the explicit installed-languages API instead of the convenience
    wrapper, which can silently return None when duplicate package entries
    are present in get_installed_languages().

    Returns the translated string, or the original *text* on failure.
    """
    if not text.strip():
        return text

    if from_code == to_code:
        return text  # nothing to translate

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

        from_lang = next((lang for lang in unique_langs if lang.code == from_code), None)
        to_lang = next((lang for lang in unique_langs if lang.code == to_code), None)

        if from_lang is None or to_lang is None:
            log.warning(
                f"[Translator] ⚠  Language not found — from={from_lang}, to={to_lang}. "
                f"Available: {[lang.code for lang in unique_langs]}"
            )
            return text

        translation = from_lang.get_translation(to_lang)
        if translation is None:
            log.warning(f"[Translator] ⚠  No translation path for {from_code}->{to_code}")
            return text

        result = translation.translate(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log.debug(
            "[Translator] translate_text done in %.1f ms (%s→%s)",
            elapsed_ms,
            from_code,
            to_code,
        )
        log.debug(f"[Translator] OK Result: {result[:50]!r}")

        return result.strip() if result and result.strip() else text

    except Exception as exc:
        log.error(f"[Translator] ERR Exception ({from_code}->{to_code}): {exc}")
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Batched translation
# ─────────────────────────────────────────────────────────────────────────────
#
# Argos' ITranslation.translate() splits its input on "\n" and runs one
# CTranslate2 call per line, so joining N OCR lines into one string does NOT
# batch them (N sequential model calls, each paying launch overhead — costly on
# a GPU). translate_many() tokenizes every sentence of every line first and
# sends them to CTranslate2 in a single translate_batch() call, using exactly
# the same tokenizer, beam size and decoding steps as Argos, so output matches.

# A boundary is "sentence-ending punctuation followed by more text". Lines
# without one are a single sentence, so the (CPU-bound) sentence-boundary
# model can be skipped.
_SENTENCE_BREAK_RE = re.compile(r"[.!?;:。！？]\s*\S")
_model_load_lock = threading.Lock()


def _get_translation(from_code: str, to_code: str):
    """Return Argos' ITranslation for the pair, or None if unavailable."""
    installed = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in installed if lang.code == from_code), None)
    to_lang = next((lang for lang in installed if lang.code == to_code), None)
    if from_lang is None or to_lang is None:
        return None
    return from_lang.get_translation(to_lang)


def _ensure_ct2_translator(pt) -> None:
    """Load pt.translator (same arguments Argos uses). Falls back to CPU if
    the CUDA load fails."""
    if pt.translator is not None:
        return
    with _model_load_lock:
        if pt.translator is not None:
            return
        model_path = str(pt.pkg.package_path / "model")

        def _load():
            return ctranslate2.Translator(
                model_path,
                device=_argo_settings.device,
                inter_threads=_argo_settings.inter_threads,
                intra_threads=_argo_settings.intra_threads,
                compute_type=_argo_settings.compute_type,
            )

        try:
            pt.translator = _load()
        except Exception as exc:
            if _argo_settings.device != "cuda":
                raise
            print(f"[Translator] CUDA model load failed ({exc}) — falling back to CPU.")
            _argo_settings.device = "cpu"
            _argo_settings.compute_type = "auto"
            pt.translator = _load()


def _split_sentences(pt, text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if not _SENTENCE_BREAK_RE.search(text):
        return [text]
    return pt.sentencizer.split_sentences(text)


def _batch_package(pt, texts: list[str]) -> list[str]:
    """Translate texts with one translate_batch call on a PackageTranslation."""
    _ensure_ct2_translator(pt)
    pkg = pt.pkg

    sentences_per_text = [_split_sentences(pt, t) for t in texts]
    flat = [s for sents in sentences_per_text for s in sents]
    if not flat:
        return list(texts)

    tokenized = [pkg.tokenizer.encode(s) for s in flat]
    target_prefix = None
    if pkg.target_prefix != "":
        target_prefix = [[pkg.target_prefix]] * len(tokenized)

    results = pt.translator.translate_batch(
        tokenized,
        target_prefix=target_prefix,
        replace_unknowns=True,
        max_batch_size=_argo_settings.batch_size,
        batch_type="tokens",
        beam_size=_argo_settings.beam_size,
        num_hypotheses=1,
        length_penalty=0.2,
        return_scores=False,
    )

    out: list[str] = []
    k = 0
    for text, sents in zip(texts, sentences_per_text):
        if not sents:
            out.append(text)
            continue
        tokens: list[str] = []
        for _ in sents:
            tokens.extend(results[k].hypotheses[0])
            k += 1
        value = pkg.tokenizer.decode(tokens)
        if pkg.target_prefix != "" and value.startswith(pkg.target_prefix):
            value = value[len(pkg.target_prefix) :]
        if value.startswith(" "):
            value = value[1:]
        out.append(value.strip() or text)
    return out


def _batch_translation(translation, texts: list[str]) -> list[str]:
    """Dispatch on Argos' translation wrapper types."""
    if isinstance(translation, argostranslate.translate.CachedTranslation):
        return _batch_translation(translation.underlying, texts)
    if isinstance(translation, argostranslate.translate.PackageTranslation):
        return _batch_package(translation, texts)
    if isinstance(translation, argostranslate.translate.CompositeTranslation):
        # Pivot translation (e.g. ro→en→fr): two batched hops.
        return _batch_translation(translation.t2, _batch_translation(translation.t1, texts))
    if isinstance(translation, argostranslate.translate.IdentityTranslation):
        return list(texts)
    # Unknown wrapper type: slow path, still correct.
    return [translation.translate(t) for t in texts]


def translate_many(texts: list[str], from_code: str, to_code: str) -> list[str]:
    """
    Translate several independent strings with a single batched model call.

    Returns a list aligned 1:1 with *texts*. Identical strings are translated
    once, blank strings and same-language requests pass through unchanged, and
    any failure returns the original text for the affected entries.
    """
    results: list[str | None] = [None] * len(texts)
    pending: dict[str, list[int]] = {}  # unique text -> indices to fill

    for i, text in enumerate(texts):
        if not text.strip() or from_code == to_code:
            results[i] = text
            continue
        pending.setdefault(text, []).append(i)

    if pending:
        unique = list(pending.keys())
        t0 = time.perf_counter()
        try:
            translation = _get_translation(from_code, to_code)
            if translation is None:
                print(f"[Translator] ⚠  No translation path for {from_code}->{to_code}")
                translated = list(unique)
            else:
                translated = _batch_translation(translation, unique)
        except Exception as exc:
            print(f"[Translator] ERR batch failed ({from_code}->{to_code}): {exc}")
            log.exception("Batch translation failed")
            translated = list(unique)
        log.debug(
            "[Translator] translate_many %d texts in %.1f ms",
            len(unique),
            (time.perf_counter() - t0) * 1000,
        )

        for text, tx in zip(unique, translated):
            for i in pending[text]:
                results[i] = tx

    return [r if r is not None else texts[i] for i, r in enumerate(results)]


def preload_model(from_code: str, to_code: str) -> bool:
    """Load the model onto the device and run a tiny batch so the first real
    translation doesn't pay the multi-second load / CUDA warm-up cost."""
    if from_code == to_code:
        return True
    t0 = time.perf_counter()
    try:
        translation = _get_translation(from_code, to_code)
        if translation is None:
            return False  # pack not installed (yet)
        _batch_translation(translation, ["Hello world"])
        print(
            f"[Translator] Model {from_code}->{to_code} warmed up in "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms."
        )
        return True
    except Exception as exc:
        print(f"[Translator] Warm-up skipped ({exc})")
        return False
