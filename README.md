# LangScope

[![CI](https://github.com/MohamedxChaouachi/LangScope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MohamedxChaouachi/LangScope/actions/workflows/ci.yml)

Offline live screen translator built with Python and PyQt6.

LangScope continuously captures a region of your screen, runs OCR on it when the content changes, translates the detected text locally with Argos Translate, and draws the translation in place over the original text. It can render inside the app window or as a click-through overlay directly on your desktop.

No internet connection is needed after the initial model downloads. There are no API keys and no cloud services.

---

## How it works

LangScope runs as a small pipeline of threads:

```
CaptureManager (thread, 60 FPS)
      │  latest_frame
      ▼
OCRWorker (QThread, up to 8 FPS, only when the screen changed)
      │  list[RawSegment]
      ▼
TranslationWorker (QThread, on demand)
      │  list[TranslationRegion]
      ▼
Render timer (60 FPS)  →  live feed panel  and/or  desktop overlay
```

1. **Capture.** `CaptureManager` grabs the selected region at up to 60 FPS. It uses DXCam (DXGI desktop duplication) when installed and falls back to MSS. Regions on a secondary monitor are always grabbed with MSS.
2. **OCR.** `OCRWorker` runs EasyOCR at up to 8 FPS, but only when the frame changed. It skips a frame if the frame hash is identical or if fewer than 12 pixels changed noticeably (so a blinking caret or compression noise does not trigger OCR).
3. **Preprocessing.** Before OCR each frame is upscaled 1.5x, converted to grayscale, contrast-enhanced with CLAHE and sharpened. Detections below a 0.35 confidence score are dropped.
4. **Tracking.** `TranslationWorker` matches each detection to an existing region by bounding-box IoU and text similarity. Matched regions keep their identity across OCR passes and their boxes are smoothed with an exponential moving average, so overlays do not jitter.
5. **Translation.** New or changed text is translated in a single batched Argos/CTranslate2 call. Results are cached by text hash, so unchanged text is never translated twice.
6. **Rendering.** Translated text is drawn over the original on a dark background. Font size is matched to the box height, shrunk to fit the width, and word-wrapped if needed.

---

## Features

- In-place translation overlay, positioned at the exact location of the original text
- Two display modes: the in-app live feed, and a click-through, always-on-top desktop overlay
- Fully offline after the first run
- GPU acceleration for EasyOCR (PyTorch CUDA) and for translation (CTranslate2 CUDA, float16), with automatic CPU fallback
- Multi-monitor and mixed-DPI support: the region selector opens on every screen
- Change detection, so OCR and translation do not run on a static screen
- Smart filtering:
  - Skips numbers, timestamps, dates and phone numbers
  - Skips text already in the target language (detected with langdetect)
- Debug mode: shows raw OCR boxes, region IDs, confidence scores, and a stats strip (OCR time split into preprocess and inference, translation time, cache hit rate)
- Live FPS counter
- 11 languages: English, French, Spanish, German, Italian, Portuguese, Romanian, Russian, Chinese, Japanese, Arabic

---

## Requirements

- Python 3.11 or newer
- Windows (tested on Windows 10/11)
- An NVIDIA GPU is optional but recommended

### Installation

Install PyTorch first. The right build depends on your hardware, so it is not one of LangScope's dependencies.

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for other CUDA versions.

**Newer Python versions.** PyTorch does not publish every CUDA build for every Python version. When this was checked, the CUDA 12.1 build above had no wheels for Python 3.13 or 3.14, and the CUDA 11.8 build had none for 3.14. On those versions, use the CPU build or a newer CUDA build such as `cu126`, `cu128` or `cu130`, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install LangScope itself:

```bash
pip install .
```

To work on it, install it in editable mode together with the test and lint tools:

```bash
pip install -e ".[dev]"
```

Optional, for lower-latency capture on Windows (extras can be combined, as in `".[dev,dxcam]"`):

```bash
pip install ".[dxcam]"
```

---

## Usage

```bash
langscope
```

`python -m langscope` does the same. The `langscope` command is installed into the Python environment you installed LangScope in, so activate that environment first, or call it by path (for example `.venv\Scripts\langscope`).

1. Choose the source (**FROM**) and target (**TO**) languages. If the language pack is not cached yet, it downloads automatically and the badge next to the dropdown shows progress.
2. Click **Select Region** and drag over the area to translate. A selector opens on each monitor. Press `Esc` or right-click to cancel.
3. Click **Start**. The first start loads the OCR and translation models, which can take a few seconds.
4. Optionally enable **Desktop Overlay** to draw translations over the real screen. The overlay is click-through; untick the checkbox to hide it.
5. Optionally enable **Debug Mode** to see OCR boxes, confidence scores and timing stats.

Changing the language pair while running clears tracked regions and caches, then re-translates.

### Configuration

| Setting | Where | Default |
|---|---|---|
| Log level | `LANGSCOPE_LOGLEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, ...) | `INFO` |
| Translation device | `ARGOS_DEVICE_TYPE` / `ARGOS_COMPUTE_TYPE` environment variables | CUDA if available, else CPU |
| OCR rate | `OCR_FPS` in `src/langscope/ocr_worker.py` | 8 |
| Change-detection sensitivity | `PIXEL_DELTA`, `MIN_CHANGED_PX` in `src/langscope/ocr_worker.py` | 40, 12 |
| OCR preprocessing and confidence | `PreprocessConfig` in `src/langscope/ocr_providers.py` | upscale 1.5x, grayscale, CLAHE, sharpen, confidence 0.35 |
| Region tracking and smoothing | constants at the top of `src/langscope/translation_worker.py` (`IOU_THRESHOLD`, `TEXT_SIM_THRESHOLD`, `EMA_ALPHA`, `MAX_MISSES`, `REGION_TTL`) | see file |

An unknown value for `LANGSCOPE_LOGLEVEL` means `INFO`.

Example, for verbose logs:

```powershell
$env:LANGSCOPE_LOGLEVEL = "DEBUG"; langscope
```

---

## Project structure

```
LangScope/
├── src/langscope/
│   ├── main.py                  # Entry point: reads the config, sets up logging and the backends, runs the app
│   ├── __main__.py              # `python -m langscope`
│   ├── controller.py            # AppController: wires the panel to the pipeline and owns its lifecycle
│   ├── config.py                # The log level, from LANGSCOPE_LOGLEVEL
│   ├── capture_manager.py       # 60 FPS screen capture thread (DXCam, MSS fallback)
│   ├── ocr_worker.py            # OCR thread; FrameChangeGate decides whether a frame is worth OCR
│   ├── ocr_providers.py         # BaseOCR interface, EasyOCR provider, preprocessing pipeline
│   ├── translation_worker.py    # Region tracking, bbox smoothing, filtering, batched translation
│   ├── translation_providers.py # BaseTranslator and LanguagePacks interfaces, with Argos implementations
│   ├── translator.py            # Argos Translate setup, device configuration, language packs, batched translation
│   ├── models.py                # RawSegment, TranslationRegion, RegionSnapshot, and the BBox and Region types
│   ├── render_gate.py           # Decides what the 60 FPS render timer needs to redraw
│   ├── ui.py                    # Control panel, live feed widget, multi-monitor region selector
│   ├── overlay_window.py        # Click-through desktop overlay
│   ├── render_utils.py          # Shared text-drawing code for the live feed and overlay
│   ├── screen_utils.py          # Native/logical pixel conversion for multi-monitor and HiDPI
│   ├── theme.py                 # UI colors, fonts and spacing
│   └── assets/                  # App icon and UI images (shipped as package data)
├── scripts/
│   ├── bench_ocr.py             # OCR benchmark harness
│   └── check_wheel.py           # Checks that a built wheel has its assets and entry point (used by CI)
├── tests/                       # pytest suite
├── .github/workflows/           # CI: ruff, wheel build and check, pytest, pyright
├── pyproject.toml               # Packaging metadata, dependencies and tool configuration
├── requirements-ci.txt          # The lightweight set CI installs (no torch, easyocr or argostranslate)
└── LICENSE                      # MIT
```

`BaseOCR` and `BaseTranslator` are the extension points: add a new engine by implementing the interface, without touching the rest of the pipeline.

---

## Dependencies

| Package | Purpose |
|---|---|
| PyQt6 | UI framework |
| qtawesome | UI icons |
| easyocr (with PyTorch) | OCR, optional CUDA acceleration |
| argostranslate | Offline neural machine translation (CTranslate2) |
| langdetect | Language identification |
| mss | Screen capture (fallback and secondary monitors) |
| numpy | Frame processing |
| dxcam (optional) | Faster capture on the primary monitor |

OpenCV (`opencv-python-headless`, imported as `cv2`) is used for preprocessing. The `argostranslate` range is bounded on both sides (`>=1.10.0,<1.12`) because `translator.py` reads some of Argos' internals to translate in batches; `tests/test_argos_contract.py` checks them.

---

## Testing

```bash
# Full environment (after installing PyTorch as described above)
pip install -e ".[dev]"

# Or the lightweight set CI uses, without torch, easyocr or argostranslate
pip install -r requirements-ci.txt

ruff check .
ruff format --check .
pyright
pytest
```

The tests cover region tracking and pruning, change detection, batched translation (including pivoting through English), the coordinate conversions, the region selector, the controller's wiring, the start-up configuration and the packaging metadata. Smoke tests import every module and build and paint the Qt widgets offscreen, so they need no display. Tests that need argostranslate skip themselves, with a reason, when it is not installed; `pytest` lists each skip.

The tests import the package from `src/` (pytest's `pythonpath` setting), so they run in a fresh clone without installing LangScope.

CI runs `ruff`, builds the wheel and checks its contents, then runs `pytest` and `pyright` on Windows with Python 3.11 and 3.14 (`.github/workflows/ci.yml`).

---

## Performance

OCR inference (EasyOCR's CRAFT detector and CRNN recognizer) is the dominant cost per frame, not the threading pipeline. On GPU, recognition runs in batches of 16 boxes and halves the batch size automatically if the card runs out of VRAM. The debug stats strip shows the live preprocess/inference split.

To measure your own machine and capture region:

```bash
python scripts/bench_ocr.py --frames 30
python scripts/bench_ocr.py --frames 40 --lang fr --top 100 --left 100 --width 600 --height 300
```

It reports preprocess and inference time, mean, median and p95 latency, and the sustained FPS your hardware can deliver. Use it before tuning `OCR_FPS` in `src/langscope/ocr_worker.py` or `PreprocessConfig` in `src/langscope/ocr_providers.py`. A smaller capture region is the most effective way to cut OCR time.

---

## Notes

- Argos language packs (~100 MB each) are downloaded once and cached under `~/.local/share/argos-translate/`. For pairs without a direct pack, Argos pivots through English (for example Romanian to French goes through Romanian to English to French).
- EasyOCR models are downloaded once on first use (~200 MB for Latin-script languages). Russian, Chinese, Japanese and Arabic each use their own model.
- Latin-script languages are loaded together with the English model so accented characters and plain ASCII are both recognized.
- The first translation after startup or a language change includes a one-time model warm-up.

---

## Roadmap

- Additional OCR and translation engines behind the existing `BaseOCR` / `BaseTranslator` interfaces
- Non-Windows support
