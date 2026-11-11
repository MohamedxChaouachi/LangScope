# LangScope

[![CI](https://github.com/MohamedxChaouachi/LangScope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MohamedxChaouachi/LangScope/actions/workflows/ci.yml)

An offline screen translator for Windows: it reads the text in a region of your screen with OCR, translates it locally and draws the translation over the original.

**Highlights**

- **Fully offline.** OCR (EasyOCR) and translation (Argos Translate) run on your machine. After the first model download there is no internet connection, API key or cloud service.
- **In-place overlay.** The translation is drawn over the original text, in the app window or as a click-through overlay on the desktop.
- **A live OCR and translation pipeline.** Threaded capture, change-gated OCR, region tracking and cached, batched translation, across 11 languages.
- **Measured speed.** On a laptop RTX 2050 (4 GB), one OCR pass over a 600 x 300 region takes about 1.1 s (0.9 passes per second); CPU only, about 7.0 s. See [Benchmark](#benchmark).

---

## Quick start

Requires Windows and Python 3.11 or newer.

```bash
git clone https://github.com/MohamedxChaouachi/LangScope.git
cd LangScope
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install .
langscope
```

In the window, choose the **FROM** and **TO** languages, click **Select Region** and drag over the text, then click **Start**. The first start downloads the models it needs and loads them.

For an NVIDIA GPU or DXCam capture, see [Installation](#installation); for every control, [Usage and configuration](#usage-and-configuration).

---

## Usage and configuration

```bash
langscope
```

`python -m langscope` does the same. The command lives in the environment you installed LangScope into, so activate it first or call it by path (for example `.venv\Scripts\langscope`).

1. Choose the source (**FROM**) and target (**TO**) languages. If the language pack is not cached yet, it downloads automatically and the badge next to the dropdown shows progress.
2. Click **Select Region** and drag over the area to translate. A selector opens on each monitor. Press `Esc` or right-click to cancel.
3. Click **Start**. The first start loads the models, which can take a few seconds.
4. Optionally enable **Desktop Overlay** to draw translations over the real screen. The overlay is click-through; untick the checkbox to hide it.
5. Optionally enable **Debug Mode** to see OCR boxes, confidence scores and timing stats.

Changing the language pair while running clears tracked regions and caches, then re-translates.

| Setting | Where | Default |
|---|---|---|
| Log level | `LANGSCOPE_LOGLEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, ...) | `INFO` |
| Translation device | `ARGOS_DEVICE_TYPE` / `ARGOS_COMPUTE_TYPE` environment variables | CUDA if available, else CPU |
| OCR rate | `OCR_FPS` in `src/langscope/ocr_worker.py` | 8 |
| Change-detection sensitivity | `PIXEL_DELTA`, `MIN_CHANGED_PX` in `src/langscope/ocr_worker.py` | 40, 12 |
| OCR preprocessing and confidence | `PreprocessConfig` in `src/langscope/ocr_providers.py` | upscale 1.5x, grayscale, CLAHE, sharpen, confidence 0.35 |
| Region tracking and smoothing | constants at the top of `src/langscope/translation_worker.py` (`IOU_THRESHOLD`, `TEXT_SIM_THRESHOLD`, `EMA_ALPHA`, `MAX_MISSES`, `REGION_TTL`) | see file |

An unknown value for `LANGSCOPE_LOGLEVEL` means `INFO`. For verbose logs:

```powershell
$env:LANGSCOPE_LOGLEVEL = "DEBUG"; langscope
```

---

## How it works

LangScope runs as a small pipeline of threads:

```
CaptureManager (thread, 60 FPS)
      │  latest_frame
      ▼
OCRWorker (QThread, capped at 8 passes per second, only when the screen changed)
      │  list[RawSegment]
      ▼
TranslationWorker (QThread, on demand)
      │  list[TranslationRegion]
      ▼
Render timer (60 FPS)  →  live feed panel  and/or  desktop overlay
```

1. **Capture.** `CaptureManager` grabs the selected region at up to 60 FPS, with DXCam (DXGI desktop duplication) when installed and MSS otherwise. A secondary monitor always uses MSS.
2. **OCR.** `OCRWorker` runs EasyOCR at most 8 times per second (a ceiling, not a measured rate; see [Benchmark](#benchmark)), and only when the frame changed. Latin-script languages load together with the English model, so accented and plain ASCII text are both recognized; Russian, Chinese, Japanese and Arabic each use their own model.
3. **Preprocessing.** Each frame is upscaled 1.5x, converted to grayscale, contrast-enhanced with CLAHE and sharpened. Detections below a 0.35 confidence score are dropped.
4. **Tracking.** `TranslationWorker` matches each detection to an existing region by bounding-box IoU and text similarity, so regions keep their identity across passes, and smooths their boxes so overlays do not jitter.
5. **Translation.** New or changed text goes to Argos/CTranslate2 in one batched call and is cached by text hash. A pair without a direct language pack pivots through English (Romanian to French goes through Romanian to English to French). The first translation after startup or a language change includes a one-time model warm-up.
6. **Rendering.** Translated text is drawn over the original on a dark background, sized to the box height, shrunk to fit the width and word-wrapped.

### Features

- Two display modes: the in-app live feed, and a click-through, always-on-top desktop overlay
- GPU acceleration for EasyOCR (PyTorch CUDA) and for translation (CTranslate2 CUDA, float16), with automatic CPU fallback
- Multi-monitor and mixed-DPI support: the region selector opens on every screen
- Skips numbers, timestamps, dates and phone numbers, and text already in the target language (detected with langdetect)
- Debug mode with raw OCR boxes, region IDs, confidence scores, a live FPS counter, and a stats strip (OCR time split into preprocess and inference, translation time, cache hit rate)
- 11 languages: English, French, Spanish, German, Italian, Portuguese, Romanian, Russian, Chinese, Japanese, Arabic

---

## Benchmark

OCR inference (EasyOCR's CRAFT detector and CRNN recognizer) is the dominant cost per frame, not the threading pipeline. On GPU, recognition runs in batches of 16 boxes and halves the batch size if the card runs out of VRAM.

Measured on a laptop with an Intel Core i5-12450H and an NVIDIA GeForce RTX 2050 (4 GB), Windows, Python 3.14, PyTorch 2.14.1+cu126. EasyOCR, lang=en, FP32, on a region full of English text. The first model load and warm-up are excluded from the timing samples.

| Device | Region | Frames | Time per OCR pass, mean / median (ms) | OCR passes per second |
|---|---|---:|---:|---:|
| NVIDIA GeForce RTX 2050 laptop GPU (4 GB) | 600 x 300 | 10 | 1134.8 / 1145.9 | 0.9 |
| NVIDIA GeForce RTX 2050 laptop GPU (4 GB) | 600 x 900 | 40 | 5907.8 / 6152.9 | 0.2 |
| Intel Core i5-12450H, CPU only | 600 x 300 | 5 | 6970.7 / 6875.3 | 0.1 |

- Preprocessing takes about 3 ms (14 ms for the 600 x 900 region); almost all the time is EasyOCR inference.
- On this hardware OCR is the bottleneck, at about one pass per second on the GPU for a 600 x 300 region, so the 8-per-second OCR cap is not reached. Smaller regions update faster.

Run these commands in PowerShell from the repository root with the project environment active. The selected rectangle must fit on a connected monitor.

```powershell
# CPU: hide the GPU from PyTorch for this process. "-1" matters: setting the variable to ""
# deletes it in PowerShell, so the GPU would still be used.
$env:CUDA_VISIBLE_DEVICES = "-1"; .\.venv\Scripts\python.exe -u scripts\bench_ocr.py --frames 40 --lang en --top 150 --left 660 --width 600 --height 300

# CUDA FP32: remove the override so PyTorch can select the GPU.
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue; .\.venv\Scripts\python.exe -u scripts\bench_ocr.py --frames 40 --lang en --top 150 --left 660 --width 600 --height 300

# CUDA FP16: experimental; use only after the FP32 run succeeds.
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue; .\.venv\Scripts\python.exe -u scripts\bench_ocr.py --frames 40 --lang en --top 150 --left 660 --width 600 --height 300 --fp16
```

The harness reports preprocess, inference and total mean/median/p95 latency and the maximum sustained rate. FP16 is opt-in and may fail with a dtype mismatch in EasyOCR; the app's default remains FP32.

---

## Design decisions & trade-offs

- **Polling instead of per-frame signals.** Workers replace their latest-frame and immutable-region snapshots, and the 60 FPS render timer polls them and redraws only when either changed. This avoids queuing high-rate signals and repainting identical frames, at the cost of skipping intermediate states when workers run faster than rendering.
- **Pixel-count change detection.** The OCR gate skips a frame whose hash is identical, then compares every other pixel in each dimension and counts those whose largest channel difference exceeds 40; at least 12 must change. This ignores compression noise and blinking cursors, while a changed text line affects enough pixels to trigger OCR. Tiny or low-contrast changes can be missed.
- **IoU plus text similarity.** Matching requires IoU of at least 0.30 and scores candidates 60% overlap, 40% text similarity, against a combined threshold. Position alone can merge neighboring lines and text alone can confuse repeated words, so both count.
- **EMA box smoothing.** Bounding boxes use an exponential moving average with `EMA_ALPHA = 0.35`. This reduces jitter between OCR passes but makes the overlay follow rapid movement with a little lag.
- **Batched CTranslate2 translation.** Lines are grouped into model batches, keeping alignment and cutting per-call overhead; pivot pairs take two batched hops. This reads Argos internals, so the dependency is version-bounded and covered by a contract test.
- **EasyOCR and Argos.** Both stay local once their models and packs are downloaded. The trade-offs are a substantial PyTorch install, download and storage needs, CPU latency, and keeping up with Argos internals as they change.

---

## Known limitations

- Windows is the supported and tested desktop platform. Capture behavior on other operating systems is not verified.
- Very small on-screen changes can wait until enough pixels differ. CPU-only inference is slow (see [Benchmark](#benchmark)); smaller capture regions help.
- Regions are selected on one monitor at a time; a capture region cannot span monitors. Secondary-monitor capture uses MSS rather than DXCam.
- Models download on first use: Argos packs are around 100 MB each (cached under `~/.local/share/argos-translate/`), and EasyOCR models around 200 MB for Latin-script languages.
- FP16 OCR is experimental and disabled by default; EasyOCR may encounter a dtype mismatch on some configurations.

---

## Installation

Requirements: Windows (tested on Windows 10/11) and Python 3.11 or newer. An NVIDIA GPU is optional but recommended.

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

**Newer Python versions.** When this was checked, the CUDA 12.1 build had no wheels for Python 3.13 or 3.14, and the CUDA 11.8 build had none for 3.14. On those versions use the CPU build or a newer CUDA build such as `cu126`, `cu128` or `cu130`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install LangScope itself. The optional `dxcam` extra gives lower-latency capture, and an editable install with the `dev` extra is for working on it (extras combine, as in `".[dev,dxcam]"`):

```bash
pip install .
pip install ".[dxcam]"
pip install -e ".[dev]"
```

### Dependencies

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

## Development

Create and activate a virtual environment, then install the development extra. Install the CPU or CUDA PyTorch build first as described in [Installation](#installation), because PyTorch uses a custom package index.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Run the checks from the activated environment:

```powershell
ruff check .
ruff format --check .
pytest
pyright
```

CI installs the lighter `requirements-ci.txt` instead, without PyTorch, EasyOCR or Argos Translate; tests that need them skip with a reason. The tests cover region tracking, change detection, batched translation (including pivoting through English), coordinate conversions, the region selector, controller wiring, startup configuration and packaging, and smoke tests build and paint the Qt widgets offscreen, so no display is needed. CI runs Ruff, builds and checks the wheel, then runs pytest and Pyright on Windows with Python 3.11 and 3.14 ([workflow](.github/workflows/ci.yml)).

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

## Roadmap

- Additional OCR and translation engines behind the existing `BaseOCR` / `BaseTranslator` interfaces
- Non-Windows support
- Prebuilt Windows release
