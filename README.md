# LangScope

Offline screen translator built with Python and PyQt6.

LangScope captures screenshots of a selected screen region at a set interval, runs OCR on each capture to detect text, translates it locally using Argos Translate, and renders the translated text in-place over where the original text was detected.

No internet connection is required after the initial model download.

---

## How it works

1. You select a region of your screen by clicking and dragging.
2. LangScope takes a screenshot of that region every ~2 seconds.
3. EasyOCR detects and locates text in the screenshot.
4. Each detected text region is translated using Argos Translate (fully offline).
5. The translated text is drawn over the original, covering it with a dark background.

The preview panel updates with each new screenshot so you can see what is being captured and translated.

---

## Features

- In-place translation overlay — translated text is drawn at the exact position of the original
- Font size matched to the detected text bounding box height
- Fully offline after first run — no API keys, no cloud services
- Smart filtering:
  - Skips numbers, timestamps, phone numbers, and dates
  - Skips text already written in the target language (detected via langdetect)
- 11 supported languages: English, French, Spanish, German, Italian, Portuguese, Romanian, Russian, Chinese, Japanese, Arabic
- Resizable window

---

## Requirements

- Python 3.10+
- Windows (tested on Windows 10/11)

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

1. Click "Select Region" and drag over the area you want to translate.
2. Choose source and target languages from the dropdowns.
3. Click "Start".
4. The panel will update with each screenshot cycle, showing the translation overlaid on the capture.

---

## Dependencies

| Package | Purpose |
|---|---|
| PyQt6 | UI framework |
| easyocr | On-device OCR |
| argostranslate | Fully offline neural machine translation |
| langdetect | Language identification |
| mss | Screen capture |
| numpy | Frame processing |

---

## Project Structure

```
LangScope/
├── main.py         # Entry point
├── engine.py       # Capture worker thread (screenshot, OCR, filtering)
├── translator.py   # Argos Translate wrapper (setup, caching, translation)
├── ui.py           # PyQt6 UI (control panel, preview widget, region selector)
└── requirements.txt
```

---

## Notes

- Language packs (~100 MB each) are downloaded once on first use and cached locally under `~/.local/share/argos-translate/`.
- EasyOCR models are also downloaded once (~200 MB for Latin-script languages).
- Non-Latin scripts (Russian, Chinese, Japanese, Arabic) use separate EasyOCR models.
- The screenshot interval is approximately 2 seconds for OCR and 0.3 seconds for the preview refresh.
