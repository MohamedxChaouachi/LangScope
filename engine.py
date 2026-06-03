"""
engine.py — DEPRECATED

This file is kept as a shim for backwards compatibility only.
The monolithic CaptureWorker has been replaced by the three-stage pipeline:

    capture_manager.py   — CaptureManager  (60 FPS screen grab daemon)
    ocr_worker.py        — OCRWorker        (8 FPS OCR QThread)
    translation_worker.py — TranslationWorker (on-demand translation QThread)

This file is no longer imported by main.py and will be removed in a future cleanup.
"""
# Nothing to import — engine.py is no longer used.
