"""
Start-up configuration of the two backends: importing their modules changes nothing, and the
explicit init functions do the work.

    translator.init_translation_backend()   Argos' data directories and the inference device
    ocr_providers.init_ocr_backend()        the GPU probe and the OCR batch size

The ML packages are replaced by fakes, so these run without them (as in CI) and never touch
the real Argos data directory or the GPU.
"""

import importlib
import logging
import pathlib
import sys
import types

import pytest

import ocr_providers
from ocr_providers import BACKEND, EasyOCRProvider, init_ocr_backend


class TestTranslationBackend:
    @pytest.fixture(autouse=True)
    def no_device_overrides(self, monkeypatch):
        monkeypatch.delenv("ARGOS_DEVICE_TYPE", raising=False)
        monkeypatch.delenv("ARGOS_COMPUTE_TYPE", raising=False)

    @pytest.fixture
    def gpu(self, fake_argos, monkeypatch):
        """A CUDA GPU that supports float16, as seen through the fake ctranslate2."""
        ct2 = fake_argos.modules["ctranslate2"]
        monkeypatch.setattr(ct2, "get_cuda_device_count", lambda: 1)
        monkeypatch.setattr(ct2, "get_supported_compute_types", lambda device: {"float16", "int8"})
        return ct2

    def test_importing_the_module_creates_nothing(self, fake_argos, translator_module):
        assert not (fake_argos.data_dir / "packages").exists()

    def test_importing_the_module_changes_no_setting_even_with_a_gpu(
        self, fake_argos, gpu, monkeypatch
    ):
        # The GPU has to be there before the import, or an import-time switch would not show.
        monkeypatch.delitem(sys.modules, "translator", raising=False)
        try:
            importlib.import_module("translator")
        finally:
            sys.modules.pop("translator", None)
        settings = fake_argos.settings
        assert (settings.device, settings.compute_type) == ("cpu", "auto")

    def test_init_creates_the_packages_directory(self, fake_argos, translator_module):
        translator_module.init_translation_backend()
        assert (fake_argos.data_dir / "packages").is_dir()

    def test_init_gives_the_data_settings_the_types_argos_expects(
        self, fake_argos, translator_module
    ):
        # data_dir is a Path (Argos applies / to it); package_data_dir is a str.
        fake_argos.settings.data_dir = str(fake_argos.data_dir)
        fake_argos.settings.package_data_dir = fake_argos.data_dir / "packages"
        translator_module.init_translation_backend()
        assert fake_argos.settings.data_dir == fake_argos.data_dir
        assert isinstance(fake_argos.settings.data_dir, pathlib.Path)
        assert fake_argos.settings.package_data_dir == str(fake_argos.data_dir / "packages")

    def test_init_twice_changes_nothing_more(self, fake_argos, translator_module, gpu):
        translator_module.init_translation_backend()
        first = (fake_argos.settings.device, fake_argos.settings.compute_type)
        translator_module.init_translation_backend()
        assert (fake_argos.settings.device, fake_argos.settings.compute_type) == first

    def test_a_gpu_with_float16_runs_translation_on_cuda_in_float16(
        self, fake_argos, translator_module, gpu
    ):
        translator_module.init_translation_backend()
        assert fake_argos.settings.device == "cuda"
        assert fake_argos.settings.compute_type == "float16"

    def test_a_gpu_without_float16_keeps_the_default_compute_type(
        self, fake_argos, translator_module, gpu, monkeypatch
    ):
        monkeypatch.setattr(gpu, "get_supported_compute_types", lambda device: {"int8"})
        translator_module.init_translation_backend()
        assert fake_argos.settings.device == "cuda"
        assert fake_argos.settings.compute_type == "auto"

    def test_a_compute_type_from_the_environment_is_left_alone(
        self, fake_argos, translator_module, gpu, monkeypatch
    ):
        monkeypatch.setenv("ARGOS_COMPUTE_TYPE", "int8")
        translator_module.init_translation_backend()
        assert fake_argos.settings.device == "cuda"
        assert fake_argos.settings.compute_type == "auto"  # Argos itself reads the variable

    def test_a_device_from_the_environment_is_left_alone(
        self, fake_argos, translator_module, gpu, monkeypatch
    ):
        monkeypatch.setenv("ARGOS_DEVICE_TYPE", "cpu")
        translator_module.init_translation_backend()
        assert fake_argos.settings.device == "cpu"
        assert fake_argos.settings.compute_type == "auto"

    def test_without_a_gpu_translation_stays_on_the_cpu(self, fake_argos, translator_module):
        translator_module.init_translation_backend()
        assert fake_argos.settings.device == "cpu"
        assert fake_argos.settings.compute_type == "auto"

    def test_a_failing_gpu_probe_falls_back_to_the_cpu(
        self, fake_argos, translator_module, monkeypatch, caplog
    ):
        def broken():
            raise RuntimeError("the CUDA driver is too old")

        monkeypatch.setattr(fake_argos.modules["ctranslate2"], "get_cuda_device_count", broken)
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translator_module.init_translation_backend()  # must not raise
        assert fake_argos.settings.device == "cpu"
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "the CUDA driver is too old" in warning.getMessage()

    def test_the_chosen_device_is_logged(self, fake_argos, translator_module, gpu, caplog, capsys):
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translator_module.init_translation_backend()
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert "[Translator] Device: CUDA (float16)" in messages
        assert capsys.readouterr().out == ""

    def test_a_device_from_the_environment_is_logged_as_such(
        self, fake_argos, translator_module, caplog, monkeypatch
    ):
        monkeypatch.setenv("ARGOS_DEVICE_TYPE", "cpu")
        with caplog.at_level(logging.INFO, logger="langscope.translator"):
            translator_module.init_translation_backend()
        assert "[Translator] Device (from env): cpu" in [r.getMessage() for r in caplog.records]


class FakeTorch(types.ModuleType):
    """torch as far as the GPU probe sees it. `touched` records every attribute read."""

    def __init__(self, cuda_available):
        super().__init__("torch")
        self.touched = []
        self._cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available, get_device_name=lambda index: "Fake GPU"
        )

    def __getattr__(self, name):
        self.touched.append(name)
        if name == "cuda":
            return self._cuda
        raise AttributeError(name)


class TestOcrBackend:
    @pytest.fixture(autouse=True)
    def restore_backend(self, monkeypatch):
        # Registers the current values, so they come back after a test that calls init.
        monkeypatch.setattr(BACKEND, "use_gpu", BACKEND.use_gpu)
        monkeypatch.setattr(BACKEND, "batch_size", BACKEND.batch_size)

    def test_importing_the_module_does_not_touch_torch(self, monkeypatch):
        torch = FakeTorch(cuda_available=True)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.delitem(sys.modules, "ocr_providers")
        fresh = importlib.import_module("ocr_providers")
        assert torch.touched == []
        assert (fresh.BACKEND.use_gpu, fresh.BACKEND.batch_size) == (False, 1)

    def test_with_a_gpu_ocr_runs_on_it_in_batches_of_16(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=True))
        assert init_ocr_backend() is BACKEND
        assert (BACKEND.use_gpu, BACKEND.batch_size) == (True, 16)

    def test_without_a_gpu_ocr_runs_on_the_cpu_one_box_at_a_time(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=False))
        init_ocr_backend()
        assert (BACKEND.use_gpu, BACKEND.batch_size) == (False, 1)

    def test_a_failing_probe_means_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError
        init_ocr_backend()
        assert (BACKEND.use_gpu, BACKEND.batch_size) == (False, 1)

    def test_the_probe_result_is_logged_as_info(self, monkeypatch, caplog):
        with caplog.at_level(logging.INFO, logger="langscope.ocr_providers"):
            monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=True))
            init_ocr_backend()
            monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=False))
            init_ocr_backend()
        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (logging.INFO, "[OCR] CUDA GPU: Fake GPU"),
            (logging.INFO, "[OCR] No CUDA GPU — CPU fallback."),
        ]

    def test_a_failing_probe_is_logged_as_a_warning_with_its_reason(self, monkeypatch, caplog):
        class BrokenTorch(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("the driver is too old")

        monkeypatch.setitem(sys.modules, "torch", BrokenTorch("torch"))
        with caplog.at_level(logging.INFO, logger="langscope.ocr_providers"):
            init_ocr_backend()
        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert record.getMessage() == "[OCR] GPU probe failed (the driver is too old) — CPU."

    def test_init_can_go_back_to_the_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=True))
        init_ocr_backend()
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=False))
        init_ocr_backend()
        assert (BACKEND.use_gpu, BACKEND.batch_size) == (False, 1)

    def test_a_provider_takes_its_batch_size_from_the_backend(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=True))
        init_ocr_backend()
        assert EasyOCRProvider()._batch_size == 16
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda_available=False))
        init_ocr_backend()
        assert EasyOCRProvider()._batch_size == 1

    def test_the_module_still_exposes_the_fp16_switch(self):
        assert ocr_providers.USE_FP16 is False
