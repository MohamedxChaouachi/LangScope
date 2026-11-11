"""The statistics scripts/bench_ocr.py prints."""

import importlib.util
import pathlib
import statistics

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "bench_ocr.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_ocr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p95_of_five_samples_is_the_largest(bench):
    # int(n * 0.95) - 1 picked the 4th of 5 samples here (4.0), which is below the mean (22.0).
    samples = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert bench.p95(samples) == 100.0
    assert bench.p95(samples) >= statistics.mean(samples)


def test_p95_uses_the_nearest_rank(bench):
    samples = [float(i) for i in range(1, 101)]
    assert bench.p95(samples) == 95.0  # rank ceil(0.95 * 100) = 95
    assert bench.p95([float(i) for i in range(1, 21)]) == 19.0  # rank ceil(19.0) = 19


def test_p95_does_not_depend_on_the_order_of_the_samples(bench):
    assert bench.p95([5.0, 1.0, 100.0, 3.0, 2.0]) == 100.0


def test_p95_of_one_sample_is_that_sample(bench):
    assert bench.p95([7.5]) == 7.5


def test_summarize_prints_the_nearest_rank_p95(bench, capsys):
    bench.summarize("total", [1.0, 2.0, 3.0, 4.0, 100.0])
    assert "p95=  100.0ms" in capsys.readouterr().out
