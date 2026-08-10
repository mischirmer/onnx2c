#!/usr/bin/env python3
"""Focused tests for the im2col Conv selection heuristic.

This intentionally avoids depending on the Python onnx package. The benchmark
Conv fixtures are already committed and are enough to exercise the selector.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    model_dir: str
    expected_selected: bool
    expected_reason: str


def run_case(onnx2c: Path, benchmarks_dir: Path, case: Case) -> None:
    model_path = benchmarks_dir / case.model_dir / "model.onnx"
    if not model_path.exists():
        raise AssertionError(f"{case.model_dir}: missing benchmark fixture {model_path}")

    result = subprocess.run(
        [str(onnx2c), "-p", "im2col", "-l", "3", str(model_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    log = result.stderr
    expected_selected = "selected=yes" if case.expected_selected else "selected=no"
    if expected_selected not in log:
        raise AssertionError(f"{case.model_dir}: expected {expected_selected} in debug log:\n{log}")
    if case.expected_reason not in log:
        raise AssertionError(f"{case.model_dir}: expected reason '{case.expected_reason}' in debug log:\n{log}")


def run_explicit_generation_case(onnx2c: Path, benchmarks_dir: Path) -> None:
    model_path = benchmarks_dir / "benchmark_conv_resnet_3x3" / "model.onnx"
    if not model_path.exists():
        raise AssertionError(f"missing benchmark fixture {model_path}")

    result = subprocess.run(
        [str(onnx2c), "-p", "im2col_explicit_all", "-l", "3", str(model_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    generated = result.stdout
    log = result.stderr
    if "Explicit Im2Col materialization" not in generated:
        raise AssertionError(f"explicit policy did not generate materialized im2col code:\n{generated[:2000]}")
    if "calloc" not in generated or "free(x_col)" not in generated:
        raise AssertionError("explicit policy should allocate and free the x_col materialization buffer")
    if "Selected implementation: explicit/materialized im2col" not in log:
        raise AssertionError(f"explicit policy did not report explicit selection:\n{log}")


def run_explicit_heuristic_generation_case(onnx2c: Path, benchmarks_dir: Path) -> None:
    model_path = benchmarks_dir / "benchmark_conv_resnet_3x3" / "model.onnx"
    if not model_path.exists():
        raise AssertionError(f"missing benchmark fixture {model_path}")

    result = subprocess.run(
        [str(onnx2c), "-p", "im2col_explicit_heuristic", "-l", "3", str(model_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if "selected=yes" not in result.stderr:
        raise AssertionError(f"explicit heuristic did not select the representative Conv:\n{result.stderr}")
    if "Explicit Im2Col materialization" not in result.stdout:
        raise AssertionError("explicit heuristic should emit materialized im2col code when the heuristic selects Conv")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: im2col_heuristic_test.py /path/to/onnx2c /path/to/test/benchmarks", file=sys.stderr)
        return 2

    onnx2c = Path(sys.argv[1])
    benchmarks_dir = Path(sys.argv[2])
    cases = [
        Case(
            "benchmark_conv_mobilenet_pointwise_1x1",
            False,
            "reject: direct Conv favored for skinny or stride-1 mid-K workload",
        ),
        Case(
            "benchmark_conv_yolov6n_lastconv",
            False,
            "reject: direct Conv favored for skinny or stride-1 mid-K workload",
        ),
        Case(
            "benchmark_conv_resnet_bottleneck_1x1",
            True,
            "select: GEMM-style im2col path favored",
        ),
        Case(
            "benchmark_conv_mobilenet_depthwise_3x3",
            True,
            "select: GEMM-style im2col path favored",
        ),
        Case(
            "benchmark_conv_resnet_3x3",
            True,
            "select: GEMM-style im2col path favored",
        ),
    ]

    for case in cases:
        run_case(onnx2c, benchmarks_dir, case)
    run_explicit_generation_case(onnx2c, benchmarks_dir)
    run_explicit_heuristic_generation_case(onnx2c, benchmarks_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
