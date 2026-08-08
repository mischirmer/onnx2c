#!/usr/bin/env python3
"""Focused tests for the im2col Conv selection heuristic."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, helper


@dataclass(frozen=True)
class ConvCase:
    name: str
    x_shape: tuple[int, int, int, int]
    w_shape: tuple[int, int, int, int]
    y_shape: tuple[int, int, int, int]
    group: int
    pads: tuple[int, int, int, int]
    strides: tuple[int, int]
    expected_selected: bool
    expected_reason: str


def make_model(case: ConvCase, path: Path) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, case.x_shape)
    w = helper.make_tensor_value_info("W", TensorProto.FLOAT, case.w_shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, case.y_shape)
    conv = helper.make_node(
        "Conv",
        ["X", "W"],
        ["Y"],
        name=case.name,
        group=case.group,
        kernel_shape=[case.w_shape[2], case.w_shape[3]],
        pads=list(case.pads),
        strides=list(case.strides),
        dilations=[1, 1],
    )
    graph = helper.make_graph([conv], f"{case.name}_graph", [x, w], [y])
    model = helper.make_model(graph, producer_name="onnx2c-im2col-heuristic-test")
    onnx.save(model, path)


def run_case(onnx2c: Path, case: ConvCase, temp_dir: Path) -> None:
    model_path = temp_dir / f"{case.name}.onnx"
    make_model(case, model_path)

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
        raise AssertionError(f"{case.name}: expected {expected_selected} in debug log:\n{log}")
    if case.expected_reason not in log:
        raise AssertionError(f"{case.name}: expected reason '{case.expected_reason}' in debug log:\n{log}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: im2col_heuristic_test.py /path/to/onnx2c", file=sys.stderr)
        return 2

    onnx2c = Path(sys.argv[1])
    cases = [
        ConvCase(
            "conv_1x1_mid_k_stride1_direct",
            (1, 32, 32, 32),
            (64, 32, 1, 1),
            (1, 64, 32, 32),
            1,
            (0, 0, 0, 0),
            (1, 1),
            False,
            "reject: direct Conv favored for skinny or stride-1 mid-K workload",
        ),
        ConvCase(
            "conv_1x1_large_k_im2col",
            (1, 256, 28, 28),
            (64, 256, 1, 1),
            (1, 64, 28, 28),
            1,
            (0, 0, 0, 0),
            (1, 1),
            True,
            "select: GEMM-style im2col path favored",
        ),
        ConvCase(
            "conv_depthwise_3x3_im2col",
            (1, 32, 32, 32),
            (32, 1, 3, 3),
            (1, 32, 30, 30),
            32,
            (0, 0, 0, 0),
            (1, 1),
            True,
            "select: GEMM-style im2col path favored",
        ),
        ConvCase(
            "conv_skinny_k_direct",
            (1, 1, 56, 56),
            (32, 1, 1, 1),
            (1, 32, 56, 56),
            1,
            (0, 0, 0, 0),
            (1, 1),
            False,
            "reject: direct Conv favored for skinny or stride-1 mid-K workload",
        ),
        ConvCase(
            "conv_large_dense_3x3",
            (1, 32, 32, 32),
            (64, 32, 3, 3),
            (1, 64, 30, 30),
            1,
            (0, 0, 0, 0),
            (1, 1),
            True,
            "select: GEMM-style im2col path favored",
        ),
        ConvCase(
            "conv_large_workspace_still_im2col",
            (1, 16, 8, 8),
            (32, 16, 7, 7),
            (1, 32, 8, 8),
            1,
            (3, 3, 3, 3),
            (1, 1),
            True,
            "select: GEMM-style im2col path favored",
        ),
    ]

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for case in cases:
            run_case(onnx2c, case, temp_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
