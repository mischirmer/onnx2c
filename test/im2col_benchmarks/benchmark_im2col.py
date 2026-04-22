#!/usr/bin/env python3
"""Benchmark one-node Conv-like models with and without im2col."""

from __future__ import annotations

import argparse
import functools
import operator
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"

TYPE_MAP = {
    1: "float",
    2: "uint8_t",
    3: "int8_t",
    4: "uint16_t",
    5: "int16_t",
    6: "int32_t",
    7: "int64_t",
    10: "_Float16",
    11: "double",
    12: "uint32_t",
    13: "uint64_t",
}

WRAPPER_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef INPUT_ELEMS
#error INPUT_ELEMS must be defined
#endif
#ifndef OUTPUT_ELEMS
#error OUTPUT_ELEMS must be defined
#endif
#ifndef INPUT_CTYPE
#error INPUT_CTYPE must be defined
#endif
#ifndef OUTPUT_CTYPE
#error OUTPUT_CTYPE must be defined
#endif

extern void entry(const INPUT_CTYPE* input, OUTPUT_CTYPE* output);

int main(int argc, char** argv) {
    int iterations = 1000;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    INPUT_CTYPE* input = malloc((size_t)INPUT_ELEMS * sizeof(INPUT_CTYPE));
    OUTPUT_CTYPE* output = malloc((size_t)OUTPUT_ELEMS * sizeof(OUTPUT_CTYPE));
    if (!input || !output) {
        fprintf(stderr, "allocation failed\n");
        free(input);
        free(output);
        return 1;
    }

    for (int i = 0; i < INPUT_ELEMS; i++) {
        input[i] = (INPUT_CTYPE)((i % 23) + 1);
    }

    for (int i = 0; i < 5; i++) {
        entry(input, output);
    }

    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        entry(input, output);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_ms = (elapsed * 1000.0) / iterations;

    printf("Iterations: %d\n", iterations);
    printf("Total time: %.6f seconds\n", elapsed);
    printf("Average per iteration: %.6f ms\n", avg_ms);
    printf("Throughput: %.2f iterations/sec\n", iterations / elapsed);

    free(input);
    free(output);
    return 0;
}
"""


@dataclass(frozen=True)
class ModelMeta:
    input_ctype: str
    input_elems: int
    output_ctype: str
    output_elems: int


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    baseline_ms: float
    im2col_ms: float

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.im2col_ms


def run(cmd: list[str], *, stdout_path: Path | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    stdout = subprocess.PIPE
    handle = None
    if stdout_path is not None:
        handle = stdout_path.open("w")
        stdout = handle
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {' '.join(cmd)}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
    finally:
        if handle is not None:
            handle.close()


def elem_count(value: onnx.ValueInfoProto) -> int:
    dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
    return functools.reduce(operator.mul, dims, 1)


def ctype(value: onnx.ValueInfoProto) -> str:
    elem_type = value.type.tensor_type.elem_type
    if elem_type not in TYPE_MAP:
        raise ValueError(f"unsupported tensor type {elem_type}")
    return TYPE_MAP[elem_type]


def model_meta(model_path: Path) -> ModelMeta:
    model = onnx.load(model_path)
    initializers = {tensor.name for tensor in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in initializers]
    if len(graph_inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"{model_path.name}: expected one non-initializer input and one output")

    return ModelMeta(
        input_ctype=ctype(graph_inputs[0]),
        input_elems=elem_count(graph_inputs[0]),
        output_ctype=ctype(model.graph.output[0]),
        output_elems=elem_count(model.graph.output[0]),
    )


def ensure_models() -> None:
    if any(SCRIPT_DIR.glob("*.onnx")):
        return
    run([sys.executable, str(SCRIPT_DIR / "generate_models.py")])


def write_wrapper(out_dir: Path) -> Path:
    wrapper = out_dir / "benchmark_wrapper.c"
    wrapper.write_text(WRAPPER_C)
    return wrapper


def generate_c(onnx2c: Path, model: Path, output: Path, *, im2col: bool) -> None:
    cmd = [str(onnx2c), "-l", "0"]
    if im2col:
        cmd.extend(["-p", "im2col"])
    cmd.append(str(model))
    run(cmd, stdout_path=output)


def compile_binary(gcc: str, source: Path, wrapper: Path, binary: Path, meta: ModelMeta) -> None:
    cmd = [
        gcc,
        "-O3",
        f"-DINPUT_CTYPE={meta.input_ctype}",
        f"-DOUTPUT_CTYPE={meta.output_ctype}",
        f"-DINPUT_ELEMS={meta.input_elems}",
        f"-DOUTPUT_ELEMS={meta.output_elems}",
        "-o",
        str(binary),
        str(source),
        str(wrapper),
        "-lm",
    ]
    run(cmd)


def parse_avg_ms(output: str) -> float:
    match = re.search(r"Average per iteration:\s+([0-9.]+)\s+ms", output)
    if not match:
        raise ValueError(f"could not parse benchmark output:\n{output}")
    return float(match.group(1))


def run_binary(binary: Path, iterations: int) -> float:
    result = run([str(binary), str(iterations)])
    return parse_avg_ms(result.stdout)


def benchmark_model(args: argparse.Namespace, model: Path, wrapper: Path) -> BenchmarkResult:
    name = model.stem
    meta = model_meta(model)

    baseline_c = args.out_dir / f"{name}_baseline.c"
    im2col_c = args.out_dir / f"{name}_im2col.c"
    baseline_bin = args.out_dir / f"{name}_baseline"
    im2col_bin = args.out_dir / f"{name}_im2col"

    generate_c(args.onnx2c, model, baseline_c, im2col=False)
    generate_c(args.onnx2c, model, im2col_c, im2col=True)
    compile_binary(args.gcc, baseline_c, wrapper, baseline_bin, meta)
    compile_binary(args.gcc, im2col_c, wrapper, im2col_bin, meta)

    baseline_ms = run_binary(baseline_bin, args.iterations)
    im2col_ms = run_binary(im2col_bin, args.iterations)
    return BenchmarkResult(name, baseline_ms, im2col_ms)


def print_table(results: list[BenchmarkResult]) -> None:
    rows = [
        ("model", "baseline ms", "im2col ms", "speedup"),
        ("-----", "-----------", "---------", "-------"),
    ]
    for result in results:
        rows.append(
            (
                result.model,
                f"{result.baseline_ms:.6f}",
                f"{result.im2col_ms:.6f}",
                f"{result.speedup:.2f}x",
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for idx, row in enumerate(rows):
        print(
            f"{row[0]:<{widths[0]}}  "
            f"{row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  "
            f"{row[3]:>{widths[3]}}"
        )
        if idx == 1:
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", "-n", type=int, default=1000, help="Measured entry() calls per binary.")
    parser.add_argument("--build-dir", type=Path, default=Path(os.environ.get("BUILD_DIR", DEFAULT_BUILD_DIR)))
    parser.add_argument("--onnx2c", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gcc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--model", action="append", default=[], help="Run only models whose filename stem contains this string.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.onnx2c = args.onnx2c or args.build_dir / "onnx2c"
    args.out_dir = args.out_dir or args.build_dir / "im2col_benchmarks"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.onnx2c.exists():
        raise SystemExit(f"onnx2c not found: {args.onnx2c}")

    ensure_models()
    models = sorted(SCRIPT_DIR.glob("*.onnx"))
    if args.model:
        models = [model for model in models if any(pattern in model.stem for pattern in args.model)]
    if not models:
        raise SystemExit("no benchmark models selected")

    wrapper = write_wrapper(args.out_dir)
    results = [benchmark_model(args, model, wrapper) for model in models]
    print_table(results)


if __name__ == "__main__":
    main()
