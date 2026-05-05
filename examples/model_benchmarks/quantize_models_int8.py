#!/usr/bin/env python3
"""
Quantize ONNX models in this directory to INT8.

By default, the script scans the directory containing this file for `.onnx`
models, skips files that already end in `_quant.onnx`, and writes quantized
models next to the originals using the suffix `_quant.onnx`.

Examples:
  python3 quantize_models_int8.py
  python3 quantize_models_int8.py resnet18_Opset17.onnx
  python3 quantize_models_int8.py --mode static --calib-samples 8
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


DEFAULT_MODELS_DIR = Path(__file__).resolve().parent


class RandomCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, shape: tuple[int, ...], samples: int):
        self._samples = [
            {input_name: np.random.standard_normal(size=shape).astype(np.float32)}
            for _ in range(samples)
        ]
        self._it = iter(self._samples)

    def get_next(self):
        return next(self._it, None)


def _direct_onnx_models(models_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in models_dir.glob("*.onnx")
        if path.is_file() and not path.name.endswith("_quant.onnx")
    )


def _resolve_models(paths: Iterable[str], models_dir: Path) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (models_dir / candidate).resolve()
        if candidate.is_dir():
            resolved.extend(_direct_onnx_models(candidate))
            continue
        if candidate.suffix != ".onnx":
            raise SystemExit(f"Expected an .onnx file or directory, got: {raw}")
        if not candidate.exists():
            raise SystemExit(f"Model not found: {candidate}")
        resolved.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path.name.endswith("_quant.onnx"):
            continue
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _default_output_path(model_path: Path, suffix: str) -> Path:
    return model_path.with_name(f"{model_path.stem}{suffix}{model_path.suffix}")


def _single_input_shape(model_path: Path) -> tuple[str, tuple[int, ...]]:
    model = onnx.load(str(model_path))
    if len(model.graph.input) != 1:
        raise RuntimeError(
            f"expected exactly 1 model input, found {len(model.graph.input)} "
            f"({[x.name for x in model.graph.input]})"
        )

    input0 = model.graph.input[0]
    shape = []
    for dim in input0.type.tensor_type.shape.dim:
        if dim.dim_value <= 0:
            raise RuntimeError(
                f"input {input0.name} has dynamic/unknown dimensions; "
                f"static random calibration needs fixed sizes"
            )
        shape.append(int(dim.dim_value))
    return input0.name, tuple(shape)


def _quantize_one(model_path: Path, out_path: Path, mode: str, calib_samples: int) -> None:
    if mode == "dynamic":
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        return

    input_name, shape = _single_input_shape(model_path)
    reader = RandomCalibrationReader(input_name=input_name, shape=shape, samples=calib_samples)
    quantize_static(
        model_input=str(model_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Quantize one or more ONNX models to INT8."
    )
    ap.add_argument(
        "models",
        nargs="*",
        help="Optional ONNX files or directories. Defaults to all direct .onnx files in this directory.",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Base directory used for default discovery and relative model paths.",
    )
    ap.add_argument(
        "--mode",
        choices=("dynamic", "static"),
        default="dynamic",
        help="Dynamic is the most robust default. Static uses random calibration and emits QOperator INT8 models.",
    )
    ap.add_argument(
        "--calib-samples",
        type=int,
        default=8,
        help="Number of random calibration samples for --mode static.",
    )
    ap.add_argument(
        "--suffix",
        default="_quant",
        help="Suffix inserted before .onnx for the quantized output.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    args = ap.parse_args()

    models_dir = args.models_dir.resolve()
    models = (
        _resolve_models(args.models, models_dir)
        if args.models
        else _direct_onnx_models(models_dir)
    )
    if not models:
        raise SystemExit(f"No input models found in {models_dir}")

    failures = 0
    for model_path in models:
        out_path = _default_output_path(model_path, args.suffix)
        if out_path.exists() and not args.overwrite:
            print(f"skip {model_path.name}: output exists at {out_path.name}")
            continue

        try:
            _quantize_one(
                model_path=model_path,
                out_path=out_path,
                mode=args.mode,
                calib_samples=args.calib_samples,
            )
            print(f"wrote {out_path}")
        except Exception as exc:
            failures += 1
            print(f"failed {model_path}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
