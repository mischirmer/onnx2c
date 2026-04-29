import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class DummyReader(CalibrationDataReader):
    def __init__(self, input_name: str, shape: tuple[int, ...], n: int):
        self.input_name = input_name
        self.data = [{input_name: np.random.randn(*shape).astype(np.float32)} for _ in range(n)]
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


class DatasetBinReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        shape: tuple[int, ...],
        sample_files: list[Path],
        input_scale: float | None,
        input_zero_point: int,
    ):
        self.input_name = input_name
        self.shape = shape
        self.sample_files = sample_files
        self.input_scale = input_scale
        self.input_zero_point = input_zero_point
        self.idx = 0

    def _load_one(self, p: Path) -> np.ndarray:
        x = np.fromfile(p, dtype=np.int8)
        expected = int(np.prod(self.shape))
        if x.size != expected:
            raise SystemExit(f"{p}: expected {expected} int8 elements, found {x.size}")

        xf = x.astype(np.float32)
        if self.input_scale is not None:
            xf = (xf - float(self.input_zero_point)) * float(self.input_scale)

        return xf.reshape(self.shape)

    def get_next(self):
        if self.idx >= len(self.sample_files):
            return None

        p = self.sample_files[self.idx]
        self.idx += 1
        return {self.input_name: self._load_one(p)}


def _load_sample_files(bin_dir: Path, label_csv: Path, limit: int) -> list[Path]:
    if not label_csv.exists():
        raise SystemExit(f"Label CSV not found: {label_csv}")
    if not bin_dir.exists():
        raise SystemExit(f"Bin directory not found: {bin_dir}")

    sample_files: list[Path] = []
    for raw in label_csv.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        cols = [c.strip() for c in line.split(",")]
        if not cols:
            continue

        p = bin_dir / cols[0]
        if p.exists():
            sample_files.append(p)
            if len(sample_files) >= limit:
                break

    if not sample_files:
        raise SystemExit(
            f"No calibration samples found from {label_csv} under {bin_dir}."
        )
    return sample_files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Quantize the KWS FP32 ONNX to either QDQ or QOperator (QLinearConv/QLinearMatMul)."
    )
    ap.add_argument(
        "--fp32",
        default="kws_ref_model_full_float32_from_savedmodel.onnx",
        help="Input FP32 ONNX path",
    )
    ap.add_argument(
        "--out",
        default="kws_ref_model_full_float32_from_savedmodel.int8.qop.onnx",
        help="Output quantized ONNX path",
    )
    ap.add_argument(
        "--format",
        choices=["qop", "qdq"],
        default="qop",
        help="qop emits integer operators like QLinearConv; qdq emits Quantize/Dequantize nodes.",
    )
    ap.add_argument("--calib-samples", type=int, default=200)
    ap.add_argument(
        "--bin-dir",
        default="../../../energyrunner/datasets/kws01",
        help="Directory containing KWS .bin samples referenced by --label-csv",
    )
    ap.add_argument(
        "--label-csv",
        default="../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv",
        help="CSV manifest whose first column is the .bin filename",
    )
    ap.add_argument(
        "--use-random-calibration",
        action="store_true",
        help="Use synthetic random tensors instead of dataset bins",
    )
    ap.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help="Optional dequant scale to convert int8 bins to float before calibration",
    )
    ap.add_argument(
        "--input-zero-point",
        type=int,
        default=0,
        help="Optional dequant zero-point paired with --input-scale",
    )
    args = ap.parse_args()

    qformat = QuantFormat.QOperator if args.format == "qop" else QuantFormat.QDQ

    model = onnx.load(args.fp32)
    if len(model.graph.input) != 1:
        raise SystemExit(
            f"Expected exactly 1 model input, found {len(model.graph.input)}: "
            f"{[i.name for i in model.graph.input]}"
        )

    input0 = model.graph.input[0]
    input_name = input0.name
    shape: list[int] = []
    for d in input0.type.tensor_type.shape.dim:
        if d.dim_value <= 0:
            raise SystemExit(f"Model input has dynamic/unknown dim; please hardcode it: {input_name}")
        shape.append(int(d.dim_value))

    if args.use_random_calibration:
        reader = DummyReader(input_name=input_name, shape=tuple(shape), n=int(args.calib_samples))
    else:
        sample_files = _load_sample_files(
            bin_dir=Path(args.bin_dir),
            label_csv=Path(args.label_csv),
            limit=int(args.calib_samples),
        )
        print(
            f"using {len(sample_files)} dataset calibration samples from {args.bin_dir} "
            f"(manifest: {args.label_csv})"
        )
        reader = DatasetBinReader(
            input_name=input_name,
            shape=tuple(shape),
            sample_files=sample_files,
            input_scale=args.input_scale,
            input_zero_point=int(args.input_zero_point),
        )

    quantize_static(
        model_input=args.fp32,
        model_output=args.out,
        calibration_data_reader=reader,
        quant_format=qformat,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

