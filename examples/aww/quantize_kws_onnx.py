import argparse
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

    reader = DummyReader(input_name=input_name, shape=tuple(shape), n=int(args.calib_samples))

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

