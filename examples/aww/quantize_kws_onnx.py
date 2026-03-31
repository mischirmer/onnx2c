import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

class DummyReader(CalibrationDataReader):
    def __init__(self, input_name, shape, n=100):
        self.input_name = input_name
        self.data = [
            {input_name: np.random.randn(*shape).astype(np.float32)} for _ in range(n)
        ]
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)

fp32 = "kws_ref_model_full_float32_from_savedmodel.onnx"
int8 = "kws_ref_model_full_float32_from_savedmodel.int8.qdq.onnx"

model = onnx.load(fp32)
if len(model.graph.input) != 1:
    raise SystemExit(
        f"Expected exactly 1 model input, found {len(model.graph.input)}: "
        f"{[i.name for i in model.graph.input]}"
    )

input0 = model.graph.input[0]
input_name = input0.name
shape = []
for d in input0.type.tensor_type.shape.dim:
    if d.dim_value <= 0:
        raise SystemExit(
            f"Model input has dynamic/unknown dim; please hardcode it: {input_name}"
        )
    shape.append(int(d.dim_value))

reader = DummyReader(input_name=input_name, shape=tuple(shape), n=200)

quantize_static(
    model_input=fp32,
    model_output=int8,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)
print("wrote", int8)
