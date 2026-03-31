CPP ?= g++
CC ?= gcc
PY ?= ../../../venv/bin/python3

CFLAGS ?= -O2 -std=c99
CXXFLAGS ?= -O2 -std=c++17

ONNX2C ?= ../../build/onnx2c
ONNX2C_FLAGS ?= --conv-im2col
TFLITE_INT8 ?= ../../../kws_ref_model.tflite

FP32_ONNX ?= kws_ref_model_full_float32_from_savedmodel.onnx
INT8_ONNX ?= kws_ref_model_full_float32_from_savedmodel.int8.qdq.onnx

FP32_C ?= resnet_fp32.c
INT8_C ?= resnet_i8.c

FP32_O ?= resnet_fp32.o
INT8_O ?= resnet_i8.o

RUNNER_CPP ?= run_generated.cpp

.PHONY: all fp32 int8 gen_fp32 gen_int8 clean help

all: help

help:
	@echo "Targets:"
	@echo "  make fp32       # build run_fp32 (generates $(FP32_C) if missing)"
	@echo "  make int8       # build run_int8 (generates $(INT8_C) if missing)"
	@echo "  make gen_fp32   # regenerate $(FP32_C) from $(FP32_ONNX)"
	@echo "  make gen_int8   # regenerate $(INT8_C) from $(INT8_ONNX)"
	@echo "  make clean"
	@echo ""
	@echo "Run:"
	@echo "  ./run_fp32 --bin-dir DIR --label-csv FILE [--limit N]"
	@echo "  ./run_int8 --bin-dir DIR --label-csv FILE [--limit N]"

fp32: run_fp32
int8: run_int8

gen_fp32: $(FP32_C)
gen_int8: $(INT8_C)

FLAGS_STAMP := .onnx2c_flags.stamp

$(FLAGS_STAMP):
	@echo "$(ONNX2C_FLAGS)" > $(FLAGS_STAMP)

run_fp32: $(FP32_O) $(RUNNER_CPP)
	$(CXX) $(CXXFLAGS) $(RUNNER_CPP) $(FP32_O) -o $@ $(RUN_DEFS)

run_int8: $(INT8_O) $(RUNNER_CPP)
	$(CXX) $(CXXFLAGS) $(RUNNER_CPP) $(INT8_O) -o $@ $(RUN_DEFS)

$(FP32_O): $(FP32_C)
	$(CC) $(CFLAGS) -c $< -o $@

$(INT8_O): $(INT8_C)
	$(CC) $(CFLAGS) -c $< -o $@

# Auto-generate C from ONNX if the file is missing.
$(FP32_C): $(FP32_ONNX) $(ONNX2C) $(FLAGS_STAMP)
	$(ONNX2C) $(ONNX2C_FLAGS) $(FP32_ONNX) > $(FP32_C)

$(INT8_C): $(INT8_ONNX) $(ONNX2C) $(FLAGS_STAMP)
	$(ONNX2C) $(ONNX2C_FLAGS) $(INT8_ONNX) > $(INT8_C)

# Use the int8 TFLite model's input quantization parameters to dequantize the MLPerf bins
# into float input expected by the generated ONNX models.
RUN_DEFS := $(shell $(PY) -c "import os,sys; p='$(TFLITE_INT8)'; \
  (print('') or sys.exit(0)) if not os.path.exists(p) else None; \
  import tensorflow as tf; \
  inter=tf.lite.Interpreter(model_path=p); inter.allocate_tensors(); \
  q=inter.get_input_details()[0]['quantization']; \
  print(f'-DINPUT_SCALE={q[0]} -DINPUT_ZERO_POINT={q[1]}')")

clean:
	rm -f run_fp32 run_int8 $(FP32_O) $(INT8_O) $(FLAGS_STAMP)
