CPP ?= g++
CC ?= gcc
PY ?= python3

CFLAGS ?= -O2 -std=c99
CXXFLAGS ?= -O2 -std=c++17
CPPFLAGS ?=

ONNX2C ?= ../../build/onnx2c
IM2COL_OPT ?= -p im2col_all
ONNX2C_FLAGS ?= $(IM2COL_OPT) --log=1
TFLITE_INT8 ?= ../../tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite
BUILD_DIR ?= build

FP32_ONNX ?= tiny_kws_ref_fp32.onnx
INT8_ONNX ?= tiny_kws_ref_int8_qop.onnx

RUNNER_CPP ?= run_generated.cpp

.PHONY: all baseline baseline_raw abft abyzft freivalds1x freivalds2x freivalds%x gvfa1x gvfa2x gvfa%x fp32 int8 gen_fp32 gen_int8 run_fp32 run_int8 clean help FORCE \
	aww_fp32_baseline aww_int8_baseline aww_fp32_abft aww_int8_abft \
	aww_fp32_abyzft aww_int8_abyzft aww_fp32_freivalds1x aww_int8_freivalds1x \
	aww_fp32_freivalds2x aww_int8_freivalds2x aww_fp32_gvfa1x aww_int8_gvfa1x \
	aww_fp32_gvfa2x aww_int8_gvfa2x \
	run_fp32_baseline run_int8_baseline run_fp32_abft run_int8_abft \
	run_fp32_abyzft run_int8_abyzft run_fp32_freivalds1x run_int8_freivalds1x \
	run_fp32_freivalds2x run_int8_freivalds2x run_fp32_gvfa1x run_int8_gvfa1x \
	run_fp32_gvfa2x run_int8_gvfa2x

BASELINE_ONNX2C_FLAGS ?= $(ONNX2C_FLAGS)
BASELINE_RAW_ONNX2C_FLAGS ?= --log=1
ABFT_ONNX2C_FLAGS ?= $(IM2COL_OPT) --abft-gemm --log=1
ABYZFT_ONNX2C_FLAGS ?= $(IM2COL_OPT) --abyzft-gemm --log=1
FREIVALDS1X_ONNX2C_FLAGS ?= $(IM2COL_OPT) --freivalds-gemm --freivalds-checks 1 --log=1
FREIVALDS2X_ONNX2C_FLAGS ?= $(IM2COL_OPT) --freivalds-gemm --freivalds-checks 2 --log=1
GVFA1X_ONNX2C_FLAGS ?= $(IM2COL_OPT) --gvfa-gemm --gvfa-checks 1 --log=1
GVFA2X_ONNX2C_FLAGS ?= $(IM2COL_OPT) --gvfa-gemm --gvfa-checks 2 --log=1

BASELINE_FP32_C ?= $(BUILD_DIR)/aww_fp32_baseline.c
BASELINE_INT8_C ?= $(BUILD_DIR)/aww_i8_baseline.c
BASELINE_RAW_FP32_C ?= $(BUILD_DIR)/aww_fp32_baseline_raw.c
BASELINE_RAW_INT8_C ?= $(BUILD_DIR)/aww_i8_baseline_raw.c
ABFT_FP32_C ?= $(BUILD_DIR)/aww_fp32_abft.c
ABFT_INT8_C ?= $(BUILD_DIR)/aww_i8_abft.c
ABYZFT_FP32_C ?= $(BUILD_DIR)/aww_fp32_abyzft.c
ABYZFT_INT8_C ?= $(BUILD_DIR)/aww_i8_abyzft.c
FREIVALDS1X_FP32_C ?= $(BUILD_DIR)/aww_fp32_freivalds1x.c
FREIVALDS1X_INT8_C ?= $(BUILD_DIR)/aww_i8_freivalds1x.c
FREIVALDS2X_FP32_C ?= $(BUILD_DIR)/aww_fp32_freivalds2x.c
FREIVALDS2X_INT8_C ?= $(BUILD_DIR)/aww_i8_freivalds2x.c
GVFA1X_FP32_C ?= $(BUILD_DIR)/aww_fp32_gvfa1x.c
GVFA1X_INT8_C ?= $(BUILD_DIR)/aww_i8_gvfa1x.c
GVFA2X_FP32_C ?= $(BUILD_DIR)/aww_fp32_gvfa2x.c
GVFA2X_INT8_C ?= $(BUILD_DIR)/aww_i8_gvfa2x.c

BASELINE_FP32_O ?= $(BUILD_DIR)/aww_fp32_baseline.o
BASELINE_INT8_O ?= $(BUILD_DIR)/aww_i8_baseline.o
BASELINE_RAW_FP32_O ?= $(BUILD_DIR)/aww_fp32_baseline_raw.o
BASELINE_RAW_INT8_O ?= $(BUILD_DIR)/aww_i8_baseline_raw.o
ABFT_FP32_O ?= $(BUILD_DIR)/aww_fp32_abft.o
ABFT_INT8_O ?= $(BUILD_DIR)/aww_i8_abft.o
ABYZFT_FP32_O ?= $(BUILD_DIR)/aww_fp32_abyzft.o
ABYZFT_INT8_O ?= $(BUILD_DIR)/aww_i8_abyzft.o
FREIVALDS1X_FP32_O ?= $(BUILD_DIR)/aww_fp32_freivalds1x.o
FREIVALDS1X_INT8_O ?= $(BUILD_DIR)/aww_i8_freivalds1x.o
FREIVALDS2X_FP32_O ?= $(BUILD_DIR)/aww_fp32_freivalds2x.o
FREIVALDS2X_INT8_O ?= $(BUILD_DIR)/aww_i8_freivalds2x.o
GVFA1X_FP32_O ?= $(BUILD_DIR)/aww_fp32_gvfa1x.o
GVFA1X_INT8_O ?= $(BUILD_DIR)/aww_i8_gvfa1x.o
GVFA2X_FP32_O ?= $(BUILD_DIR)/aww_fp32_gvfa2x.o
GVFA2X_INT8_O ?= $(BUILD_DIR)/aww_i8_gvfa2x.o

BASELINE_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_baseline
BASELINE_INT8_BIN ?= $(BUILD_DIR)/aww_int8_baseline
BASELINE_RAW_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_baseline_raw
BASELINE_RAW_INT8_BIN ?= $(BUILD_DIR)/aww_int8_baseline_raw
ABFT_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_abft
ABFT_INT8_BIN ?= $(BUILD_DIR)/aww_int8_abft
ABYZFT_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_abyzft
ABYZFT_INT8_BIN ?= $(BUILD_DIR)/aww_int8_abyzft
FREIVALDS1X_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_freivalds1x
FREIVALDS1X_INT8_BIN ?= $(BUILD_DIR)/aww_int8_freivalds1x
FREIVALDS2X_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_freivalds2x
FREIVALDS2X_INT8_BIN ?= $(BUILD_DIR)/aww_int8_freivalds2x
GVFA1X_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_gvfa1x
GVFA1X_INT8_BIN ?= $(BUILD_DIR)/aww_int8_gvfa1x
GVFA2X_FP32_BIN ?= $(BUILD_DIR)/aww_fp32_gvfa2x
GVFA2X_INT8_BIN ?= $(BUILD_DIR)/aww_int8_gvfa2x

BASELINE_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_baseline.stamp
BASELINE_RAW_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_baseline_raw.stamp
ABFT_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_abft.stamp
ABYZFT_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_abyzft.stamp
FREIVALDS1X_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_freivalds1x.stamp
FREIVALDS2X_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_freivalds2x.stamp
GVFA1X_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_gvfa1x.stamp
GVFA2X_FLAGS_STAMP := $(BUILD_DIR)/.onnx2c_flags_gvfa2x.stamp

ALL_BINS := \
	$(BASELINE_FP32_BIN) $(BASELINE_INT8_BIN) \
	$(BASELINE_RAW_FP32_BIN) $(BASELINE_RAW_INT8_BIN) \
	$(ABFT_FP32_BIN) $(ABFT_INT8_BIN) \
	$(ABYZFT_FP32_BIN) $(ABYZFT_INT8_BIN) \
	$(FREIVALDS1X_FP32_BIN) $(FREIVALDS1X_INT8_BIN) \
	$(FREIVALDS2X_FP32_BIN) $(FREIVALDS2X_INT8_BIN) \
	$(GVFA1X_FP32_BIN) $(GVFA1X_INT8_BIN) \
	$(GVFA2X_FP32_BIN) $(GVFA2X_INT8_BIN)

all: $(ALL_BINS)

baseline: $(BASELINE_FP32_BIN) $(BASELINE_INT8_BIN)
baseline_raw: $(BASELINE_RAW_FP32_BIN) $(BASELINE_RAW_INT8_BIN)
abft: $(ABFT_FP32_BIN) $(ABFT_INT8_BIN)
abyzft: $(ABYZFT_FP32_BIN) $(ABYZFT_INT8_BIN)
freivalds1x: $(FREIVALDS1X_FP32_BIN) $(FREIVALDS1X_INT8_BIN)
freivalds2x: $(FREIVALDS2X_FP32_BIN) $(FREIVALDS2X_INT8_BIN)
gvfa1x: $(GVFA1X_FP32_BIN) $(GVFA1X_INT8_BIN)
gvfa2x: $(GVFA2X_FP32_BIN) $(GVFA2X_INT8_BIN)

# Dynamic Freivalds target:
#   make freivalds3x
# builds:
#   build/run_fp32_freivalds3x and build/run_int8_freivalds3x
freivalds%x:
	@case "$*" in \
	  ''|*[!0-9]*) echo "error: expected numeric N in target freivaldsNx (e.g. make freivalds3x)"; exit 1 ;; \
	  0) echo "error: Freivalds checks must be >= 1"; exit 1 ;; \
	esac
	$(MAKE) \
	  FREIVALDS1X_ONNX2C_FLAGS="$(IM2COL_OPT) --freivalds-gemm --freivalds-checks $* --log=1" \
	  FREIVALDS1X_FP32_C="$(BUILD_DIR)/aww_fp32_freivalds$*x.c" \
	  FREIVALDS1X_INT8_C="$(BUILD_DIR)/aww_i8_freivalds$*x.c" \
	  FREIVALDS1X_FP32_O="$(BUILD_DIR)/aww_fp32_freivalds$*x.o" \
	  FREIVALDS1X_INT8_O="$(BUILD_DIR)/aww_i8_freivalds$*x.o" \
	  FREIVALDS1X_FP32_BIN="$(BUILD_DIR)/aww_fp32_freivalds$*x" \
	  FREIVALDS1X_INT8_BIN="$(BUILD_DIR)/aww_int8_freivalds$*x" \
	  FREIVALDS1X_FLAGS_STAMP="$(BUILD_DIR)/.onnx2c_flags_freivalds$*x.stamp" \
	  freivalds1x

gvfa%x:
	@case "$*" in \
	  ''|*[!0-9]*) echo "error: expected numeric N in target gvfaNx (e.g. make gvfa3x)"; exit 1 ;; \
	  0) echo "error: GVFA checks must be >= 1"; exit 1 ;; \
	esac
	$(MAKE) \
	  GVFA1X_ONNX2C_FLAGS="$(IM2COL_OPT) --gvfa-gemm --gvfa-checks $* --log=1" \
	  GVFA1X_FP32_C="$(BUILD_DIR)/aww_fp32_gvfa$*x.c" \
	  GVFA1X_INT8_C="$(BUILD_DIR)/aww_i8_gvfa$*x.c" \
	  GVFA1X_FP32_O="$(BUILD_DIR)/aww_fp32_gvfa$*x.o" \
	  GVFA1X_INT8_O="$(BUILD_DIR)/aww_i8_gvfa$*x.o" \
	  GVFA1X_FP32_BIN="$(BUILD_DIR)/aww_fp32_gvfa$*x" \
	  GVFA1X_INT8_BIN="$(BUILD_DIR)/aww_int8_gvfa$*x" \
	  GVFA1X_FLAGS_STAMP="$(BUILD_DIR)/.onnx2c_flags_gvfa$*x.stamp" \
	  gvfa1x

help:
	@echo "Targets:"
	@echo "  make all        # build baseline, abft, abyzft, freivalds1x/2x, and gvfa1x/2x binaries"
	@echo "  make baseline   # build baseline fp32/int8 binaries"
	@echo "  make abft       # build ABFT fp32/int8 binaries"
	@echo "  make abyzft     # build AByzFT fp32/int8 binaries"
	@echo "  make freivalds1x # build Freivalds(1x) fp32/int8 binaries"
	@echo "  make freivalds2x # build Freivalds(2x) fp32/int8 binaries"
	@echo "  make freivaldsNx # build Freivalds(Nx) fp32/int8 binaries (e.g. freivalds3x)"
	@echo "  make gvfa1x      # build GVFA(1x) fp32/int8 binaries"
	@echo "  make gvfa2x      # build GVFA(2x) fp32/int8 binaries"
	@echo "  make gvfaNx      # build GVFA(Nx) fp32/int8 binaries (e.g. gvfa3x)"
	@echo "  make fp32       # build baseline fp32 binary only"
	@echo "  make int8       # build baseline int8 binary only"
	@echo "  make clean"
	@echo ""
	@echo "Run:"
	@echo "  ./$(BASELINE_FP32_BIN) --bin-dir DIR --label-csv FILE [--limit N]"
	@echo "  ./$(BASELINE_INT8_BIN) --bin-dir DIR --label-csv FILE [--limit N]"

fp32: $(BASELINE_FP32_BIN)
int8: $(BASELINE_INT8_BIN)
gen_fp32: $(BASELINE_FP32_C)
gen_int8: $(BASELINE_INT8_C)

$(BUILD_DIR):
	mkdir -p $@

FORCE:

$(BASELINE_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(BASELINE_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(ABFT_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(ABFT_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(ABYZFT_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(ABYZFT_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(FREIVALDS1X_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(FREIVALDS1X_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(FREIVALDS2X_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(FREIVALDS2X_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(GVFA1X_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(GVFA1X_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(GVFA2X_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(GVFA2X_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(BASELINE_FP32_C): $(FP32_ONNX) $(ONNX2C) $(BASELINE_FLAGS_STAMP)
	$(ONNX2C) $(BASELINE_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(BASELINE_INT8_C): $(INT8_ONNX) $(ONNX2C) $(BASELINE_FLAGS_STAMP)
	$(ONNX2C) $(BASELINE_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(BASELINE_RAW_FP32_C): $(FP32_ONNX) $(ONNX2C) $(BASELINE_RAW_FLAGS_STAMP)
	$(ONNX2C) $(BASELINE_RAW_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(BASELINE_RAW_INT8_C): $(INT8_ONNX) $(ONNX2C) $(BASELINE_RAW_FLAGS_STAMP)
	$(ONNX2C) $(BASELINE_RAW_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(BASELINE_RAW_FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; echo "$(BASELINE_RAW_ONNX2C_FLAGS)" > "$$tmp"; cmp -s "$$tmp" "$@" || mv "$$tmp" "$@"; rm -f "$$tmp"

$(ABFT_FP32_C): $(FP32_ONNX) $(ONNX2C) $(ABFT_FLAGS_STAMP)
	$(ONNX2C) $(ABFT_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(ABFT_INT8_C): $(INT8_ONNX) $(ONNX2C) $(ABFT_FLAGS_STAMP)
	$(ONNX2C) $(ABFT_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(ABYZFT_FP32_C): $(FP32_ONNX) $(ONNX2C) $(ABYZFT_FLAGS_STAMP)
	$(ONNX2C) $(ABYZFT_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(ABYZFT_INT8_C): $(INT8_ONNX) $(ONNX2C) $(ABYZFT_FLAGS_STAMP)
	$(ONNX2C) $(ABYZFT_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(FREIVALDS1X_FP32_C): $(FP32_ONNX) $(ONNX2C) $(FREIVALDS1X_FLAGS_STAMP)
	$(ONNX2C) $(FREIVALDS1X_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(FREIVALDS1X_INT8_C): $(INT8_ONNX) $(ONNX2C) $(FREIVALDS1X_FLAGS_STAMP)
	$(ONNX2C) $(FREIVALDS1X_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(FREIVALDS2X_FP32_C): $(FP32_ONNX) $(ONNX2C) $(FREIVALDS2X_FLAGS_STAMP)
	$(ONNX2C) $(FREIVALDS2X_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(FREIVALDS2X_INT8_C): $(INT8_ONNX) $(ONNX2C) $(FREIVALDS2X_FLAGS_STAMP)
	$(ONNX2C) $(FREIVALDS2X_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(GVFA1X_FP32_C): $(FP32_ONNX) $(ONNX2C) $(GVFA1X_FLAGS_STAMP)
	$(ONNX2C) $(GVFA1X_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(GVFA1X_INT8_C): $(INT8_ONNX) $(ONNX2C) $(GVFA1X_FLAGS_STAMP)
	$(ONNX2C) $(GVFA1X_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(GVFA2X_FP32_C): $(FP32_ONNX) $(ONNX2C) $(GVFA2X_FLAGS_STAMP)
	$(ONNX2C) $(GVFA2X_ONNX2C_FLAGS) $(FP32_ONNX) > $@

$(GVFA2X_INT8_C): $(INT8_ONNX) $(ONNX2C) $(GVFA2X_FLAGS_STAMP)
	$(ONNX2C) $(GVFA2X_ONNX2C_FLAGS) $(INT8_ONNX) > $@

$(BASELINE_FP32_O): $(BASELINE_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BASELINE_INT8_O): $(BASELINE_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(ABFT_FP32_O): $(ABFT_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(ABFT_INT8_O): $(ABFT_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(ABYZFT_FP32_O): $(ABYZFT_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(ABYZFT_INT8_O): $(ABYZFT_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(FREIVALDS1X_FP32_O): $(FREIVALDS1X_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(FREIVALDS1X_INT8_O): $(FREIVALDS1X_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(FREIVALDS2X_FP32_O): $(FREIVALDS2X_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(FREIVALDS2X_INT8_O): $(FREIVALDS2X_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GVFA1X_FP32_O): $(GVFA1X_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GVFA1X_INT8_O): $(GVFA1X_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GVFA2X_FP32_O): $(GVFA2X_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(GVFA2X_INT8_O): $(GVFA2X_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BASELINE_FP32_BIN): $(BASELINE_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(BASELINE_FP32_O) -o $@ $(RUN_DEFS)

$(BASELINE_INT8_BIN): $(BASELINE_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(BASELINE_INT8_O) -o $@ $(RUN_DEFS)

$(BASELINE_RAW_FP32_O): $(BASELINE_RAW_FP32_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BASELINE_RAW_INT8_O): $(BASELINE_RAW_INT8_C)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BASELINE_RAW_FP32_BIN): $(BASELINE_RAW_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(BASELINE_RAW_FP32_O) -o $@ $(RUN_DEFS)

$(BASELINE_RAW_INT8_BIN): $(BASELINE_RAW_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(BASELINE_RAW_INT8_O) -o $@ $(RUN_DEFS)

$(ABFT_FP32_BIN): $(ABFT_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(ABFT_FP32_O) -o $@ $(RUN_DEFS)

$(ABFT_INT8_BIN): $(ABFT_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(ABFT_INT8_O) -o $@ $(RUN_DEFS)

$(ABYZFT_FP32_BIN): $(ABYZFT_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(ABYZFT_FP32_O) -o $@ $(RUN_DEFS)

$(ABYZFT_INT8_BIN): $(ABYZFT_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(ABYZFT_INT8_O) -o $@ $(RUN_DEFS)

$(FREIVALDS1X_FP32_BIN): $(FREIVALDS1X_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(FREIVALDS1X_FP32_O) -o $@ $(RUN_DEFS)

$(FREIVALDS1X_INT8_BIN): $(FREIVALDS1X_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(FREIVALDS1X_INT8_O) -o $@ $(RUN_DEFS)

$(FREIVALDS2X_FP32_BIN): $(FREIVALDS2X_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(FREIVALDS2X_FP32_O) -o $@ $(RUN_DEFS)

$(FREIVALDS2X_INT8_BIN): $(FREIVALDS2X_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(FREIVALDS2X_INT8_O) -o $@ $(RUN_DEFS)

$(GVFA1X_FP32_BIN): $(GVFA1X_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(GVFA1X_FP32_O) -o $@ $(RUN_DEFS)

$(GVFA1X_INT8_BIN): $(GVFA1X_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(GVFA1X_INT8_O) -o $@ $(RUN_DEFS)

$(GVFA2X_FP32_BIN): $(GVFA2X_FP32_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(GVFA2X_FP32_O) -o $@ $(RUN_DEFS)

$(GVFA2X_INT8_BIN): $(GVFA2X_INT8_O) $(RUNNER_CPP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNNER_CPP) $(GVFA2X_INT8_O) -o $@ $(RUN_DEFS)

# AWW-named executable targets.
aww_fp32_baseline: $(BASELINE_FP32_BIN)
aww_int8_baseline: $(BASELINE_INT8_BIN)
aww_fp32_abft: $(ABFT_FP32_BIN)
aww_int8_abft: $(ABFT_INT8_BIN)
aww_fp32_abyzft: $(ABYZFT_FP32_BIN)
aww_int8_abyzft: $(ABYZFT_INT8_BIN)
aww_fp32_freivalds1x: $(FREIVALDS1X_FP32_BIN)
aww_int8_freivalds1x: $(FREIVALDS1X_INT8_BIN)
aww_fp32_freivalds2x: $(FREIVALDS2X_FP32_BIN)
aww_int8_freivalds2x: $(FREIVALDS2X_INT8_BIN)
aww_fp32_gvfa1x: $(GVFA1X_FP32_BIN)
aww_int8_gvfa1x: $(GVFA1X_INT8_BIN)
aww_fp32_gvfa2x: $(GVFA2X_FP32_BIN)
aww_int8_gvfa2x: $(GVFA2X_INT8_BIN)

# Legacy aliases kept for compatibility with older scripts.
run_fp32: $(BASELINE_FP32_BIN)
run_int8: $(BASELINE_INT8_BIN)

# Compatibility aliases (old target names still valid).
run_fp32_baseline: $(BASELINE_FP32_BIN)
run_int8_baseline: $(BASELINE_INT8_BIN)
run_fp32_abft: $(ABFT_FP32_BIN)
run_int8_abft: $(ABFT_INT8_BIN)
run_fp32_abyzft: $(ABYZFT_FP32_BIN)
run_int8_abyzft: $(ABYZFT_INT8_BIN)
run_fp32_freivalds1x: $(FREIVALDS1X_FP32_BIN)
run_int8_freivalds1x: $(FREIVALDS1X_INT8_BIN)
run_fp32_freivalds2x: $(FREIVALDS2X_FP32_BIN)
run_int8_freivalds2x: $(FREIVALDS2X_INT8_BIN)
run_fp32_gvfa1x: $(GVFA1X_FP32_BIN)
run_int8_gvfa1x: $(GVFA1X_INT8_BIN)
run_fp32_gvfa2x: $(GVFA2X_FP32_BIN)
run_int8_gvfa2x: $(GVFA2X_INT8_BIN)

# Use the int8 TFLite model's input quantization parameters to dequantize the MLPerf bins
# into float input expected by the generated ONNX models.
RUN_DEFS := $(shell $(PY) -c "import os,sys; p='$(TFLITE_INT8)'; \
  (sys.stderr.write(f'Missing TFLite model for input quantization: {p}\\n') or sys.exit(2)) if not os.path.exists(p) else None; \
  import tensorflow as tf; \
  inter=tf.lite.Interpreter(model_path=p); inter.allocate_tensors(); \
  q=inter.get_input_details()[0]['quantization']; \
  print(f'-DINPUT_SCALE={q[0]} -DINPUT_ZERO_POINT={q[1]}')")

clean:
	rm -f $(ALL_BINS) \
	  $(BASELINE_FP32_O) $(BASELINE_INT8_O) $(ABFT_FP32_O) $(ABFT_INT8_O) \
	  $(ABYZFT_FP32_O) $(ABYZFT_INT8_O) $(FREIVALDS1X_FP32_O) $(FREIVALDS1X_INT8_O) \
	  $(FREIVALDS2X_FP32_O) $(FREIVALDS2X_INT8_O) $(GVFA1X_FP32_O) $(GVFA1X_INT8_O) \
	  $(GVFA2X_FP32_O) $(GVFA2X_INT8_O) \
	  $(BASELINE_FP32_C) $(BASELINE_INT8_C) $(ABFT_FP32_C) $(ABFT_INT8_C) \
	  $(ABYZFT_FP32_C) $(ABYZFT_INT8_C) $(FREIVALDS1X_FP32_C) $(FREIVALDS1X_INT8_C) \
	  $(FREIVALDS2X_FP32_C) $(FREIVALDS2X_INT8_C) $(GVFA1X_FP32_C) $(GVFA1X_INT8_C) \
	  $(GVFA2X_FP32_C) $(GVFA2X_INT8_C) \
	  $(BASELINE_FLAGS_STAMP) $(ABFT_FLAGS_STAMP) $(ABYZFT_FLAGS_STAMP) $(FREIVALDS1X_FLAGS_STAMP) $(FREIVALDS2X_FLAGS_STAMP) $(GVFA1X_FLAGS_STAMP) $(GVFA2X_FLAGS_STAMP) \
	  run_fp32_* run_int8_* \
	  aww_fp32_* aww_int8_* \
	  aww_fp32_*.c aww_i8_*.c aww_fp32_*.o aww_i8_*.o \
	  $(BUILD_DIR)/aww_fp32_* $(BUILD_DIR)/aww_int8_* \
	  $(BUILD_DIR)/aww_i8_*.c $(BUILD_DIR)/aww_fp32_*.c \
	  $(BUILD_DIR)/aww_i8_*.o $(BUILD_DIR)/aww_fp32_*.o \
	  $(BUILD_DIR)/.onnx2c_flags_*.stamp \
	  .onnx2c_flags_baseline.stamp .onnx2c_flags_abft.stamp .onnx2c_flags_abyzft.stamp .onnx2c_flags_freivalds1x.stamp .onnx2c_flags_freivalds2x.stamp .onnx2c_flags_gvfa1x.stamp .onnx2c_flags_gvfa2x.stamp
