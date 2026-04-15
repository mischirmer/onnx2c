#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

./build/aww_fp32_abyzft --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_abyzft.log"

./build/aww_int8_abyzft --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_abyzft.log"

./build/aww_fp32_abft --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_abft.log"

./build/aww_int8_abft --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_abft.log"

./build/aww_fp32_freivalds1x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_freivalds1x.log"

./build/aww_int8_freivalds1x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_freivalds1x.log"

./build/aww_fp32_freivalds2x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_freivalds2x.log"

./build/aww_int8_freivalds2x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_freivalds2x.log"

./build/aww_fp32_freivalds3x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_freivalds3x.log"

./build/aww_int8_freivalds3x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_freivalds3x.log"

./build/aww_fp32_freivalds4x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_freivalds4x.log"

./build/aww_int8_freivalds4x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_freivalds4x.log"

./build/aww_fp32_gvfa1x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_gvfa1x.log"

./build/aww_int8_gvfa1x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_gvfa1x.log"

./build/aww_fp32_gvfa2x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_fp32_gvfa2x.log"

./build/aww_int8_gvfa2x --bin-dir ../../../energyrunner/datasets/kws01 \
  --label-csv ../../../energyrunner/datasets/kws01-open/mfcc/y_labels.csv \
  --limit 50 --sweep > "${LOG_DIR}/output_int8_gvfa2x.log"

python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_abyzft.log" --int8-log "${LOG_DIR}/output_int8_abyzft.log" --out-dir plots_abyzft
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_abft.log" --int8-log "${LOG_DIR}/output_int8_abft.log" --out-dir plots_abft
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds1x.log" --int8-log "${LOG_DIR}/output_int8_freivalds1x.log" --out-dir plots_freivalds1x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds2x.log" --int8-log "${LOG_DIR}/output_int8_freivalds2x.log" --out-dir plots_freivalds2x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds3x.log" --int8-log "${LOG_DIR}/output_int8_freivalds3x.log" --out-dir plots_freivalds3x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds4x.log" --int8-log "${LOG_DIR}/output_int8_freivalds4x.log" --out-dir plots_freivalds4x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_gvfa1x.log" --int8-log "${LOG_DIR}/output_int8_gvfa1x.log" --out-dir plots_gvfa1x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_gvfa2x.log" --int8-log "${LOG_DIR}/output_int8_gvfa2x.log" --out-dir plots_gvfa2x
python3 plot_worst_case_drop.py --log-dir "${LOG_DIR}" --out-dir plots_worst_case
