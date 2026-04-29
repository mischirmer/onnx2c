#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"
AUTO_BUILD="${AUTO_BUILD:-1}"
mkdir -p "${LOG_DIR}"

# The Tiny evaluation tree in this repo carries the official manifest, while the
# local .bin payloads are stored under energyrunner/datasets/kws01.
BIN_DIR="${BIN_DIR:-../../energyrunner/datasets/kws01}"
LABEL_CSV="${LABEL_CSV:-../../tiny/benchmark/evaluation/datasets/kws01-open/mfcc/y_labels.csv}"
LIMIT="${LIMIT:-150}"

if [[ "${AUTO_BUILD}" == "1" ]]; then
  make all
fi

jobs=(
  "./build/aww_fp32_abyzft|${LOG_DIR}/output_fp32_abyzft.log|fp32_abyzft"
  "./build/aww_int8_abyzft|${LOG_DIR}/output_int8_abyzft.log|int8_abyzft"
  "./build/aww_fp32_abft|${LOG_DIR}/output_fp32_abft.log|fp32_abft"
  "./build/aww_int8_abft|${LOG_DIR}/output_int8_abft.log|int8_abft"
  "./build/aww_fp32_freivalds1x|${LOG_DIR}/output_fp32_freivalds1x.log|fp32_freivalds1x"
  "./build/aww_int8_freivalds1x|${LOG_DIR}/output_int8_freivalds1x.log|int8_freivalds1x"
  "./build/aww_fp32_freivalds2x|${LOG_DIR}/output_fp32_freivalds2x.log|fp32_freivalds2x"
  "./build/aww_int8_freivalds2x|${LOG_DIR}/output_int8_freivalds2x.log|int8_freivalds2x"
  "./build/aww_fp32_freivalds3x|${LOG_DIR}/output_fp32_freivalds3x.log|fp32_freivalds3x"
  "./build/aww_int8_freivalds3x|${LOG_DIR}/output_int8_freivalds3x.log|int8_freivalds3x"
  "./build/aww_fp32_freivalds4x|${LOG_DIR}/output_fp32_freivalds4x.log|fp32_freivalds4x"
  "./build/aww_int8_freivalds4x|${LOG_DIR}/output_int8_freivalds4x.log|int8_freivalds4x"
  "./build/aww_fp32_gvfa1x|${LOG_DIR}/output_fp32_gvfa1x.log|fp32_gvfa1x"
  "./build/aww_int8_gvfa1x|${LOG_DIR}/output_int8_gvfa1x.log|int8_gvfa1x"
  "./build/aww_fp32_gvfa2x|${LOG_DIR}/output_fp32_gvfa2x.log|fp32_gvfa2x"
  "./build/aww_int8_gvfa2x|${LOG_DIR}/output_int8_gvfa2x.log|int8_gvfa2x"
)

total_jobs="${#jobs[@]}"
declare -a pids
declare -a launched
declare -a finished
completed=0
next_launch=0

run_sweep() {
  local exe="$1"
  local out="$2"
  "${exe}" --bin-dir "${BIN_DIR}" --label-csv "${LABEL_CSV}" --limit "${LIMIT}" --sweep > "${out}" 2>&1
}

job_status_from_log() {
  local log="$1"
  local last
  if [[ ! -s "${log}" ]]; then
    echo "starting..."
    return
  fi
  last="$(tr '\r' '\n' < "${log}" | grep -a -E "progress[[:space:]]+[0-9]+%" | tail -n 1 || true)"
  if [[ -n "${last}" ]]; then
    echo "${last}"
  else
    echo "running..."
  fi
}

render_dashboard() {
  local running=0
  local i exe log label
  for i in "${!pids[@]}"; do
    if [[ -n "${pids[$i]:-}" ]] && kill -0 "${pids[$i]}" 2>/dev/null; then
      running=$((running + 1))
    fi
  done

  printf '\033[H\033[J'
  echo "Sweep dashboard"
  echo "overall: ${completed}/${total_jobs} finished | ${running} running | ${next_launch}/${total_jobs} launched"
  echo
  printf '%-22s | %-10s | %s\n' "job" "state" "live progress"
  printf '%-22s-+-%-10s-+-%s\n' "----------------------" "----------" "------------------------------"

  for i in "${!jobs[@]}"; do
    IFS='|' read -r exe log label <<< "${jobs[$i]}"
    if [[ "${finished[$i]:-0}" == "1" ]]; then
      printf '%-22s | %-10s | %s\n' "${label}" "done" "progress 100%"
    elif [[ "${launched[$i]:-0}" == "1" ]]; then
      printf '%-22s | %-10s | %s\n' "${label}" "running" "$(job_status_from_log "${log}")"
    else
      printf '%-22s | %-10s | %s\n' "${label}" "queued" "-"
    fi
  done
}

if [[ "${RUN_PARALLEL}" == "1" ]]; then
  while (( completed < total_jobs )); do
    while (( next_launch < total_jobs )); do
      running_now=0
      for i in "${!pids[@]}"; do
        if [[ -n "${pids[$i]:-}" ]] && kill -0 "${pids[$i]}" 2>/dev/null; then
          running_now=$((running_now + 1))
        fi
      done
      (( running_now < MAX_PARALLEL )) || break

      IFS='|' read -r exe log _ <<< "${jobs[$next_launch]}"
      launched[$next_launch]=1
      run_sweep "${exe}" "${log}" &
      pids[$next_launch]=$!
      next_launch=$((next_launch + 1))
    done

    for i in "${!pids[@]}"; do
      if [[ "${finished[$i]:-0}" == "0" ]] && [[ -n "${pids[$i]:-}" ]] && ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}" || true
        finished[$i]=1
        completed=$((completed + 1))
      fi
    done

    render_dashboard
    sleep 1
  done
else
  for i in "${!jobs[@]}"; do
    IFS='|' read -r exe log _ <<< "${jobs[$i]}"
    launched[$i]=1
    render_dashboard
    run_sweep "${exe}" "${log}"
    finished[$i]=1
    completed=$((completed + 1))
    next_launch=$((next_launch + 1))
    render_dashboard
  done
fi

python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_abyzft.log" --int8-log "${LOG_DIR}/output_int8_abyzft.log" --out-dir plots_abyzft
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_abft.log" --int8-log "${LOG_DIR}/output_int8_abft.log" --out-dir plots_abft
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds1x.log" --int8-log "${LOG_DIR}/output_int8_freivalds1x.log" --out-dir plots_freivalds1x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds2x.log" --int8-log "${LOG_DIR}/output_int8_freivalds2x.log" --out-dir plots_freivalds2x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds3x.log" --int8-log "${LOG_DIR}/output_int8_freivalds3x.log" --out-dir plots_freivalds3x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_freivalds4x.log" --int8-log "${LOG_DIR}/output_int8_freivalds4x.log" --out-dir plots_freivalds4x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_gvfa1x.log" --int8-log "${LOG_DIR}/output_int8_gvfa1x.log" --out-dir plots_gvfa1x
python3 plot_sweep_results.py --fp32-log "${LOG_DIR}/output_fp32_gvfa2x.log" --int8-log "${LOG_DIR}/output_int8_gvfa2x.log" --out-dir plots_gvfa2x
python3 plot_worst_case_drop.py --log-dir "${LOG_DIR}" --out-dir plots_worst_case
