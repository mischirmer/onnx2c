#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_STRATEGIES = {
    "default_small": {"s8": [-4, -2, 2, 4], "u8": [2, 4, 8]},
    "pos_pow2": {"s8": [2, 4, 8], "u8": [2, 4, 8]},
    "medium": {"s8": [-8, -4, 4, 8], "u8": [4, 8, 16]},
    "wide": {"s8": [-16, -8, 8, 16], "u8": [8, 16, 32]},
    "very_wide": {"s8": [-32, -16, -8, 8, 16, 32], "u8": [8, 16, 32, 64]},
}


def parse_scale_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_strategy_arg(text: str) -> tuple[str, dict[str, list[int]]]:
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"invalid strategy '{text}'; expected name;s8=a,b,c;u8=a,b,c"
        )
    name = parts[0]
    spec: dict[str, list[int]] = {}
    for part in parts[1:]:
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in {"s8", "u8"}:
            raise ValueError(f"invalid key '{key}' in strategy '{text}'")
        spec[key] = parse_scale_list(value)
    if "s8" not in spec or "u8" not in spec:
        raise ValueError(f"strategy '{text}' must define both s8 and u8 sets")
    return name, spec


def make_macro_header(path: Path, s8: list[int], u8: list[int]) -> None:
    def fmt(values: list[int]) -> str:
        return ", ".join(f"{float(v):.1f}f" for v in values)

    path.write_text(
        "\n".join(
            [
                "#pragma once",
                f"#define ABYZFT_S8_SCALE_COUNT {len(s8)}u",
                f"#define ABYZFT_S8_SCALES {{{fmt(s8)}}}",
                f"#define ABYZFT_U8_SCALE_COUNT {len(u8)}u",
                f"#define ABYZFT_U8_SCALES {{{fmt(u8)}}}",
                "",
            ]
        )
    )


def run_checked(cmd: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def measure_accuracy(stdout: str) -> tuple[int, int, float]:
    m = re.search(r"samples=(\d+)\s+correct=(\d+)\s+accuracy=([0-9]*\.?[0-9]+)", stdout)
    if not m:
        raise ValueError("failed to parse accuracy from runner output")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def measure_detection(stdout: str) -> dict[str, float]:
    metrics = {
        "trivial_det_pct": 0.0,
        "trivial_undetected_acc_drop_mean": 0.0,
        "checkered_det_pct": 0.0,
        "checkered_undetected_acc_drop_mean": 0.0,
    }
    counts = {"trivial": 0, "checkered": 0}
    detected_sum = {"trivial": 0.0, "checkered": 0.0}
    injected_sum = {"trivial": 0.0, "checkered": 0.0}
    trial_re = re.compile(
        r"trial=\d+\s+order=\d+\s+layer=\d+\s+op=(QLinearConv|ConvInteger|Conv)\s+pattern=(trivial|checkered)\s+value="
        r"(?P<value>[0-9]+).*?\s+detected_samples=(?P<detected>[0-9]*\.?[0-9]+).*?\s+fault_injections=(?P<injected>[0-9]*\.?[0-9]+)",
        re.MULTILINE,
    )
    summary_re = re.compile(
        r"^order=\d+\s+layer=\d+\s+op=(QLinearConv|ConvInteger|Conv)\s+pattern=(trivial|checkered)\s+value="
        r"(?P<value>[0-9]+)\s+idx_trials=\d+\s+acc_min=(?P<acc_min>[0-9]*\.?[0-9]+)\s+acc_max=(?P<acc_max>[0-9]*\.?[0-9]+)\s+acc_mean=(?P<acc_mean>[0-9]*\.?[0-9]+)"
        r"(?:\s+acc_corr_min=(?P<acc_corr_min>[0-9]*\.?[0-9]+)\s+acc_corr_max=(?P<acc_corr_max>[0-9]*\.?[0-9]+)\s+acc_corr_mean=(?P<acc_corr_mean>[0-9]*\.?[0-9]+))?",
        re.MULTILINE,
    )
    baseline_re = re.search(r"baseline_accuracy=([0-9]*\.?[0-9]+)", stdout)
    if not baseline_re:
        raise ValueError("failed to parse baseline accuracy from sweep output")
    baseline = float(baseline_re.group(1))
    for match in trial_re.finditer(stdout):
        if int(match.group("value")) == 0:
            continue
        pattern = match.group(2)
        detected_sum[pattern] += float(match.group("detected"))
        injected_sum[pattern] += float(match.group("injected"))
    for match in summary_re.finditer(stdout):
        if int(match.group("value")) == 0:
            continue
        pattern = match.group(2)
        counts[pattern] += 1
        acc_for_drop = (
            float(match.group("acc_corr_mean"))
            if match.group("acc_corr_mean") is not None
            else float(match.group("acc_mean"))
        )
        metrics[f"{pattern}_undetected_acc_drop_mean"] += max(0.0, baseline - acc_for_drop)
    for pattern in counts:
        if injected_sum[pattern] > 0:
            metrics[f"{pattern}_det_pct"] = 100.0 * detected_sum[pattern] / injected_sum[pattern]
        if counts[pattern]:
            metrics[f"{pattern}_undetected_acc_drop_mean"] /= counts[pattern]
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark INT8 AByzFT baseline accuracy under different scale sets.")
    ap.add_argument("--bin-dir", type=Path, default=Path("../../energyrunner/datasets/kws01"))
    ap.add_argument("--label-csv", type=Path, default=Path("../../tiny/benchmark/evaluation/datasets/kws01-open/mfcc/y_labels.csv"))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out-csv", type=Path, default=Path("build/abyzft_scale_benchmark.csv"))
    ap.add_argument("--sweep-limit", type=int, default=100)
    ap.add_argument("--sweep-indexes", type=int, default=2)
    ap.add_argument("--sweep-values", default="1000")
    ap.add_argument("--sweep-patterns", default="trivial,checkered")
    ap.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Custom strategy: name;s8=-4,-2,2,4;u8=2,4,8",
    )
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parent
    strategies = dict(DEFAULT_STRATEGIES)
    for raw in args.strategy:
        name, spec = parse_strategy_arg(raw)
        strategies[name] = spec

    rows: list[dict[str, str]] = []
    print("strategy           accuracy  triv_det%  triv_undrop  chk_det%  chk_undrop  s8_scales")
    print("-----------------  --------  ---------  -----------  ---------  -----------  -----------------------------")

    with tempfile.TemporaryDirectory(prefix="abyzft_scales_") as tmpdir:
        header = Path(tmpdir) / "abyzft_scale_override.h"
        for name, spec in strategies.items():
            make_macro_header(header, spec["s8"], spec["u8"])
            cppflags = f"-include {shlex.quote(str(header))}"

            run_checked(
                [
                    "make",
                    "-B",
                    "aww_int8_abyzft",
                    f"CPPFLAGS={cppflags}",
                ],
                cwd=workdir,
            )

            stdout = run_checked(
                [
                    "./build/aww_int8_abyzft",
                    "--bin-dir",
                    str(args.bin_dir),
                    "--label-csv",
                    str(args.label_csv),
                    "--limit",
                    str(args.limit),
                ],
                cwd=workdir,
            )
            samples, correct, acc = measure_accuracy(stdout)
            rows.append(
                {
                    "strategy": name,
                    "s8_scales": ",".join(str(v) for v in spec["s8"]),
                    "u8_scales": ",".join(str(v) for v in spec["u8"]),
                    "samples": str(samples),
                    "correct": str(correct),
                    "accuracy": f"{acc:.6f}",
                }
            )
            sweep_stdout = run_checked(
                [
                    "./build/aww_int8_abyzft",
                    "--bin-dir",
                    str(args.bin_dir),
                    "--label-csv",
                    str(args.label_csv),
                    "--limit",
                    str(args.sweep_limit),
                    "--sweep",
                    "--sweep-patterns",
                    args.sweep_patterns,
                    "--sweep-values",
                    args.sweep_values,
                    "--sweep-indexes",
                    str(args.sweep_indexes),
                    "--no-progress",
                ],
                cwd=workdir,
            )
            det = measure_detection(sweep_stdout)
            rows[-1].update(
                {
                    "trivial_det_pct": f"{det['trivial_det_pct']:.6f}",
                    "trivial_undetected_acc_drop_mean": f"{det['trivial_undetected_acc_drop_mean']:.6f}",
                    "checkered_det_pct": f"{det['checkered_det_pct']:.6f}",
                    "checkered_undetected_acc_drop_mean": f"{det['checkered_undetected_acc_drop_mean']:.6f}",
                }
            )
            print(
                f"{name:<17}  {acc:>8.4f}  {det['trivial_det_pct']:>9.2f}  {det['trivial_undetected_acc_drop_mean']:>11.4f}  "
                f"{det['checkered_det_pct']:>9.2f}  {det['checkered_undetected_acc_drop_mean']:>11.4f}  "
                f"{','.join(str(v) for v in spec['s8']):<29}"
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "s8_scales",
                "u8_scales",
                "samples",
                "correct",
                "accuracy",
                "trivial_det_pct",
                "trivial_undetected_acc_drop_mean",
                "checkered_det_pct",
                "checkered_undetected_acc_drop_mean",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nwrote CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
