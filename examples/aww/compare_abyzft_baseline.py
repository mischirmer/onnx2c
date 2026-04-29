#!/usr/bin/env python3
"""
Analyze AByzFT scale strategy results by comparing against baseline accuracy.
Adds baseline_accuracy and accuracy_drop columns to the scale benchmark CSV.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def measure_baseline_accuracy(workdir: Path, bin_dir: Path, label_csv: Path, limit: int) -> float:
    """Run baseline (no AByzFT) binary and extract accuracy."""
    proc = subprocess.run(
        [
            str(workdir / "build" / "aww_int8_baseline"),
            "--bin-dir",
            str(bin_dir),
            "--label-csv",
            str(label_csv),
            "--limit",
            str(limit),
        ],
        cwd=workdir,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"failed to run baseline; did you build aww_int8_baseline?", file=sys.stderr)
        sys.exit(1)
    
    m = re.search(r"samples=\d+\s+correct=\d+\s+accuracy=([0-9]*\.?[0-9]+)", proc.stdout)
    if not m:
        print(f"failed to parse baseline accuracy from output", file=sys.stderr)
        sys.exit(1)
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare AByzFT scale strategies against baseline accuracy."
    )
    ap.add_argument("--results-csv", type=Path, required=True, help="output from benchmark_abyzft_scales.py")
    ap.add_argument("--bin-dir", type=Path, default=Path("../../energyrunner/datasets/kws01"))
    ap.add_argument("--label-csv", type=Path, default=Path("../../tiny/benchmark/evaluation/datasets/kws01-open/mfcc/y_labels.csv"))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parent
    
    # measure baseline accuracy
    print(f"Measuring baseline accuracy...", file=sys.stderr)
    baseline_acc = measure_baseline_accuracy(workdir, args.bin_dir, args.label_csv, args.limit)
    print(f"Baseline accuracy: {baseline_acc:.6f}\n", file=sys.stderr)

    # read results CSV
    rows = []
    with args.results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            abyzft_acc = float(row["accuracy"])
            drop = baseline_acc - abyzft_acc
            row["baseline_accuracy"] = f"{baseline_acc:.6f}"
            row["accuracy_drop"] = f"{drop:.6f}"
            rows.append(row)

    # print summary table
    print("strategy           baseline  abyzft    drop      triv_det  triv_drop  chk_det   chk_drop")
    print("-" * 100)
    for row in rows:
        baseline = float(row["baseline_accuracy"])
        abyzft = float(row["accuracy"])
        drop = float(row["accuracy_drop"])
        triv_det = float(row["trivial_det_mean"])
        triv_drop = float(row["trivial_acc_drop_mean"])
        chk_det = float(row["checkered_det_mean"])
        chk_drop = float(row["checkered_acc_drop_mean"])
        print(
            f"{row['strategy']:<17}  {baseline:>8.4f}  {abyzft:>8.4f}  {drop:>8.4f}  "
            f"{triv_det:>8.2f}  {triv_drop:>9.4f}  {chk_det:>8.2f}  {chk_drop:>9.4f}"
        )

    # write output CSV if requested
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\nwrote CSV: {args.out_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
