#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot runtime overhead bar chart")
    ap.add_argument("--results-csv", type=Path, required=True, help="CSV emitted by benchmark_runtime.py")
    ap.add_argument("--quant", type=str, default="", help="Optional quant filter, e.g. fp32 or int8")
    ap.add_argument("--baseline-raw", type=str, default="baseline_raw", help="Name of raw baseline")
    ap.add_argument("--baseline-im2col", type=str, default="baseline", help="Name of im2col baseline")
    ap.add_argument("--out-pdf", type=Path, default=Path("runtime_overhead.pdf"))
    ap.add_argument("--title", type=str, default="Runtime Overhead vs Baselines")
    args = ap.parse_args()

    times = {}
    with open(args.results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            quant = (row.get("quant") or "").strip()
            if args.quant and quant != args.quant:
                continue
            name = (row.get("mechanism") or "").strip()
            median_s = (row.get("median_s") or "").strip()
            if not name or not median_s:
                continue
            try:
                times[name] = float(median_s)
            except ValueError:
                continue

    if not times:
        print(f"Error: No data found in {args.results_csv}", file=sys.stderr)
        return 1

    raw_time = times.get(args.baseline_raw)
    im2col_time = times.get(args.baseline_im2col)

    if raw_time is None:
        print(f"Error: Baseline '{args.baseline_raw}' not found. Available: {list(times.keys())}", file=sys.stderr)
        return 1
    if im2col_time is None:
        print(f"Warning: Baseline '{args.baseline_im2col}' not found, using raw", file=sys.stderr)
        im2col_time = raw_time

    data = []
    for name, t in times.items():
        oh_raw = (t - raw_time) / raw_time * 100
        oh_im2col = (t - im2col_time) / im2col_time * 100
        data.append((name, t, oh_raw, oh_im2col))

    data.sort(key=lambda x: x[1])

    names = [x[0] for x in data]
    times_list = [x[1] for x in data]
    overhead_raw = [x[2] for x in data]
    overhead_im2col = [x[3] for x in data]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["gray", "steelblue", "coral", "forestgreen", "purple", "orange", "teal", "magenta"]
    if len(names) > len(colors):
        colors = colors * ((len(names) // len(colors)) + 1)

    bars = ax.bar(names, times_list, color=colors[:len(names)], edgecolor="black", linewidth=0.5)

    for name, bar, t, oh_raw, oh_im2col in zip(names, bars, times_list, overhead_raw, overhead_im2col):
        x = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()

        ax.text(x, height + 0.015, f"{t:.3f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

        if name == args.baseline_im2col:
            continue
        if name == args.baseline_raw:
            ax.text(
                x,
                height + 0.06,
                f"{oh_raw:+.1f}% vs raw",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )
            ax.text(
                x,
                height + 0.085,
                f"{oh_im2col:+.1f}% vs im2col",
                ha="center",
                va="bottom",
                fontsize=7,
                color="steelblue",
            )
            continue

        ax.text(
            x,
            height + 0.06,
            f"{oh_raw:+.1f}% vs raw",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
        )
        ax.text(
            x,
            height + 0.085,
            f"{oh_im2col:+.1f}% vs im2col",
            ha="center",
            va="bottom",
            fontsize=7,
            color="steelblue",
        )

    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_xlabel("Mechanism", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(args.title, fontsize=12)

    max_time = max(times_list)
    ax.set_ylim(0, max_time * 1.25)

    ax.axhline(y=im2col_time, color="steelblue", linestyle="--", linewidth=1, alpha=0.5, label="im2col baseline")
    ax.axhline(y=raw_time, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="raw baseline")

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(args.out_pdf)
    print(f"Wrote {args.out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
