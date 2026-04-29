#!/usr/bin/env python3
"""
Plot one runtime/overhead chart per model from benchmark_all_models.py output.

Usage:
  python3 plot_model_runtime_overheads.py \
    --results-csv build/all_model_runtimes.csv \
    --out-dir build/runtime_plots
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Avoid matplotlib trying to write under ~/.config on restricted systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")


MECHANISM_ORDER = [
    "baseline_raw",
    "baseline",
    "abft",
    "abyzft",
    "freivalds1x",
    "freivalds2x",
    "freivalds3x",
    "freivalds4x",
    "gvfa1x",
    "gvfa2x",
]

COLORS = {
    "baseline_raw": "gray",
    "baseline": "steelblue",
    "abft": "coral",
    "abyzft": "forestgreen",
    "freivalds1x": "purple",
    "freivalds2x": "orange",
    "freivalds3x": "darkgoldenrod",
    "freivalds4x": "firebrick",
    "gvfa1x": "teal",
    "gvfa2x": "magenta",
}


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def _read_results(path: Path, metric: str) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            model = (row.get("model") or "").strip()
            mechanism = (row.get("mechanism") or "").strip()
            value = (row.get(metric) or "").strip()
            if not model or not mechanism or not value:
                continue
            try:
                t = float(value)
            except ValueError:
                continue
            if math.isnan(t) or t <= 0.0:
                continue
            by_model.setdefault(model, {})[mechanism] = t
    return by_model


def _ordered_items(times: Dict[str, float]) -> List[Tuple[str, float]]:
    known = [(m, times[m]) for m in MECHANISM_ORDER if m in times]
    unknown = sorted((m, t) for m, t in times.items() if m not in MECHANISM_ORDER)
    return known + unknown


def _plot_model(
    model: str,
    times: Dict[str, float],
    out_path: Path,
    metric: str,
    baseline_raw: str,
    baseline_im2col: str,
    title_prefix: str,
) -> bool:
    raw_time = times.get(baseline_raw)
    im2col_time = times.get(baseline_im2col)

    if raw_time is None:
        print(f"Skip {model}: baseline '{baseline_raw}' not found", file=sys.stderr)
        return False
    if im2col_time is None:
        print(f"Warning: {model}: baseline '{baseline_im2col}' not found, using raw", file=sys.stderr)
        im2col_time = raw_time

    data = _ordered_items(times)
    names = [x[0] for x in data]
    runtimes = [x[1] for x in data]

    import matplotlib.pyplot as plt

    fig, (ax_runtime, ax_overhead) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(10, 1.1 * len(names)), 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )

    bar_colors = [COLORS.get(name, "slategray") for name in names]
    bars = ax_runtime.bar(names, runtimes, color=bar_colors, edgecolor="black", linewidth=0.5)

    y_pad = max(runtimes) * 0.02
    for bar, runtime in zip(bars, runtimes):
        x = bar.get_x() + bar.get_width() / 2
        ax_runtime.text(
            x,
            runtime + y_pad,
            f"{runtime:.4g}s",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax_runtime.axhline(raw_time, color="gray", linestyle="--", linewidth=1, alpha=0.55, label="raw baseline")
    ax_runtime.axhline(
        im2col_time,
        color="steelblue",
        linestyle="--",
        linewidth=1,
        alpha=0.55,
        label="baseline",
    )
    ax_runtime.set_ylabel(f"{metric.replace('_', ' ')} (seconds)")
    ax_runtime.set_title(f"{title_prefix}: {model}" if title_prefix else model)
    ax_runtime.set_ylim(0, max(runtimes) * 1.18)
    ax_runtime.legend(loc="upper right")
    ax_runtime.grid(axis="y", alpha=0.2)

    x_positions = list(range(len(names)))
    overhead_vs_raw = [(t - raw_time) / raw_time * 100.0 for t in runtimes]
    overhead_vs_im2col = [(t - im2col_time) / im2col_time * 100.0 for t in runtimes]
    width = 0.38

    ax_overhead.bar(
        [x - width / 2 for x in x_positions],
        overhead_vs_raw,
        width=width,
        color="gray",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.4,
        label="vs raw",
    )
    ax_overhead.bar(
        [x + width / 2 for x in x_positions],
        overhead_vs_im2col,
        width=width,
        color="steelblue",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.4,
        label="vs baseline",
    )
    ax_overhead.axhline(0.0, color="black", linewidth=0.8)
    ax_overhead.set_ylabel("Overhead (%)")
    ax_overhead.set_xlabel("Mechanism")
    ax_overhead.set_xticks(x_positions)
    ax_overhead.set_xticklabels(names, rotation=35, ha="right")
    ax_overhead.legend(loc="upper right")
    ax_overhead.grid(axis="y", alpha=0.2)

    for x, oh_raw, oh_base in zip(x_positions, overhead_vs_raw, overhead_vs_im2col):
        y = max(oh_raw, oh_base)
        va = "bottom"
        offset = 1.5
        if y < 0:
            y = min(oh_raw, oh_base)
            va = "top"
            offset = -1.5
        ax_overhead.text(
            x,
            y + offset,
            f"{oh_raw:+.1f}% raw\n{oh_base:+.1f}% base",
            ha="center",
            va=va,
            fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot per-model runtime and overhead charts.")
    ap.add_argument("--results-csv", type=Path, required=True, help="CSV emitted by benchmark_all_models.py")
    ap.add_argument("--out-dir", type=Path, default=Path("build/runtime_plots"))
    ap.add_argument("--metric", choices=("median_s", "mean_s"), default="median_s")
    ap.add_argument("--baseline-raw", type=str, default="baseline_raw", help="Name of raw baseline mechanism")
    ap.add_argument("--baseline", type=str, default="baseline", help="Name of baseline/im2col mechanism")
    ap.add_argument("--format", choices=("pdf", "png", "svg"), default="pdf")
    ap.add_argument("--title-prefix", type=str, default="Runtime Overhead")
    args = ap.parse_args()

    results = _read_results(args.results_csv, args.metric)
    if not results:
        print(f"Error: no usable rows found in {args.results_csv}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for model in sorted(results):
        out_path = args.out_dir / f"{_safe_stem(model)}_runtime_overhead.{args.format}"
        if _plot_model(
            model=model,
            times=results[model],
            out_path=out_path,
            metric=args.metric,
            baseline_raw=args.baseline_raw,
            baseline_im2col=args.baseline,
            title_prefix=args.title_prefix,
        ):
            written += 1
            print(f"Wrote {out_path}")

    print(f"Wrote {written} plot(s) to {args.out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
