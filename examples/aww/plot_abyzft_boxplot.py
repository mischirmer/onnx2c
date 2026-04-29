#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_kv_line(line: str):
    out = {}
    for k, v in re.findall(r"(\w+)=([^\s]+)", line):
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="sweep log containing trial lines")
    ap.add_argument("--out", type=Path, default=Path("abyzft_accuracy_drop_boxplot.png"))
    ap.add_argument("--metric", default="accuracy_drop",
                    choices=["accuracy_drop", "accuracy_corrected_drop", "detected_sample_rate"])
    ap.add_argument("--pattern", default=None, help="optional fault pattern filter")
    ap.add_argument("--value", default=None, help="optional fault value filter, e.g. 10")
    args = ap.parse_args()

    rows = []
    for line in args.log.read_text(errors="replace").splitlines():
        if "trial=" in line:
            rows.append(parse_kv_line(line[line.index("trial="):]))
            continue
        if line.startswith("trial "):
            rows.append(parse_kv_line(line))

    if not rows:
        raise SystemExit("no trial rows found; regenerate the log with the updated sweep binary")

    df = pd.DataFrame(rows)

    for c in [
        "order", "layer", "value", "trial", "fault_index",
        "accuracy", "accuracy_drop",
        "accuracy_corrected", "accuracy_corrected_drop",
        "detected_samples", "detected_sample_rate",
        "tampering_detections", "fault_injections",
    ]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "detected_sample_rate" not in df and {"detected_samples", "samples"}.issubset(df.columns):
        df["detected_sample_rate"] = df["detected_samples"] / df["samples"]

    if args.pattern:
        df = df[df["pattern"] == args.pattern]
    if args.value is not None:
        df = df[df["value"].astype(str) == str(args.value)]

    if df.empty:
        raise SystemExit("no rows left after filtering")

    df["fault_config"] = df["pattern"].astype(str) + " / " + df["value"].astype(str)

    labels = []
    data = []

    for (order, layer, op, cfg), g in df.sort_values(["order", "pattern", "value"]).groupby(
        ["order", "layer", "op", "fault_config"], sort=False
    ):
        labels.append(f"L{int(layer)} {op}\n{cfg}")
        data.append(g[args.metric].dropna().to_numpy())

    plt.figure(figsize=(max(12, len(labels) * 0.45), 6))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xticks(rotation=90)
    plt.ylabel(args.metric)
    plt.title(f"AByzFT per-fault {args.metric}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()