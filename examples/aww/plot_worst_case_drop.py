#!/usr/bin/env python3
import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


@dataclass(frozen=True)
class SweepRow:
    layer: int
    op: str
    pattern: str
    value: float
    acc_min: float
    acc_max: float
    acc_mean: float
    acc_corr_min: Optional[float] = None


@dataclass(frozen=True)
class ParsedLog:
    baseline_acc: Optional[float]
    rows: List[SweepRow]
    layer_order: List[int]
    layer_op: Dict[int, str]


_RE_BASELINE = re.compile(r"\bsamples=(\d+)\s+correct=(\d+)\s+accuracy=([0-9]*\.?[0-9]+)\b")
_RE_BASELINE_SWEEP = re.compile(r"\bbaseline_accuracy=([0-9]*\.?[0-9]+)\b")
_RE_SWEEP = re.compile(
    r"\blayer=(\d+)\s+op=([^\s]+)\s+pattern=([^\s]+)\s+value=([0-9eE\+\-\.]+)\s+"
    r"idx_trials=\d+\s+acc_min=([0-9]*\.?[0-9]+)\s+acc_max=([0-9]*\.?[0-9]+)\s+acc_mean=([0-9]*\.?[0-9]+)"
    r"(?:\s+acc_corr_min=([0-9]*\.?[0-9]+)\s+acc_corr_max=([0-9]*\.?[0-9]+)\s+acc_corr_mean=([0-9]*\.?[0-9]+))?"
)


def _parse_log(path: Path) -> ParsedLog:
    baseline_acc: Optional[float] = None
    rows: List[SweepRow] = []
    layer_order: List[int] = []
    layer_op: Dict[int, str] = {}

    for line in path.read_text(errors="replace").splitlines():
        if baseline_acc is None:
            mh = _RE_BASELINE_SWEEP.search(line)
            if mh:
                baseline_acc = float(mh.group(1))
                continue
            m0 = _RE_BASELINE.search(line)
            if m0:
                baseline_acc = float(m0.group(3))
                continue

        m = _RE_SWEEP.search(line)
        if not m:
            continue

        layer = int(m.group(1))
        op = m.group(2)
        if layer not in layer_op:
            layer_op[layer] = op
        if layer not in layer_order:
            layer_order.append(layer)

        rows.append(
            SweepRow(
                layer=layer,
                op=op,
                pattern=m.group(3),
                value=float(m.group(4)),
                acc_min=float(m.group(5)),
                acc_max=float(m.group(6)),
                acc_mean=float(m.group(7)),
                acc_corr_min=float(m.group(8)) if m.group(8) is not None else None,
            )
        )

    if baseline_acc is None and rows:
        baseline_acc = max(r.acc_max for r in rows)

    return ParsedLog(baseline_acc=baseline_acc, rows=rows, layer_order=layer_order, layer_op=layer_op)


def _value_key(v: float) -> str:
    return f"{v:g}"


def _gather_pairs(log_dir: Path) -> List[Tuple[str, Path, Path]]:
    fp32_logs: Dict[str, Path] = {}
    int8_logs: Dict[str, Path] = {}

    for p in sorted(log_dir.glob("output_fp32_*.log")):
        mech = p.name[len("output_fp32_") : -len(".log")]
        fp32_logs[mech] = p
    for p in sorted(log_dir.glob("output_int8_*.log")):
        mech = p.name[len("output_int8_") : -len(".log")]
        int8_logs[mech] = p

    mechs = sorted(set(fp32_logs.keys()) & set(int8_logs.keys()))
    return [(m, fp32_logs[m], int8_logs[m]) for m in mechs]


def _build_worst_drop_table(parsed: ParsedLog, *, undetected_only: bool) -> Dict[Tuple[str, float, int], float]:
    if parsed.baseline_acc is None:
        raise ValueError("Missing baseline accuracy in log")

    out: Dict[Tuple[str, float, int], float] = {}
    baseline = parsed.baseline_acc
    for r in parsed.rows:
        acc_for_drop = r.acc_corr_min if (undetected_only and r.acc_corr_min is not None) else r.acc_min
        drop = baseline - acc_for_drop
        if drop < 0.0:
            drop = 0.0
        key = (r.pattern, r.value, r.layer)
        prev = out.get(key)
        if prev is None or drop > prev:
            out[key] = drop
    return out


def _draw_heatmap(ax, matrix, xlabels: List[str], ylabels: List[str], subtitle: str, vmax: float) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    cmap = plt.get_cmap("magma")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(subtitle, fontsize=10)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Protection mechanism")

    for yi in range(len(ylabels)):
        for xi in range(len(xlabels)):
            val = matrix[yi, xi]
            if not np.isfinite(val):
                txt = "-"
                color = "#d0d0d0"
                outline = "#000000"
            else:
                txt = f"{100.0*val:.1f}%"
                norm = 0.0 if vmax <= 0 else min(1.0, max(0.0, float(val) / float(vmax)))
                r, g, b, _ = cmap(norm)
                # Perceived luminance (sRGB weights).
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                color = "white" if lum < 0.45 else "black"
                outline = "black" if color == "white" else "white"
            ax.text(
                xi,
                yi,
                txt,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                path_effects=[pe.withStroke(linewidth=1.4, foreground=outline, alpha=0.9)],
            )
    return im


def _plot_heatmap_pair(
    out_path: Path,
    title: str,
    matrix_overall,
    matrix_undetected,
    xlabels: List[str],
    ylabels: List[str],
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    fig_w = max(10, 0.48 * len(xlabels))
    fig_h_single = max(3.5, 0.42 * len(ylabels) + 1.8)
    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(fig_w, fig_h_single * 2 + 0.8), constrained_layout=True
    )

    vmax_candidates = []
    if np.isfinite(matrix_overall).any():
        vmax_candidates.append(float(np.nanmax(matrix_overall)))
    if np.isfinite(matrix_undetected).any():
        vmax_candidates.append(float(np.nanmax(matrix_undetected)))
    vmax = max(vmax_candidates) if vmax_candidates else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig.suptitle(title, fontsize=11)
    im = _draw_heatmap(ax0, matrix_overall, xlabels, ylabels, "Worst-case overall (baseline - acc_min)", vmax)
    _draw_heatmap(ax1, matrix_undetected, xlabels, ylabels, "Worst-case undetected (baseline - acc_corr_min)", vmax)

    cbar = fig.colorbar(im, ax=[ax0, ax1], fraction=0.02, pad=0.02)
    cbar.set_label("Worst-case accuracy drop")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_summary_bars(out_path: Path, title: str, values: List[float], mechs: List[str], score_by: Dict[Tuple[str, float], float]) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    if not values or not mechs:
        return

    x = np.arange(len(values), dtype=float)
    width = 0.82 / max(1, len(mechs))

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(values)), 5))

    for i, mech in enumerate(mechs):
        offs = (i - (len(mechs) - 1) / 2.0) * width
        y = [score_by.get((mech, v), 0.0) for v in values]
        ax.bar(x + offs, y, width=width, label=mech, alpha=0.9)

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.set_xlabel("Fault value")
    ax.set_ylabel("Worst layer drop")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, ncol=min(4, len(mechs)))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Worst-case layer-drop comparison across protection mechanisms.")
    ap.add_argument("--log-dir", type=Path, default=Path("."), help="Directory containing output_fp32_*.log and output_int8_*.log")
    ap.add_argument("--out-dir", type=Path, default=Path("plots_worst_case"))
    ap.add_argument("--only-mechanisms", type=str, default="", help="Comma-separated subset, e.g. abft,abyzft,freivalds1x")
    ap.add_argument("--raw", action="store_true", help="Use raw accuracy (acc_min) instead of undetected-only (acc_corr_min when available).")
    args = ap.parse_args()

    pairs = _gather_pairs(args.log_dir)
    if not pairs:
        raise SystemExit("No matching fp32/int8 log pairs found.")

    only = {x.strip() for x in args.only_mechanisms.split(",") if x.strip()}
    if only:
        pairs = [p for p in pairs if p[0] in only]
    if not pairs:
        raise SystemExit("No log pairs left after --only-mechanisms filter.")

    parsed: Dict[Tuple[str, str], ParsedLog] = {}
    for mech, fp32_log, int8_log in pairs:
        parsed[(mech, "fp32")] = _parse_log(fp32_log)
        parsed[(mech, "int8")] = _parse_log(int8_log)

    mechs = [m for (m, _, _) in pairs]
    undetected_only = not args.raw

    rows_csv: List[Dict[str, str]] = []

    for quant in ("fp32", "int8"):
        scenarios: Dict[Tuple[str, float], bool] = {}
        for mech in mechs:
            for r in parsed[(mech, quant)].rows:
                scenarios[(r.pattern, r.value)] = True

        for pattern, value in sorted(scenarios.keys(), key=lambda x: (x[0], x[1])):
            layer_union: List[int] = []
            for mech in mechs:
                pl = parsed[(mech, quant)]
                layer_set = {r.layer for r in pl.rows if r.pattern == pattern and r.value == value}
                for lid in pl.layer_order:
                    if lid in layer_set and lid not in layer_union:
                        layer_union.append(lid)

            if not layer_union:
                continue

            import numpy as np

            mat_overall = np.full((len(mechs), len(layer_union)), np.nan, dtype=float)
            mat_undetected = np.full((len(mechs), len(layer_union)), np.nan, dtype=float)

            for yi, mech in enumerate(mechs):
                pl = parsed[(mech, quant)]
                table_overall = _build_worst_drop_table(pl, undetected_only=False)
                table_undetected = _build_worst_drop_table(pl, undetected_only=True)
                for xi, layer in enumerate(layer_union):
                    v_overall = table_overall.get((pattern, value, layer))
                    v_undetected = table_undetected.get((pattern, value, layer))
                    if v_overall is not None:
                        mat_overall[yi, xi] = v_overall
                    if v_undetected is not None:
                        mat_undetected[yi, xi] = v_undetected
                    if v_undetected is not None:
                        rows_csv.append(
                            {
                                "quant": quant,
                                "mechanism": mech,
                                "fault_model": pattern,
                                "fault_value": _value_key(value),
                                "layer": str(layer),
                                "op": pl.layer_op.get(layer, "?"),
                                "worst_drop": f"{v_undetected:.8f}",
                            }
                        )

            xlabels = [f"L{l}:{parsed[(mechs[0], quant)].layer_op.get(l, '?')}" for l in layer_union]
            title = f"Worst-case drop | {quant.upper()} | pattern={pattern} value={value:g}"
            out_png = args.out_dir / quant / f"heatmap_{pattern}_v{_value_key(value).replace('.', 'p')}.png"
            _plot_heatmap_pair(out_png, title, mat_overall, mat_undetected, xlabels, mechs)

        for pattern in sorted({p for (p, _) in scenarios.keys()}):
            values = sorted({v for (p, v) in scenarios.keys() if p == pattern})
            score_by: Dict[Tuple[str, float], float] = {}
            for mech in mechs:
                table = _build_worst_drop_table(parsed[(mech, quant)], undetected_only=undetected_only)
                for v in values:
                    best = 0.0
                    for layer in parsed[(mech, quant)].layer_order:
                        w = table.get((pattern, v, layer))
                        if w is not None and w > best:
                            best = w
                    score_by[(mech, v)] = best
            out_png = args.out_dir / quant / f"summary_{pattern}.png"
            _plot_summary_bars(
                out_png,
                f"Worst-layer drop by mechanism | {quant.upper()} | pattern={pattern}",
                values,
                mechs,
                score_by,
            )

    csv_path = args.out_dir / "worst_case_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["quant", "mechanism", "fault_model", "fault_value", "layer", "op", "worst_drop"],
        )
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    print(f"wrote outputs to {args.out_dir} (rows={len(rows_csv)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
