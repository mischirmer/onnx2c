#!/usr/bin/env python3
import argparse
import csv
import h5py
import matplotlib
import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")


@dataclass
class LayerResult:
    layer: int
    op: str
    pattern: str
    value: float
    acc_min: float
    acc_max: float
    acc_mean: float
    acc_corr_min: Optional[float]
    acc_corr_max: Optional[float]
    acc_corr_mean: Optional[float]
    det_min: Optional[float]
    det_max: Optional[float]
    det_mean: Optional[float]


@dataclass
class ExperimentResult:
    exp_id: str
    mechanism: str
    quant: str
    pattern: str
    value: float
    status: str
    baseline_accuracy: float
    baseline_accuracy_corrected: Optional[float]
    layers: List[LayerResult]


def _optional_attr(attrs: h5py.AttributeManager, name: str) -> Optional[float]:
    value = attrs.get(name)
    if value is None:
        return None
    return float(value)


def _load_results(h5_path: Path) -> List[ExperimentResult]:
    results: List[ExperimentResult] = []
    with h5py.File(h5_path, "r") as f:
        for mech_grp in f["experiments"].values():
            for exp_id, exp_grp in mech_grp.items():
                status = str(exp_grp.attrs.get("status", "pending"))
                if status != "completed":
                    continue
                baseline_accuracy = _optional_attr(exp_grp.attrs, "baseline_accuracy")
                if baseline_accuracy is None:
                    continue

                layers: List[LayerResult] = []
                if "layers" in exp_grp:
                    for layer_grp in exp_grp["layers"].values():
                        layers.append(
                            LayerResult(
                                layer=int(layer_grp.attrs["layer"]),
                                op=str(layer_grp.attrs["op"]),
                                pattern=str(layer_grp.attrs["pattern"]),
                                value=float(layer_grp.attrs["value"]),
                                acc_min=float(layer_grp.attrs["acc_min"]),
                                acc_max=float(layer_grp.attrs["acc_max"]),
                                acc_mean=float(layer_grp.attrs["acc_mean"]),
                                acc_corr_min=_optional_attr(layer_grp.attrs, "acc_corr_min"),
                                acc_corr_max=_optional_attr(layer_grp.attrs, "acc_corr_max"),
                                acc_corr_mean=_optional_attr(layer_grp.attrs, "acc_corr_mean"),
                                det_min=_optional_attr(layer_grp.attrs, "det_min"),
                                det_max=_optional_attr(layer_grp.attrs, "det_max"),
                                det_mean=_optional_attr(layer_grp.attrs, "det_mean"),
                            )
                        )

                results.append(
                    ExperimentResult(
                        exp_id=exp_id,
                        mechanism=str(exp_grp.attrs["mechanism"]),
                        quant=str(exp_grp.attrs["quant"]),
                        pattern=str(exp_grp.attrs["pattern"]),
                        value=float(exp_grp.attrs["value"]),
                        status=status,
                        baseline_accuracy=baseline_accuracy,
                        baseline_accuracy_corrected=_optional_attr(exp_grp.attrs, "baseline_accuracy_corrected"),
                        layers=layers,
                    )
                )
    return results


def _unique_sorted(xs: Iterable) -> List:
    return sorted(set(xs))


def _build_worst_drop_table(
    results: List[ExperimentResult],
    *,
    undetected_only: bool,
) -> Dict[Tuple[str, str, float, int], float]:
    out: Dict[Tuple[str, str, float, int], float] = {}
    for r in results:
        for layer_row in r.layers:
            acc_for_drop = layer_row.acc_corr_min if (undetected_only and layer_row.acc_corr_min is not None) else layer_row.acc_min
            drop = r.baseline_accuracy - acc_for_drop
            if drop < 0.0:
                drop = 0.0
            key = (r.mechanism, r.pattern, r.value, layer_row.layer)
            prev = out.get(key)
            if prev is None or drop > prev:
                out[key] = drop
    return out


def _plot_stacked_single_heatmaps(out_path: Path, title: str, value_blocks: List[Tuple[float, np.ndarray, List[str]]], ylabels: List[str]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    if not value_blocks:
        return

    fig_w = max(10, max(0.44 * len(xlabels) for _, _, xlabels in value_blocks))
    fig_h = max(3.2, len(value_blocks) * max(2.0, 0.22 * len(ylabels) + 0.9))
    fig, axes = plt.subplots(nrows=len(value_blocks), ncols=1, figsize=(fig_w, fig_h), squeeze=True)
    if len(value_blocks) == 1:
        axes = [axes]

    vmax_candidates = []
    for _, matrix_combined, _ in value_blocks:
        if np.isfinite(matrix_combined).any():
            vmax_candidates.append(float(np.nanmax(matrix_combined)))
    vmax = max(vmax_candidates) if vmax_candidates else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig.suptitle(title, fontsize=11)

    for ri, (value, matrix_combined, xlabels) in enumerate(value_blocks):
        ax = axes[ri]
        cmap = plt.get_cmap("magma")
        im = ax.imshow(matrix_combined, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(f"value={value:g}", fontsize=10)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_ylabel("Protection")

        for yi in range(len(ylabels)):
            for xi in range(len(xlabels)):
                val = matrix_combined[yi, xi]
                if not np.isfinite(val):
                    txt = "-"
                    color = "#d0d0d0"
                    outline = "#000000"
                else:
                    txt = f"{100.0 * val:.1f}%"
                    norm = min(1.0, max(0.0, float(val) / float(vmax)))
                    r, g, b, _ = cmap(norm)
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    color = "white" if lum < 0.45 else "black"
                    outline = "black" if color == "white" else "white"
                ax.text(
                    xi,
                    yi,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground=outline, alpha=0.9)],
                )

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Worst-case accuracy drop")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_summary_bars(out_path: Path, title: str, values: List[float], mechs: List[str], score_by: Dict[Tuple[str, float], float]) -> None:
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


def _export_csv(results: List[ExperimentResult], out_path: Path) -> None:
    rows_csv: List[Dict[str, str]] = []
    for r in results:
        for layer_row in r.layers:
            acc_for_drop = layer_row.acc_corr_min if layer_row.acc_corr_min is not None else layer_row.acc_min
            drop = r.baseline_accuracy - acc_for_drop
            if drop < 0.0:
                drop = 0.0
            rows_csv.append(
                {
                    "quant": r.quant,
                    "mechanism": r.mechanism,
                    "fault_model": r.pattern,
                    "fault_value": f"{r.value:g}",
                    "layer": str(layer_row.layer),
                    "op": layer_row.op,
                    "baseline_accuracy": f"{r.baseline_accuracy:.8f}",
                    "worst_drop": f"{drop:.8f}",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "quant",
                "mechanism",
                "fault_model",
                "fault_value",
                "layer",
                "op",
                "baseline_accuracy",
                "worst_drop",
            ],
        )
        writer.writeheader()
        for row in rows_csv:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze fault injection campaign results.")
    ap.add_argument("--h5", type=Path, default=Path("campaign.h5"), help="Input HDF5 campaign file")
    ap.add_argument("--out-dir", type=Path, default=Path("plots_worst_case"), help="Output directory for analysis")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"Campaign file not found: {args.h5}")

    results = _load_results(args.h5)
    if not results:
        raise SystemExit("No completed experiments found in campaign.")

    mechs = _unique_sorted(r.mechanism for r in results)

    for quant in ("fp32", "int8"):
        results_quant = [r for r in results if r.quant == quant]
        if not results_quant:
            continue

        for pattern in sorted({r.pattern for r in results_quant}):
            values = sorted({r.value for r in results_quant if r.pattern == pattern})
            table_overall = _build_worst_drop_table(results_quant, undetected_only=False)
            table_undetected = _build_worst_drop_table(results_quant, undetected_only=True)
            value_blocks: List[Tuple[float, np.ndarray, List[str]]] = []

            for value in values:
                layer_union: List[int] = []
                for r in results_quant:
                    if r.pattern != pattern or r.value != value:
                        continue
                    for layer_row in r.layers:
                        if layer_row.layer not in layer_union:
                            layer_union.append(layer_row.layer)
                layer_union = sorted(layer_union)
                if not layer_union:
                    continue

                mat_overall = np.full((len(mechs), len(layer_union)), np.nan, dtype=float)
                mat_undetected = np.full((len(mechs), len(layer_union)), np.nan, dtype=float)

                for yi, mech in enumerate(mechs):
                    for xi, layer in enumerate(layer_union):
                        v_overall = table_overall.get((mech, pattern, value, layer))
                        v_undetected = table_undetected.get((mech, pattern, value, layer))
                        if v_overall is not None:
                            mat_overall[yi, xi] = v_overall
                        if v_undetected is not None:
                            mat_undetected[yi, xi] = v_undetected

                xlabels = [f"L{layer}" for layer in layer_union]
                none_row = np.nanmax(mat_overall, axis=0, keepdims=True)
                mat_combined = np.vstack([none_row, mat_undetected])
                value_blocks.append((value, mat_combined, xlabels))

            if value_blocks:
                out_png = args.out_dir / f"heatmap_stacked_{pattern}_{quant}.png"
                _plot_stacked_single_heatmaps(
                    out_png,
                    f"Worst-case drop | {quant.upper()} | pattern={pattern}",
                    value_blocks,
                    ["none"] + mechs,
                )

            score_by: Dict[Tuple[str, float], float] = {}
            for mech in mechs:
                for value in values:
                    best = 0.0
                    for r in results_quant:
                        if r.mechanism != mech or r.pattern != pattern or r.value != value:
                            continue
                        for layer_row in r.layers:
                            acc_for_drop = layer_row.acc_corr_min if layer_row.acc_corr_min is not None else layer_row.acc_min
                            drop = r.baseline_accuracy - acc_for_drop
                            if drop < 0.0:
                                drop = 0.0
                            if drop > best:
                                best = drop
                    score_by[(mech, value)] = best

            if score_by:
                out_png = args.out_dir / f"summary_{pattern}_{quant}.png"
                _plot_summary_bars(
                    out_png,
                    f"Worst-layer drop by mechanism | {quant.upper()} | pattern={pattern}",
                    values,
                    mechs,
                    score_by,
                )

    csv_path = args.out_dir / "worst_case_table.csv"
    _export_csv(results, csv_path)

    print(f"wrote outputs to {args.out_dir}")
    print(f"total experiments: {len(results)}")
    print(f"completed: {sum(1 for r in results if r.status == 'completed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
