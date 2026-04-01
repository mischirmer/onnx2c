#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Avoid matplotlib trying to write under ~/.config on restricted systems.
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
	acc_corr_max: Optional[float] = None
	acc_corr_mean: Optional[float] = None


@dataclass(frozen=True)
class ParsedLog:
	baseline_acc: Optional[float]
	rows: List[SweepRow]
	layer_order: List[int]
	layer_op: Dict[int, str]


_RE_BASELINE = re.compile(r"\bsamples=(\d+)\s+correct=(\d+)\s+accuracy=([0-9]*\.?[0-9]+)\b")
_RE_BASELINE_SWEEP = re.compile(r"\bbaseline_accuracy=([0-9]*\.?[0-9]+)\b")
_RE_SWEEP_HEADER = re.compile(r"\bsweep_layers=\d+\b")
_RE_SWEEP = re.compile(
	r"\blayer=(\d+)\s+op=([^\s]+)\s+pattern=([^\s]+)\s+value=([0-9eE\+\-\.]+)\s+"
	r"idx_trials=\d+\s+acc_min=([0-9]*\.?[0-9]+)\s+acc_max=([0-9]*\.?[0-9]+)\s+acc_mean=([0-9]*\.?[0-9]+)"
	r"(?:\s+acc_corr_min=([0-9]*\.?[0-9]+)\s+acc_corr_max=([0-9]*\.?[0-9]+)\s+acc_corr_mean=([0-9]*\.?[0-9]+))?\b"
)


def _parse_log(path: Path) -> ParsedLog:
	baseline_acc: Optional[float] = None
	rows: List[SweepRow] = []
	layer_order: List[int] = []
	layer_op: Dict[int, str] = {}

	lines = path.read_text(errors="replace").splitlines()
	for line in lines:
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
				acc_corr_max=float(m.group(9)) if m.group(9) is not None else None,
				acc_corr_mean=float(m.group(10)) if m.group(10) is not None else None,
			)
		)

	# Many sweep logs don't include the baseline "samples=... accuracy=..." line.
	# In that case, infer baseline as the maximum observed accuracy across all sweep rows.
	if baseline_acc is None and rows:
		baseline_acc = max(r.acc_max for r in rows)

	return ParsedLog(baseline_acc=baseline_acc, rows=rows, layer_order=layer_order, layer_op=layer_op)


def _unique_sorted(xs: Iterable) -> List:
	return sorted(set(xs))


def _format_value(v: float) -> str:
	# Keep filenames stable-ish: 1, 5, 10, 100, 1e+09
	if v == 0:
		return "0"
	av = abs(v)
	if 1e-3 <= av < 1e4:
		s = f"{v:g}"
		return s.replace(".", "p").replace("-", "m").replace("+", "")
	s = f"{v:.0e}"
	return s.replace("+", "").replace(".", "p").replace("-", "m")


def _plot_pattern(
	*,
	out_path: Path,
	title: str,
	baseline_fp32: float,
	baseline_int8: float,
	fp32: ParsedLog,
	int8: ParsedLog,
	pattern: str,
	metric: str,
):
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle

	def rows_to_map(rows: List[SweepRow]) -> Dict[Tuple[int, float], SweepRow]:
		return {(r.layer, r.value): r for r in rows if r.pattern == pattern}

	fp32_map = rows_to_map(fp32.rows)
	int8_map = rows_to_map(int8.rows)

	values = _unique_sorted([v for (_, v) in fp32_map.keys()] + [v for (_, v) in int8_map.keys()])

	# Match layers by *order of appearance in the logs*, not by numeric layer id.
	# This is useful when the same logical layers have different node_id numbering across models.
	n_layers = min(len(fp32.layer_order), len(int8.layer_order))
	if n_layers == 0 or not values:
		return False
	layer_pairs: List[Tuple[int, int]] = list(zip(fp32.layer_order[:n_layers], int8.layer_order[:n_layers]))

	# A compact palette that works for up to ~10 values.
	colors = [
		"#1f77b4",
		"#ff7f0e",
		"#2ca02c",
		"#d62728",
		"#9467bd",
		"#8c564b",
		"#e377c2",
		"#7f7f7f",
		"#bcbd22",
		"#17becf",
	]
	color_by_value = {v: colors[i % len(colors)] for i, v in enumerate(values)}

	# Layout: two rows (fp32, int8), shared x.
	fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(max(10, 0.6 * len(layer_pairs)), 7), sharex=True)
	fig.suptitle(title, fontsize=12)

	def draw(ax,
	         baseline: float,
	         rows_map: Dict[Tuple[int, float], SweepRow],
	         label: str,
	         which: str):
		ax.set_title(label, fontsize=10)
		ax.axhline(0.0, color="#444", linewidth=0.8)
		ax.set_ylabel("Accuracy drop (baseline - acc)")

		# For each layer we draw "box-like" ranges per value:
		# - rectangle spans [drop_min, drop_max]
		# - horizontal line at drop_mean
		# This is not a statistical boxplot (no quartiles), but conveys min/max/mean.
		nv = len(values)
		total_width = 0.8
		w = total_width / max(1, nv)

		for li, (layer_fp32, layer_int8) in enumerate(layer_pairs):
			layer = layer_fp32 if which == "fp32" else layer_int8
			x_center = li
			for vi, v in enumerate(values):
				row = rows_map.get((layer, v))
				if row is None:
					continue
				if metric == "raw":
					a_min, a_max, a_mean = row.acc_min, row.acc_max, row.acc_mean
				else:
					if row.acc_corr_min is None or row.acc_corr_max is None or row.acc_corr_mean is None:
						continue
					a_min, a_max, a_mean = row.acc_corr_min, row.acc_corr_max, row.acc_corr_mean

				drop_min = baseline - a_max
				drop_max = baseline - a_min
				drop_mean = baseline - a_mean

				x0 = x_center - total_width / 2 + vi * w
				rect = Rectangle(
					(x0, drop_min),
					w * 0.95,
					max(0.0, drop_max - drop_min),
					facecolor=color_by_value[v],
					alpha=0.35,
					edgecolor=color_by_value[v],
					linewidth=1.0,
				)
				ax.add_patch(rect)
				ax.plot([x0, x0 + w * 0.95], [drop_mean, drop_mean], color=color_by_value[v], linewidth=1.8)

		ax.set_xlim(-0.5, len(layer_pairs) - 0.5)
		ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

	draw(ax0, baseline_fp32, fp32_map, f"FP32 (baseline={baseline_fp32:.4f})", "fp32")
	draw(ax1, baseline_int8, int8_map, f"INT8 (baseline={baseline_int8:.4f})", "int8")

	ax1.set_xticks(list(range(len(layer_pairs))))
	xt = [f"{i}\n{fp}/{iq}" for i, (fp, iq) in enumerate(layer_pairs)]
	ax1.set_xticklabels(xt, rotation=0, fontsize=8)
	ax1.set_xlabel("Layer order (fp32_layer_id/int8_layer_id)")

	# Legend: fault values
	handles = []
	labels = []
	for v in values:
		handles.append(Rectangle((0, 0), 1, 1, facecolor=color_by_value[v], alpha=0.35, edgecolor=color_by_value[v]))
		labels.append(f"value={v:g}")
	ax0.legend(handles, labels, ncol=min(5, len(values)), fontsize=8, loc="upper right")

	fig.tight_layout(rect=[0, 0, 1, 0.95])
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=160)
	plt.close(fig)
	return True


def main() -> int:
	ap = argparse.ArgumentParser(description="Plot sweep logs (FP32 vs INT8) for KWS example.")
	ap.add_argument("--fp32-log", type=Path, required=True)
	ap.add_argument("--int8-log", type=Path, required=True)
	ap.add_argument("--out-dir", type=Path, default=Path("plots"))
	ap.add_argument("--baseline-fp32", type=float, default=None)
	ap.add_argument("--baseline-int8", type=float, default=None)
	ap.add_argument("--title", type=str, default="KWS sweep: accuracy drop by layer")
	args = ap.parse_args()

	fp32 = _parse_log(args.fp32_log)
	int8 = _parse_log(args.int8_log)

	baseline_fp32 = args.baseline_fp32 if args.baseline_fp32 is not None else fp32.baseline_acc
	baseline_int8 = args.baseline_int8 if args.baseline_int8 is not None else int8.baseline_acc
	if baseline_fp32 is None:
		raise SystemExit("No FP32 baseline found in log; pass --baseline-fp32")
	if baseline_int8 is None:
		raise SystemExit("No INT8 baseline found in log; pass --baseline-int8")

	patterns = _unique_sorted([r.pattern for r in fp32.rows] + [r.pattern for r in int8.rows])
	if not patterns:
		raise SystemExit("No sweep rows found in either log (expected lines with layer=... acc_min=... etc.)")

	written = 0
	for pat in patterns:
		for metric in ("raw", "undetected"):
			suffix = "" if metric == "raw" else "_undetected"
			out_path = args.out_dir / f"sweep_{pat}{suffix}.png"
			ok = _plot_pattern(
				out_path=out_path,
				title=f"{args.title} ({pat})" + ("" if metric == "raw" else " (undetected only)"),
				baseline_fp32=float(baseline_fp32),
				baseline_int8=float(baseline_int8),
				fp32=fp32,
				int8=int8,
				pattern=pat,
				metric=metric,
			)
			if ok:
				written += 1

	print(f"wrote {written} plot(s) to {args.out_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
