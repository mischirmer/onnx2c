#!/usr/bin/env python3
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    det_min: Optional[float] = None
    det_max: Optional[float] = None
    det_mean: Optional[float] = None


@dataclass(frozen=True)
class ParsedSweepOutput:
    baseline_acc: Optional[float]
    baseline_acc_corrected: Optional[float]
    rows: List[SweepRow]
    layer_order: List[int]
    layer_op: Dict[int, str]


_RE_BASELINE = re.compile(r"\bsamples=(\d+)\s+correct=(\d+)\s+accuracy=([0-9]*\.?[0-9]+)\b")
_RE_BASELINE_SWEEP = re.compile(
    r"\bbaseline_accuracy=([0-9]*\.?[0-9]+)"
    r"(?:\s+baseline_accuracy_corrected=([0-9]*\.?[0-9]+))?\b"
)
_RE_SWEEP = re.compile(
    r"\blayer=(\d+)\s+op=([^\s]+)\s+pattern=([^\s]+)\s+value=([0-9eE\+\-\.]+)\s+"
    r"idx_trials=\d+\s+acc_min=([0-9]*\.?[0-9]+)\s+acc_max=([0-9]*\.?[0-9]+)\s+acc_mean=([0-9]*\.?[0-9]+)"
    r"(?:\s+acc_corr_min=([0-9]*\.?[0-9]+)\s+acc_corr_max=([0-9]*\.?[0-9]+)\s+acc_corr_mean=([0-9]*\.?[0-9]+))?\b"
    r"(?:\s+det_min=([0-9]*\.?[0-9]+)\s+det_max=([0-9]*\.?[0-9]+)\s+det_mean=([0-9]*\.?[0-9]+))?"
)


def parse_sweep_output(output: str) -> ParsedSweepOutput:
    baseline_acc: Optional[float] = None
    baseline_acc_corrected: Optional[float] = None
    rows: List[SweepRow] = []
    layer_order: List[int] = []
    layer_op: Dict[int, str] = {}

    for line in output.splitlines():
        if baseline_acc is None:
            mh = _RE_BASELINE_SWEEP.search(line)
            if mh:
                baseline_acc = float(mh.group(1))
                baseline_acc_corrected = float(mh.group(2)) if mh.group(2) is not None else None
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
                det_min=float(m.group(11)) if m.group(11) is not None else None,
                det_max=float(m.group(12)) if m.group(12) is not None else None,
                det_mean=float(m.group(13)) if m.group(13) is not None else None,
            )
        )

    if baseline_acc is None and rows:
        baseline_acc = max(r.acc_max for r in rows)

    return ParsedSweepOutput(
        baseline_acc=baseline_acc,
        baseline_acc_corrected=baseline_acc_corrected,
        rows=rows,
        layer_order=layer_order,
        layer_op=layer_op,
    )
