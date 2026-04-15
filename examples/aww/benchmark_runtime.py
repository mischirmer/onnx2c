#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_mech(bin_name: str, prefix: str) -> str:
    # run_fp32_abft -> abft
    return bin_name[len(prefix) :]


def _sort_key_mech(m: str) -> Tuple[int, int, str]:
    if m == "baseline":
        return (0, 0, m)
    if m == "abft":
        return (1, 0, m)
    if m == "abyzft":
        return (2, 0, m)
    if m.startswith("freivalds") and m.endswith("x"):
        n = m[len("freivalds") : -1]
        if n.isdigit():
            return (3, int(n), m)
    if m.startswith("gvfa") and m.endswith("x"):
        n = m[len("gvfa") : -1]
        if n.isdigit():
            return (4, int(n), m)
    return (9, 0, m)


def _discover_mechanisms(build_dir: Path, quant: str) -> List[str]:
    if quant == "fp32":
        pref = "run_fp32_"
        mechs = [_parse_mech(p.name, pref) for p in build_dir.glob("run_fp32_*") if p.is_file()]
        return sorted(set(mechs), key=_sort_key_mech)
    if quant == "int8":
        pref = "run_int8_"
        mechs = [_parse_mech(p.name, pref) for p in build_dir.glob("run_int8_*") if p.is_file()]
        return sorted(set(mechs), key=_sort_key_mech)

    # both: intersection to ensure fair comparison set
    fp32 = set(_parse_mech(p.name, "run_fp32_") for p in build_dir.glob("run_fp32_*") if p.is_file())
    int8 = set(_parse_mech(p.name, "run_int8_") for p in build_dir.glob("run_int8_*") if p.is_file())
    return sorted(fp32 & int8, key=_sort_key_mech)


def _run_once(cmd: List[str]) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    t1 = time.perf_counter()
    return t1 - t0


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _print_progress(done: int, total: int, *, label: str) -> None:
    if total <= 0:
        return
    pct = (100.0 * done) / float(total)
    print(f"\r[{label}] progress {pct:6.2f}% ({done}/{total})", end="", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark runtime of generated binaries without fault injection and report overhead vs baseline."
        )
    )
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--bin-dir", type=Path, required=True)
    ap.add_argument("--label-csv", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--quant", choices=["fp32", "int8", "both"], default="both")
    ap.add_argument("--mechanisms", type=str, default="", help="CSV list, e.g. baseline,abft,abyzft,freivalds1x")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    build_dir = args.build_dir
    if not build_dir.exists():
        raise SystemExit(f"build dir not found: {build_dir}")

    wanted = None
    if args.mechanisms.strip():
        wanted = {x.strip() for x in args.mechanisms.split(",") if x.strip()}

    if args.quant == "both":
        quants = ["fp32", "int8"]
    else:
        quants = [args.quant]

    records: List[Dict[str, str]] = []

    for q in quants:
        mechs = _discover_mechanisms(build_dir, q)
        if wanted is not None:
            mechs = [m for m in mechs if m in wanted]
        if not mechs:
            print(f"[{q}] no binaries found for selected mechanisms", file=sys.stderr)
            continue
        if "baseline" not in mechs:
            print(f"[{q}] warning: baseline not found; overheads will be N/A", file=sys.stderr)

        prefix = "run_fp32_" if q == "fp32" else "run_int8_"
        existing_mechs = [m for m in mechs if (build_dir / f"{prefix}{m}").exists()]
        total_runs = len(existing_mechs) * (args.warmup + args.repeats)
        done_runs = 0

        stats: Dict[str, Dict[str, float]] = {}

        print(f"\n[{q}] benchmarking mechanisms: {', '.join(mechs)}")
        for mech in mechs:
            exe = build_dir / f"{prefix}{mech}"
            if not exe.exists():
                print(f"  skip {mech}: missing {exe}", file=sys.stderr)
                continue

            cmd = [
                str(exe),
                "--bin-dir",
                str(args.bin_dir),
                "--label-csv",
                str(args.label_csv),
                "--limit",
                str(args.limit),
            ]

            for _ in range(args.warmup):
                _run_once(cmd)
                done_runs += 1
                _print_progress(done_runs, total_runs, label=q)

            times: List[float] = []
            for _ in range(args.repeats):
                times.append(_run_once(cmd))
                done_runs += 1
                _print_progress(done_runs, total_runs, label=q)

            mean_s = statistics.fmean(times)
            med_s = statistics.median(times)
            std_s = statistics.stdev(times) if len(times) > 1 else 0.0

            stats[mech] = {
                "mean_s": mean_s,
                "median_s": med_s,
                "std_s": std_s,
            }

        if total_runs > 0:
            print(file=sys.stderr)

        if not stats:
            print(f"[{q}] no successful runs", file=sys.stderr)
            continue

        baseline_med = stats["baseline"]["median_s"] if "baseline" in stats else math.nan

        print("mechanism           mean[s]    median[s]   std[s]     overhead_vs_baseline")
        print("------------------  ---------  ----------  ---------  --------------------")
        for mech in sorted(stats.keys(), key=_sort_key_mech):
            s = stats[mech]
            if math.isnan(baseline_med) or baseline_med <= 0:
                ov_txt = "N/A"
                ov = math.nan
            else:
                ov = (s["median_s"] / baseline_med) - 1.0
                ov_txt = _fmt_pct(ov)
            print(
                f"{mech:<18}  {s['mean_s']:>9.4f}  {s['median_s']:>10.4f}  {s['std_s']:>9.4f}  {ov_txt:>20}"
            )

            records.append(
                {
                    "quant": q,
                    "mechanism": mech,
                    "mean_s": f"{s['mean_s']:.8f}",
                    "median_s": f"{s['median_s']:.8f}",
                    "std_s": f"{s['std_s']:.8f}",
                    "overhead_vs_baseline": "" if math.isnan(ov) else f"{ov:.8f}",
                    "limit": str(args.limit),
                    "repeats": str(args.repeats),
                    "warmup": str(args.warmup),
                }
            )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "quant",
                    "mechanism",
                    "mean_s",
                    "median_s",
                    "std_s",
                    "overhead_vs_baseline",
                    "limit",
                    "repeats",
                    "warmup",
                ],
            )
            w.writeheader()
            for r in records:
                w.writerow(r)
        print(f"\nwrote CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
