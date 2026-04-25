#!/usr/bin/env python3
import argparse
import h5py
from pathlib import Path
from typing import List, Optional, Set, Tuple


def _discover_binaries(build_dir: Path, prefix: str) -> List[str]:
    mechs = []
    for p in build_dir.glob(f"{prefix}*"):
        if p.is_file():
            mech = p.name[len(prefix) :]
            mechs.append(mech)
    return sorted(set(mechs))


def _default_job_order() -> List[str]:
    return [
        "abyzft",
        "abft",
        "freivalds1x",
        "freivalds2x",
        "freivalds3x",
        "freivalds4x",
        "gvfa1x",
        "gvfa2x",
    ]


def _build_jobs(build_dir: Path, quant: str, wanted: Set[str]) -> List[Tuple[str, str, str]]:
    jobs: List[Tuple[str, str, str]] = []
    quants = ["fp32", "int8"] if quant == "both" else [quant]
    for mech in _default_job_order():
        if mech not in wanted:
            continue
        for q in quants:
            binary = f"aww_{q}_{mech}"
            if not (build_dir / binary).exists():
                continue
            log_name = f"output_{q}_{mech}.log"
            jobs.append((mech, q, log_name))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a resumable campaign matching examples/aww/run_sweeps.sh.")
    ap.add_argument("--h5", type=Path, default=Path("campaign.h5"), help="Output HDF5 file")
    ap.add_argument("--build-dir", type=Path, default=Path("../examples/aww/build"), help="Build directory with binaries")
    ap.add_argument("--bin-dir", type=Path, required=True, help="Directory with binary input data")
    ap.add_argument("--label-csv", type=Path, required=True, help="Labels CSV file")
    ap.add_argument("--quant", choices=["fp32", "int8", "both"], default="both")
    ap.add_argument("--mechanisms", type=str, default="", help="Comma-separated list of mechanisms")
    ap.add_argument("--limit", type=int, default=1000, help="Number of inference samples to process")
    args = ap.parse_args()

    build_dir = args.build_dir.resolve()
    bin_dir = args.bin_dir.resolve()
    label_csv = args.label_csv.resolve()
    if not build_dir.exists():
        raise SystemExit(f"Build dir not found: {build_dir}")

    available = set(_discover_binaries(build_dir, "aww_fp32_")) | set(_discover_binaries(build_dir, "aww_int8_"))
    if not available:
        raise SystemExit(f"No sweep binaries found in {build_dir}")

    if args.mechanisms.strip():
        wanted = {x.strip() for x in args.mechanisms.split(",") if x.strip()}
        invalid = wanted - available
        if invalid:
            raise SystemExit(f"Unknown mechanisms: {sorted(invalid)}. Available: {sorted(available)}")
    else:
        wanted = set(_default_job_order()) & available

    jobs = _build_jobs(build_dir, args.quant, wanted)
    if not jobs:
        raise SystemExit("No jobs selected.")

    h5_path = args.h5
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "a") as f:
        f.attrs["version"] = "3.0"
        f.attrs["build_dir"] = str(build_dir)
        f.attrs["bin_dir"] = str(bin_dir)
        f.attrs["label_csv"] = str(label_csv)
        f.attrs["limit"] = args.limit
        f.attrs["job_mode"] = "run_sweeps_exact"

        jobs_grp = f.require_group("jobs")
        for mech, quant, log_name in jobs:
            job_id = f"{quant}_{mech}"
            job_grp = jobs_grp.require_group(job_id)
            job_grp.attrs["mechanism"] = mech
            job_grp.attrs["quant"] = quant
            job_grp.attrs["binary"] = f"aww_{quant}_{mech}"
            job_grp.attrs["log_name"] = log_name
            if "status" not in job_grp.attrs:
                job_grp.attrs["status"] = "pending"

    print(f"Generated campaign: {h5_path}")
    print(f"  jobs: {[f'{q}_{m}' for m, q, _ in jobs]}")
    print(f"  limit: {args.limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
