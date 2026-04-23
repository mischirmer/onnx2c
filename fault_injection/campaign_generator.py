#!/usr/bin/env python3
import argparse
import h5py
from pathlib import Path
from typing import List, Optional, Set


def _discover_binaries(build_dir: Path, prefix: str) -> List[str]:
    mechs = []
    for p in build_dir.glob(f"{prefix}*"):
        if p.is_file():
            mech = p.name
            if mech.startswith(prefix):
                mech = mech[len(prefix):]
            mechs.append(mech)
    return sorted(set(mechs))


def _default_fault_values() -> List[float]:
    return [0.0, 5.0, 10.0, 100.0, 1000.0]


def _default_patterns() -> List[str]:
    return ["single_point", "trivial", "checkered"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate fault injection campaign experiments to HDF5.")
    ap.add_argument("--h5", type=Path, default=Path("campaign.h5"), help="Output HDF5 file")
    ap.add_argument("--build-dir", type=Path, default=Path("../examples/aww/build"), help="Build directory with binaries")
    ap.add_argument("--bin-dir", type=Path, required=True, help="Directory with binary input data")
    ap.add_argument("--label-csv", type=Path, required=True, help="Labels CSV file")
    ap.add_argument("--quant", choices=["fp32", "int8", "both"], default="both")
    ap.add_argument("--mechanisms", type=str, default="", help="Comma-separated list of mechanisms")
    ap.add_argument("--patterns", type=str, default="", help="Comma-separated fault patterns (default: single_point,trivial,checkered)")
    ap.add_argument("--values", type=str, default="", help="Comma-separated fault values")
    ap.add_argument("--idx-trials", type=int, default=25, help="Number of index trials per experiment (default: 25)")
    ap.add_argument("--limit", type=int, default=100, help="Number of inference samples to process")
    args = ap.parse_args()

    build_dir = args.build_dir.resolve()
    if not build_dir.exists():
        raise SystemExit(f"Build dir not found: {build_dir}")
    bin_dir = args.bin_dir.resolve()
    label_csv = args.label_csv.resolve()

    fp32_prefix = "aww_fp32_"
    int8_prefix = "aww_int8_"

    discovered_fp32 = set(_discover_binaries(build_dir, fp32_prefix))
    discovered_int8 = set(_discover_binaries(build_dir, int8_prefix))

    available_mechs = discovered_fp32 | discovered_int8
    if not available_mechs:
        raise SystemExit(f"No binaries found in {build_dir}")

    wanted: Optional[Set[str]] = None
    if args.mechanisms.strip():
        wanted = {x.strip() for x in args.mechanisms.split(",") if x.strip()}
        invalid = wanted - available_mechs
        if invalid:
            raise SystemExit(f"Unknown mechanisms: {invalid}. Available: {available_mechs}")
    else:
        wanted = available_mechs

    patterns = _default_patterns()
    if args.patterns.strip():
        patterns = [x.strip() for x in args.patterns.split(",") if x.strip()]

    fault_values: List[float] = [0.0]
    if args.values.strip():
        fault_values = [float(x.strip()) for x in args.values.split(",") if x.strip()]
    elif not args.values.strip():
        fault_values = _default_fault_values()

    h5_path = args.h5
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "a") as f:
        f.attrs["version"] = "2.0"
        f.attrs["build_dir"] = str(build_dir)
        f.attrs["bin_dir"] = str(bin_dir)
        f.attrs["label_csv"] = str(label_csv)
        f.attrs["limit"] = args.limit
        f.attrs["idx_trials"] = args.idx_trials
        f.attrs["patterns"] = ",".join(patterns)
        f.attrs["values"] = ",".join(str(v) for v in fault_values)

        grp_experiments = f.require_group("experiments")

        quants = ["fp32", "int8"] if args.quant == "both" else [args.quant]

        for mech in sorted(wanted):
            for q in quants:
                prefix = fp32_prefix if q == "fp32" else int8_prefix
                binary = f"{prefix}{mech}"
                if not (build_dir / binary).exists():
                    continue

                grp_mech = grp_experiments.require_group(f"{mech}_{q}")
                grp_mech.attrs["mechanism"] = mech
                grp_mech.attrs["quant"] = q
                grp_mech.attrs["binary"] = binary

                for pattern in patterns:
                    for value in fault_values:
                        exp_id = f"{mech}_{q}_{pattern}_{value}"
                        exp_grp = grp_mech.require_group(exp_id)
                        exp_grp.attrs["mechanism"] = mech
                        exp_grp.attrs["quant"] = q
                        exp_grp.attrs["pattern"] = pattern
                        exp_grp.attrs["value"] = value
                        exp_grp.attrs["idx_trials"] = args.idx_trials
                        if "status" not in exp_grp.attrs:
                            exp_grp.attrs["status"] = "pending"

    print(f"Generated campaign: {h5_path}")
    print(f"  mechanisms: {sorted(wanted)}")
    print(f"  quant: {args.quant}")
    print(f"  patterns: {patterns}")
    print(f"  fault_values: {fault_values}")
    print(f"  idx_trials: {args.idx_trials}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
