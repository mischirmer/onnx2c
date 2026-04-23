#!/usr/bin/env python3
import argparse
import h5py
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

if __package__ in (None, ""):
    from sweep_parser import ParsedSweepOutput, parse_sweep_output
else:
    from .sweep_parser import ParsedSweepOutput, parse_sweep_output


def _resolve_maybe_relative(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_experiment(
    binary: Path,
    bin_dir: Path,
    label_csv: Path,
    limit: int,
    pattern: str,
    value: float,
    idx_trials: int,
) -> Tuple[int, List[str], str, str, ParsedSweepOutput]:
    cmd = [
        str(binary),
        "--bin-dir",
        str(bin_dir),
        "--label-csv",
        str(label_csv),
        "--limit",
        str(limit),
        "--sweep",
        "--sweep-patterns",
        pattern,
        "--sweep-values",
        str(value),
        "--sweep-indexes",
        str(idx_trials),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        return 2, cmd, stdout, stderr, parse_sweep_output(stdout)

    parsed = parse_sweep_output(result.stdout)
    return result.returncode, cmd, result.stdout, result.stderr, parsed


def _experiment_complete(exp_grp: h5py.Group) -> bool:
    if exp_grp.attrs.get("status", "pending") != "completed":
        return False
    if "stdout" not in exp_grp:
        return False
    if "layers" not in exp_grp:
        return False
    return True


def _collect_experiments(h5_path: Path, build_dir_override: Path) -> List[Dict]:
    experiments: List[Dict] = []
    with h5py.File(h5_path, "r") as f:
        build_dir_attr = f.attrs.get("build_dir")
        base_dir = h5_path.parent.resolve()
        build_dir = (
            _resolve_maybe_relative(str(build_dir_attr), base_dir)
            if build_dir_attr is not None
            else _resolve_maybe_relative(str(build_dir_override), base_dir)
        )
        bin_dir = _resolve_maybe_relative(str(f.attrs["bin_dir"]), base_dir)
        label_csv = _resolve_maybe_relative(str(f.attrs["label_csv"]), base_dir)
        limit = int(f.attrs["limit"])
        grp_experiments = f["experiments"]
        for mech_group_name in grp_experiments:
            grp_mech = grp_experiments[mech_group_name]
            for exp_id in grp_mech:
                if exp_id.startswith("."):
                    continue
                exp_grp = grp_mech[exp_id]
                if _experiment_complete(exp_grp):
                    continue
                experiments.append(
                    {
                        "exp_id": exp_id,
                        "mech_group_name": mech_group_name,
                        "binary": grp_mech.attrs["binary"],
                        "pattern": str(exp_grp.attrs["pattern"]),
                        "value": float(exp_grp.attrs["value"]),
                        "idx_trials": int(exp_grp.attrs["idx_trials"]),
                        "build_dir": build_dir,
                        "bin_dir": bin_dir,
                        "label_csv": label_csv,
                        "limit": limit,
                    }
                )
    return experiments


def _replace_text_dataset(group: h5py.Group, name: str, value: str) -> None:
    if name in group:
        del group[name]
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=dtype)


def _write_layer_results(exp_grp: h5py.Group, parsed: ParsedSweepOutput) -> None:
    if "layers" in exp_grp:
        del exp_grp["layers"]
    layers_grp = exp_grp.create_group("layers")
    for row in parsed.rows:
        layer_grp = layers_grp.create_group(f"layer_{row.layer}")
        layer_grp.attrs["layer"] = row.layer
        layer_grp.attrs["op"] = row.op
        layer_grp.attrs["pattern"] = row.pattern
        layer_grp.attrs["value"] = row.value
        layer_grp.attrs["acc_min"] = row.acc_min
        layer_grp.attrs["acc_max"] = row.acc_max
        layer_grp.attrs["acc_mean"] = row.acc_mean
        if row.acc_corr_min is not None:
            layer_grp.attrs["acc_corr_min"] = row.acc_corr_min
            layer_grp.attrs["acc_corr_max"] = row.acc_corr_max
            layer_grp.attrs["acc_corr_mean"] = row.acc_corr_mean
        if row.det_min is not None:
            layer_grp.attrs["det_min"] = row.det_min
            layer_grp.attrs["det_max"] = row.det_max
            layer_grp.attrs["det_mean"] = row.det_mean


def _save_result(h5_path: Path, exp_info: Dict, returncode: int, cmd: List[str], stdout: str, stderr: str, parsed: ParsedSweepOutput) -> None:
    status = "completed"
    if returncode == 2:
        status = "timeout"
    elif returncode != 0 or not parsed.rows or parsed.baseline_acc is None:
        status = "failed"

    with h5py.File(h5_path, "r+") as f:
        exp_grp = f["experiments"][exp_info["mech_group_name"]][exp_info["exp_id"]]
        exp_grp.attrs["status"] = status
        exp_grp.attrs["returncode"] = returncode
        exp_grp.attrs["cwd"] = os.getcwd()
        exp_grp.attrs["command"] = " ".join(cmd)

        _replace_text_dataset(exp_grp, "stdout", stdout)
        _replace_text_dataset(exp_grp, "stderr", stderr)

        if status != "completed":
            return

        exp_grp.attrs["baseline_accuracy"] = parsed.baseline_acc
        if parsed.baseline_acc_corrected is not None:
            exp_grp.attrs["baseline_accuracy_corrected"] = parsed.baseline_acc_corrected
        exp_grp.attrs["acc_min"] = min(row.acc_min for row in parsed.rows)
        exp_grp.attrs["acc_max"] = max(row.acc_max for row in parsed.rows)
        exp_grp.attrs["acc_mean"] = sum(row.acc_mean for row in parsed.rows) / len(parsed.rows)

        acc_corr_rows = [row for row in parsed.rows if row.acc_corr_min is not None]
        if acc_corr_rows:
            exp_grp.attrs["acc_corr_min"] = min(row.acc_corr_min for row in acc_corr_rows if row.acc_corr_min is not None)
            exp_grp.attrs["acc_corr_max"] = max(row.acc_corr_max for row in acc_corr_rows if row.acc_corr_max is not None)
            exp_grp.attrs["acc_corr_mean"] = sum(row.acc_corr_mean for row in acc_corr_rows if row.acc_corr_mean is not None) / len(acc_corr_rows)

        det_rows = [row for row in parsed.rows if row.det_min is not None]
        if det_rows:
            exp_grp.attrs["det_min"] = min(row.det_min for row in det_rows if row.det_min is not None)
            exp_grp.attrs["det_max"] = max(row.det_max for row in det_rows if row.det_max is not None)
            exp_grp.attrs["det_mean"] = sum(row.det_mean for row in det_rows if row.det_mean is not None) / len(det_rows)

        _write_layer_results(exp_grp, parsed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute fault injection campaign experiments.")
    ap.add_argument("--h5", type=Path, default=Path("campaign.h5"), help="Input HDF5 campaign file")
    ap.add_argument("--build-dir", type=Path, default=Path("../examples/aww/build"), help="Build directory with binaries")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"Campaign file not found: {args.h5}")

    experiments = _collect_experiments(args.h5, args.build_dir)
    if not experiments:
        print("No missing experiments.")
        return 0

    print(f"Found {len(experiments)} missing experiments")
    iterator = tqdm(experiments, desc="Running", unit="exp") if tqdm is not None else experiments

    completed = 0
    failed = 0
    for exp_info in iterator:
        binary = exp_info["build_dir"] / exp_info["binary"]
        returncode, cmd, stdout, stderr, parsed = _run_experiment(
            binary=binary,
            bin_dir=exp_info["bin_dir"],
            label_csv=exp_info["label_csv"],
            limit=exp_info["limit"],
            pattern=exp_info["pattern"],
            value=exp_info["value"],
            idx_trials=exp_info["idx_trials"],
        )
        _save_result(args.h5, exp_info, returncode, cmd, stdout, stderr, parsed)
        if returncode == 0 and parsed.rows and parsed.baseline_acc is not None:
            completed += 1
        else:
            failed += 1
            print(f"failed: {exp_info['exp_id']}", file=sys.stderr)

    print(f"Completed {completed}/{len(experiments)} experiments")
    if failed:
        print(f"Failed {failed}/{len(experiments)} experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
