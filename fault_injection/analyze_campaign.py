#!/usr/bin/env python3
import argparse
import h5py
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


def _job_logs_from_h5(h5_path: Path) -> Dict[str, str]:
    logs: Dict[str, str] = {}
    with h5py.File(h5_path, "r") as f:
        if "jobs" not in f:
            raise SystemExit("Campaign file does not contain exact-sweep jobs. Regenerate the campaign with the updated generator.")
        for job_id, job_grp in f["jobs"].items():
            if str(job_grp.attrs.get("status", "pending")) != "completed":
                continue
            if "stdout" not in job_grp:
                continue
            log_name = str(job_grp.attrs["log_name"])
            raw = job_grp["stdout"][()]
            stdout = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            logs[log_name] = stdout
    return logs


def _completed_pairs(logs: Dict[str, str]) -> List[str]:
    mechs = set()
    for log_name in logs:
        if log_name.startswith("output_fp32_") and log_name.endswith(".log"):
            mech = log_name[len("output_fp32_") : -len(".log")]
            if f"output_int8_{mech}.log" in logs:
                mechs.add(mech)
    return sorted(mechs)


def _run_script(cmd: List[str], cwd: Path) -> None:
    env = dict(**os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze campaign results using the exact examples/aww plotting scripts.")
    ap.add_argument("--h5", type=Path, default=Path("campaign.h5"), help="Input HDF5 campaign file")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis"), help="Output directory for analysis")
    ap.add_argument("--examples-dir", type=Path, default=SCRIPT_DIR.parent / "examples" / "aww", help="Path to examples/aww")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"Campaign file not found: {args.h5}")

    examples_dir = args.examples_dir.resolve()
    plot_sweep = examples_dir / "plot_sweep_results.py"
    plot_worst = examples_dir / "plot_worst_case_drop.py"
    if not plot_sweep.exists() or not plot_worst.exists():
        raise SystemExit(f"Could not find plotting scripts in {examples_dir}")

    logs = _job_logs_from_h5(args.h5)
    mechs = _completed_pairs(logs)
    if not mechs:
        raise SystemExit("No completed fp32/int8 log pairs found in campaign.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fault_injection_logs_") as tmp:
        tmp_dir = Path(tmp)
        log_dir = tmp_dir / "logs"
        log_dir.mkdir()

        for log_name, stdout in logs.items():
            (log_dir / log_name).write_text(stdout)

        for mech in mechs:
            mech_out = out_dir / f"plots_{mech}"
            _run_script(
                [
                    "python3",
                    str(plot_sweep),
                    "--fp32-log",
                    str(log_dir / f"output_fp32_{mech}.log"),
                    "--int8-log",
                    str(log_dir / f"output_int8_{mech}.log"),
                    "--out-dir",
                    str(mech_out),
                ],
                cwd=examples_dir,
            )

        _run_script(
            [
                "python3",
                str(plot_worst),
                "--log-dir",
                str(log_dir),
                "--out-dir",
                str(out_dir / "plots_worst_case"),
            ],
            cwd=examples_dir,
        )

        summary_csv = out_dir / "worst_case_table.csv"
        src_csv = out_dir / "plots_worst_case" / "worst_case_table.csv"
        if src_csv.exists():
            shutil.copy2(src_csv, summary_csv)

    print(f"wrote outputs to {out_dir}")
    print(f"completed mechanism pairs: {len(mechs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
