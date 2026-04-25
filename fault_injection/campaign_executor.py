#!/usr/bin/env python3
import argparse
import h5py
import os
import re
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

if __package__ in (None, ""):
    from sweep_parser import parse_sweep_output
else:
    from .sweep_parser import parse_sweep_output

_RE_PROGRESS = re.compile(r"progress\s+\d+%\s+\((\d+)/(\d+)\)")


def _resolve_maybe_relative(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _replace_text_dataset(group: h5py.Group, name: str, value: str) -> None:
    if name in group:
        del group[name]
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=dtype)


def _job_complete(job_grp: h5py.Group) -> bool:
    return (
        job_grp.attrs.get("status", "pending") == "completed"
        and "stdout" in job_grp
        and "layers" in job_grp
    )


def _collect_jobs(h5_path: Path, build_dir_override: Path) -> List[Dict]:
    jobs: List[Dict] = []
    with h5py.File(h5_path, "r") as f:
        base_dir = h5_path.parent.resolve()
        build_dir_attr = f.attrs.get("build_dir")
        build_dir = (
            _resolve_maybe_relative(str(build_dir_attr), base_dir)
            if build_dir_attr is not None
            else _resolve_maybe_relative(str(build_dir_override), base_dir)
        )
        bin_dir = _resolve_maybe_relative(str(f.attrs["bin_dir"]), base_dir)
        label_csv = _resolve_maybe_relative(str(f.attrs["label_csv"]), base_dir)
        limit = int(f.attrs["limit"])
        for job_id, job_grp in f["jobs"].items():
            if _job_complete(job_grp):
                continue
            jobs.append(
                {
                    "job_id": job_id,
                    "mechanism": str(job_grp.attrs["mechanism"]),
                    "quant": str(job_grp.attrs["quant"]),
                    "binary": str(job_grp.attrs["binary"]),
                    "build_dir": build_dir,
                    "bin_dir": bin_dir,
                    "label_csv": label_csv,
                    "limit": limit,
                }
            )
    return jobs


def _run_job(
    binary: Path,
    bin_dir: Path,
    label_csv: Path,
    limit: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, List[str], str, str]:
    cmd = [
        str(binary),
        "--bin-dir",
        str(bin_dir),
        "--label-csv",
        str(label_csv),
        "--limit",
        str(limit),
        "--sweep",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        sel = selectors.DefaultSelector()
        assert proc.stdout is not None
        assert proc.stderr is not None
        sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        stderr_text = ""
        last_progress: Tuple[int, int] = (0, 0)

        while sel.get_map():
            for key, _ in sel.select(timeout=0.2):
                chunk = key.fileobj.read1(65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    stdout_chunks.append(chunk)
                    continue

                stderr_chunks.append(chunk)
                stderr_text += chunk.decode("utf-8", errors="replace")
                matches = list(_RE_PROGRESS.finditer(stderr_text))
                if matches:
                    m = matches[-1]
                    progress = (int(m.group(1)), int(m.group(2)))
                    if progress != last_progress and progress_cb is not None:
                        progress_cb(*progress)
                    last_progress = progress
                if len(stderr_text) > 4096:
                    stderr_text = stderr_text[-4096:]

            if proc.poll() is not None and not sel.get_map():
                break

        returncode = proc.wait(timeout=7200)
        if progress_cb is not None and last_progress[1] > 0:
            progress_cb(last_progress[1], last_progress[1])
        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        return returncode, cmd, stdout, stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 2, cmd, stdout, stderr


def _write_layers(job_grp: h5py.Group, stdout: str) -> bool:
    parsed = parse_sweep_output(stdout)
    if parsed.baseline_acc is None or not parsed.rows:
        return False

    if "layers" in job_grp:
        del job_grp["layers"]
    layers_grp = job_grp.create_group("layers")
    for row in parsed.rows:
        row_id = f"{row.pattern}_{row.value:g}_layer_{row.layer}"
        layer_grp = layers_grp.create_group(row_id)
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

    job_grp.attrs["baseline_accuracy"] = parsed.baseline_acc
    if parsed.baseline_acc_corrected is not None:
        job_grp.attrs["baseline_accuracy_corrected"] = parsed.baseline_acc_corrected
    job_grp.attrs["row_count"] = len(parsed.rows)
    return True


def _save_job_result(h5_path: Path, job: Dict, returncode: int, cmd: List[str], stdout: str, stderr: str) -> str:
    with h5py.File(h5_path, "r+") as f:
        job_grp = f["jobs"][job["job_id"]]
        job_grp.attrs["returncode"] = returncode
        job_grp.attrs["cwd"] = os.getcwd()
        job_grp.attrs["command"] = " ".join(cmd)
        _replace_text_dataset(job_grp, "stdout", stdout)
        _replace_text_dataset(job_grp, "stderr", stderr)

        if returncode == 2:
            job_grp.attrs["status"] = "timeout"
            return "timeout"
        if returncode != 0:
            job_grp.attrs["status"] = "failed"
            return "failed"

        ok = _write_layers(job_grp, stdout)
        job_grp.attrs["status"] = "completed" if ok else "failed"
        return "completed" if ok else "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute a resumable sweep campaign matching examples/aww/run_sweeps.sh.")
    ap.add_argument("--h5", type=Path, default=Path("campaign.h5"), help="Input HDF5 campaign file")
    ap.add_argument("--build-dir", type=Path, default=Path("../examples/aww/build"), help="Build directory with binaries")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"Campaign file not found: {args.h5}")

    jobs = _collect_jobs(args.h5, args.build_dir)
    if not jobs:
        print("No missing jobs.")
        return 0

    print(f"Found {len(jobs)} missing jobs")
    iterator = jobs

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=0, desc="Sweep runs", unit="run")

    completed = 0
    failed = 0
    for i, job in enumerate(iterator, start=1):
        binary = job["build_dir"] / job["binary"]

        def _progress_cb(done: int, total: int) -> None:
            if pbar is None:
                return
            if pbar.total != total:
                pbar.total = total
            pbar.n = done
            pbar.set_description(f"{job['job_id']} [{i}/{len(jobs)}]")
            pbar.refresh()

        if pbar is None:
            print(f"running {job['job_id']} [{i}/{len(jobs)}]", file=sys.stderr)

        returncode, cmd, stdout, stderr = _run_job(
            binary=binary,
            bin_dir=job["bin_dir"],
            label_csv=job["label_csv"],
            limit=job["limit"],
            progress_cb=_progress_cb,
        )
        if pbar is not None:
            pbar.reset(total=0)
            pbar.set_description("Sweep runs")

        status = _save_job_result(args.h5, job, returncode, cmd, stdout, stderr)
        if status == "completed":
            completed += 1
        else:
            failed += 1
            print(f"failed: {job['job_id']}", file=sys.stderr)

    if pbar is not None:
        pbar.close()

    print(f"Completed {completed}/{len(jobs)} jobs")
    if failed:
        print(f"Failed {failed}/{len(jobs)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
