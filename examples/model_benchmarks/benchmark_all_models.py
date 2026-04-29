#!/usr/bin/env python3
"""
Simple harness to compile ONNX models with onnx2c for multiple verification
mechanisms and run a small micro-benchmark that repeatedly calls the generated
`entry(...)` function. Designed for quick comparisons without needing real
datasets or labels.

Usage:
  python3 benchmark_all_models.py --models-dir . --out-csv build/all_model_runtimes.csv
  python3 plot_model_runtime_overheads.py --results-csv build/all_model_runtimes.csv --out-dir build/runtime_plots

Notes:
 - Requires a built `onnx2c` at the path given by `--onnx2c` (default ../../build/onnx2c).
 - Uses `g++/gcc` from PATH.
 - Only processes .onnx files. TFLite files are ignored.
 - For each model, generates per-mechanism C via onnx2c, a tiny runner that
   allocates zeroed input/output arrays matching the `entry(...)` signature and
   times repeated calls to `entry`.
"""

import argparse
import csv
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict


MECH_TO_FLAGS = {
    "baseline": "--conv-im2col --log=1",
    "baseline_raw": "--log=1",
    "abft": "--conv-im2col --abft-gemm",
    "abyzft": "--conv-im2col --abyzft-gemm",
    "freivalds1x": "--conv-im2col --freivalds-gemm --freivalds-checks 1",
    "freivalds2x": "--conv-im2col --freivalds-gemm --freivalds-checks 2",
    "freivalds3x": "--conv-im2col --freivalds-gemm --freivalds-checks 3",
    "freivalds4x": "--conv-im2col --freivalds-gemm --freivalds-checks 4",
    "gvfa1x": "--conv-im2col --gvfa-gemm --gvfa-checks 1",
    "gvfa2x": "--conv-im2col --gvfa-gemm --gvfa-checks 2",
}


def flags_for_mechanism(mech: str) -> str:
    if mech in MECH_TO_FLAGS:
        return MECH_TO_FLAGS[mech]

    m = re.fullmatch(r"freivalds([1-9][0-9]*)x", mech)
    if m:
        return f"--conv-im2col --freivalds-gemm --freivalds-checks {m.group(1)}"

    raise KeyError(mech)


def run(cmd: List[str], **kwargs):
    print("+", shlex.join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def extract_entry_signature(c_file: Path) -> str:
    txt = c_file.read_text(errors="ignore")
    m = re.search(r"void\s+entry\s*\((.*?)\)\s*\{", txt, flags=re.S)
    if not m:
        raise RuntimeError(f"failed to find entry(...) signature in {c_file}")
    params = m.group(1).strip()
    # normalize whitespace
    params = re.sub(r"\s+", " ", params)
    return params


RUNNER_TEMPLATE = r"""
#include <chrono>
#include <iostream>
#include <string>
#include <cstring>

extern "C" void entry({params});

int main(int argc, char** argv) {{
    int warmup = {warmup};
    int repeats = {repeats};
    for (int i = 1; i < argc; ++i) {{
        if (std::string(argv[i]) == "--warmup" && i+1 < argc) warmup = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "--repeats" && i+1 < argc) repeats = std::stoi(argv[++i]);
    }}

    // allocate zeroed arrays for each parameter
{decls}

    // warmup
    for (int i = 0; i < warmup; ++i) entry({call_args});

    std::vector<double> times;
    times.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {{
        auto t0 = std::chrono::high_resolution_clock::now();
        entry({call_args});
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        times.push_back(dt.count());
    }}

    // print per-run times as CSV on stdout
    for (size_t i = 0; i < times.size(); ++i) std::cout << times[i] << (i+1<times.size()?",":"\n");
    return 0;
}}
"""


def make_decls_and_args(params: str) -> (str, str):
    # params example: 'const float tensor_input_1[1][49][10][1], float tensor_Identity[1][12]'
    if not params:
        return "", ""
    parts = [p.strip() for p in params.split(",")]
    decls = []
    call_args = []
    for p in parts:
        # split by last space to separate type and name
        # handle cases like 'const float name[... ]'
        m = re.match(r"(.*\S)\s+(\w+\s*\[.*\])", p)
        if m:
            typ = m.group(1)
            name_and_arr = m.group(2)
            # convert to non-const for allocation
            typ_base = typ.replace("const ", "")
            decls.append(f"    static {typ_base} {name_and_arr} = {{0}};")
            # pass the variable name (strip array brackets)
            vn = re.match(r"(\w+)", name_and_arr)
            varname = vn.group(1) if vn else name_and_arr.split("=")[0].strip()
            call_args.append(varname)
            continue
        # fallback: split by last space
        toks = p.rsplit(" ", 1)
        if len(toks) == 2:
            typ, name = toks
            decls.append(f"    static {typ} {name} = {{0}};")
            vn = re.match(r"(\w+)$", name)
            varname = vn.group(1) if vn else name
            call_args.append(varname)
        else:
            # unexpected; create a void* placeholder
            name = toks[0]
            vn = re.match(r"(\w+)$", name)
            varname = vn.group(1) if vn else name
            decls.append(f"    void* {varname} = nullptr;")
            call_args.append(varname)
    return "\n".join(decls), ", ".join(call_args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=Path("build/model_runtimes.csv"))
    ap.add_argument("--onnx2c", type=Path, default=Path("../../build/onnx2c"))
    ap.add_argument("--build-root", type=Path, default=Path("build"))
    ap.add_argument("--mechanisms", type=str, default=",".join(MECH_TO_FLAGS.keys()))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    models_dir: Path = args.models_dir
    onnx2c = args.onnx2c
    build_root: Path = args.build_root
    mechs = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    if not onnx2c.exists():
        print(f"onnx2c not found at {onnx2c}", file=sys.stderr)
        return 2

    build_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str,str]] = []
    failed: List[str] = []  # track (model, mech, reason)

    for onnx_path in sorted(models_dir.glob("*.onnx")):
        model_stem = onnx_path.stem
        model_dir = build_root / model_stem
        model_dir.mkdir(parents=True, exist_ok=True)

        for mech in mechs:
            try:
                flags = flags_for_mechanism(mech)
            except KeyError:
                print(f"unknown mech {mech}; skipping", file=sys.stderr)
                continue
            c_out = model_dir / f"{model_stem}_{mech}.c"
            obj_out = model_dir / f"{model_stem}_{mech}.o"
            exe = model_dir / f"{model_stem}_{mech}"

            try:
                # generate C
                cmd = [str(onnx2c)] + shlex.split(flags) + [str(onnx_path)]
                with c_out.open("wb") as f:
                    subprocess.run(cmd, stdout=f, check=True)

                # compile C to object
                run(["gcc", "-O2", "-std=c99", "-c", str(c_out), "-o", str(obj_out)])

                # extract entry signature
                params = extract_entry_signature(c_out)
                decls, call_args = make_decls_and_args(params)

                runner_cpp = model_dir / f"runner_{mech}.cpp"
                runner_code = RUNNER_TEMPLATE.format(params=params, decls=decls, call_args=call_args, warmup=args.warmup, repeats=args.repeats)
                runner_cpp.write_text(runner_code)

                # compile runner and link
                run(["g++", "-O2", "-std=c++17", str(runner_cpp), str(obj_out), "-o", str(exe)])

                # run the exe and parse per-run times (CSV on stdout)
                proc = subprocess.run([str(exe), "--warmup", str(args.warmup), "--repeats", str(args.repeats)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                out = proc.stdout.strip()
                times = [float(x) for x in out.split(",") if x.strip()]
                mean_s = statistics.fmean(times) if times else float('nan')
                med_s = statistics.median(times) if times else float('nan')
                std_s = statistics.stdev(times) if len(times) > 1 else 0.0

                records.append({
                    "model": model_stem,
                    "mechanism": mech,
                    "flags": flags,
                    "mean_s": f"{mean_s:.8f}",
                    "median_s": f"{med_s:.8f}",
                    "std_s": f"{std_s:.8f}",
                    "warmup": str(args.warmup),
                    "repeats": str(args.repeats),
                })
            except subprocess.CalledProcessError as e:
                reason = f"compile/link error (exit {e.returncode})"
                failed.append(f"{model_stem}/{mech}: {reason}")
                print(f"SKIP {model_stem}/{mech}: {reason}", file=sys.stderr)
            except Exception as e:
                reason = str(e)
                failed.append(f"{model_stem}/{mech}: {reason}")
                print(f"SKIP {model_stem}/{mech}: {reason}", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","mechanism","flags","mean_s","median_s","std_s","warmup","repeats"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"wrote CSV: {args.out_csv}")
    print(f"Success: {len(records)} model/mechanism combinations")
    if failed:
        print(f"Failed: {len(failed)} model/mechanism combinations")
        for f in failed[:10]:
            print(f"  {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
