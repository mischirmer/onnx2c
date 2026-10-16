"""Command line entry point for the operator coverage campaign."""

import argparse
import datetime
import fnmatch
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time

import discovery
import report as report_module
import runner

DEFAULT_ACCURACY = 2e-05
DEFAULT_CC_FLAGS = runner.DEFAULT_CC_FLAGS


def _repo_root(start):
    current = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(current, "src", "graph.cc")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(start)
        current = parent


def _find_binary(repo_root, explicit, kind):
    if explicit:
        return explicit
    for build in sorted(os.listdir(repo_root)):
        if not build.startswith("build"):
            continue
        base = os.path.join(repo_root, build)
        if kind == "onnx2c":
            candidates = [os.path.join(base, "onnx2c")]
        else:
            candidates = [os.path.join(base, "test", kind),
                          os.path.join(base, kind)]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return shutil.which(kind)


def _git(repo_root, *args):
    try:
        out = subprocess.run(["git", "-C", repo_root] + list(args),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=False)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace").strip()


def _git_commit(repo_root):
    return _git(repo_root, "rev-parse", "HEAD") or "unknown"


def _git_dirty(repo_root):
    status = _git(repo_root, "status", "--porcelain")
    if not status:
        return False
    return any(not line.startswith("??") for line in status.splitlines())


def _compiler_info(cc):
    try:
        out = subprocess.run([cc, "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return cc, "unknown"
    text = out.stdout.decode("utf-8", "replace").strip().splitlines()
    return cc, (text[0] if text else "unknown")


def _registration_summary(registrations, cases, missing_cases):
    """Summarise how upstream wires tests into CTest vs the corpus."""
    active = [r for r in registrations.values()
              if r["status"] == discovery.STATUS_ACTIVE]
    return {
        "total": len(registrations),
        "active": len(active),
        "disabled": sum(1 for r in registrations.values()
                        if r["status"] == discovery.STATUS_DISABLED),
        "active_singlefile": sum(1 for r in active
                                 if r["mode"] == discovery.MODE_SINGLEFILE),
        "active_split": sum(1 for r in active
                            if r["mode"] == discovery.MODE_SPLIT),
        "active_node": sum(1 for r in active if r["family"] == "node"),
        "active_pytorch_converted": sum(
            1 for r in active if r["family"] == "pytorch-converted"),
        "active_local_ops": sum(1 for r in active
                                if r["family"] == "local_ops"),
        "missing_from_corpus": list(missing_cases),
        "cases_active": sum(1 for c in cases
                            if c["upstream"]["status"] ==
                            discovery.STATUS_ACTIVE),
        "cases_disabled": sum(1 for c in cases
                              if c["upstream"]["status"] ==
                              discovery.STATUS_DISABLED),
        "cases_absent": sum(1 for c in cases
                            if c["upstream"]["status"] ==
                            discovery.STATUS_ABSENT),
        "cases_singlefile": sum(1 for c in cases
                                if c["upstream"]["mode"] ==
                                discovery.MODE_SINGLEFILE),
    }


def _select_quick(cases, registered):
    chosen = {}
    for name in registered:
        best = None
        for case in cases:
            names = {op for _domain, op in case["meta"].get("operators", [])}
            if name not in names:
                continue
            score = (case["meta"].get("operator_count", 1), case["test"])
            if best is None or score < best[0]:
                best = (score, case)
        if best is not None:
            chosen[best[1]["test"] + "/" + best[1]["dataset"]] = best[1]
    return sorted(chosen.values(), key=lambda case: case["test"])


def _apply_filters(cases, args):
    selected = cases
    if args.filter_op:
        wanted = set(args.filter_op)
        selected = [case for case in selected
                    if wanted & {op for _domain, op in
                                 case["meta"].get("operators", [])}]
    if args.filter_test:
        selected = [case for case in selected
                    if any(fnmatch.fnmatch(case["test"], pattern)
                           for pattern in args.filter_test)]
    if args.quick:
        selected = _select_quick(selected, registered_from(args))
    if args.sample:
        generator = random.Random(args.seed)
        selected = list(selected)
        generator.shuffle(selected)
        selected = selected[:args.sample]
        selected.sort(key=lambda case: case["test"])
    if args.max_tests:
        selected = selected[:args.max_tests]
    return selected


def registered_from(args):
    return getattr(args, "_registered", [])


def _progress_factory(total):
    state = {"done": 0, "passed": 0, "last": 0.0}

    def progress(_index, result):
        state["done"] += 1
        if result["result"] == runner.PASS:
            state["passed"] += 1
        now = time.monotonic()
        if state["done"] == total or now - state["last"] > 2.0:
            state["last"] = now
            sys.stderr.write(
                "\r  %d/%d cases (%d passed)      "
                % (state["done"], total, state["passed"]))
            sys.stderr.flush()
    return progress


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure ONNX operator coverage of onnx2c against the "
                    "official ONNX backend node tests.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--onnx2c", default=None)
    parser.add_argument("--testgen", default=None)
    parser.add_argument("--testgen-singlefile", default=None)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--accuracy", type=float, default=None,
                        help="override the per-test tolerance taken from "
                             "test/CMakeLists.txt")
    parser.add_argument("--cc-flag", action="append", dest="cc_flags")
    parser.add_argument("--test-data-dir", default=None)
    parser.add_argument("--onnx-defs-dir", default=None)
    parser.add_argument("--cmake-file", default=None,
                        help="upstream test/CMakeLists.txt used to learn each "
                             "test's mode and tolerance")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--badges-dir", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--no-cross-check", dest="cross_check",
                        action="store_false", default=True,
                        help="do not re-run each case in the other test mode")
    parser.add_argument("--filter-op", action="append", default=[])
    parser.add_argument("--filter-test", action="append", default=[])
    parser.add_argument("--quick", action="store_true",
                        help="one representative case per registered operator")
    parser.add_argument("--sample", type=int, default=0,
                        help="randomly sample N cases (subset mode)")
    parser.add_argument("--max-tests", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("auto", "exhaustive", "subset"),
                        default="auto")
    parser.add_argument("--list", action="store_true",
                        help="only list discovered tests and operators")
    parser.add_argument("--queue-report", action="store_true",
                        help="write .github/operator-coverage-status.json")
    parser.add_argument("--compare-baseline", default=None,
                        help="previous JSON report; fail on regression")
    return parser


def _resolve_paths(args):
    repo_root = args.repo_root or _repo_root(os.path.dirname(__file__))
    repo_root = os.path.abspath(repo_root)
    output_dir = args.output_dir or os.path.join(repo_root, "coverage-out")
    paths = {
        "repo_root": repo_root,
        "test_data_dir": args.test_data_dir or os.path.join(
            repo_root, "onnx", "onnx", "backend", "test", "data", "node"),
        "defs_dir": args.onnx_defs_dir or os.path.join(
            repo_root, "onnx", "onnx", "defs"),
        "graph_cc": os.path.join(repo_root, "src", "graph.cc"),
        "cmake_file": args.cmake_file or os.path.join(
            repo_root, "test", "CMakeLists.txt"),
        "output_dir": output_dir,
        "json": args.json or os.path.join(output_dir,
                                          "operator-coverage.json"),
        "markdown": args.markdown or os.path.join(
            output_dir, "operator-coverage.md"),
        "badges_dir": args.badges_dir or os.path.join(output_dir, "badges"),
    }
    return paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = _resolve_paths(args)

    if not os.path.exists(paths["graph_cc"]):
        sys.stderr.write(
            f"error: {paths['graph_cc']} not found; pass --repo-root\n")
        return 2

    onnx2c = _find_binary(paths["repo_root"], args.onnx2c, "onnx2c")
    testgen = _find_binary(paths["repo_root"], args.testgen, "testgen")
    testgen_singlefile = _find_binary(
        paths["repo_root"], args.testgen_singlefile, "testgen_singlefile")
    if not onnx2c or not testgen:
        sys.stderr.write(
            "error: could not locate onnx2c/testgen binaries.\n"
            "       build them first, or pass --onnx2c/--testgen\n")
        return 2
    if not testgen_singlefile:
        sys.stderr.write(
            "warning: testgen_singlefile not found; tests that upstream "
            "registers in singlefile mode cannot run\n")

    spec = discovery.read_spec_operators(paths["defs_dir"])
    registered = discovery.read_registered_operators(paths["graph_cc"])
    args._registered = registered

    registrations = discovery.read_upstream_registrations(paths["cmake_file"])
    cases = discovery.build_test_cases(paths["test_data_dir"], registrations)
    if args.accuracy is not None:
        for case in cases:
            case["upstream"]["accuracy"] = args.accuracy
            case["upstream"]["accuracy_overridden"] = True
    selected = _apply_filters(cases, args)

    missing_cases = discovery.missing_upstream_cases(
        registrations, paths["test_data_dir"])
    registration_summary = _registration_summary(registrations, cases,
                                                 missing_cases)

    limited = bool(args.filter_op or args.filter_test or args.quick
                   or args.sample or args.max_tests)
    if args.mode == "exhaustive":
        complete_requested = not limited
    elif args.mode == "subset":
        complete_requested = False
    else:
        complete_requested = not limited

    if args.list:
        print(f"repository:        {paths['repo_root']}")
        print(f"onnx2c:            {onnx2c}")
        print(f"testgen:           {testgen}")
        print(f"testgen_singlefile:{testgen_singlefile}")
        print(f"spec operators:    "
              f"{len(spec.get('ai.onnx', {}))} (ai.onnx)")
        print(f"registered ops:    {len(registered)}")
        print(f"upstream tests:    {registration_summary['active']} active, "
              f"{registration_summary['disabled']} disabled "
              f"({registration_summary['active_singlefile']} singlefile)")
        print(f"discovered cases:  {len(cases)}")
        print(f"selected cases:    {len(selected)}")
        return 0

    tools = {"onnx2c": onnx2c, "testgen": testgen,
             "testgen_singlefile": testgen_singlefile, "cc": args.cc}
    cc_flags = tuple(args.cc_flags) if args.cc_flags else DEFAULT_CC_FLAGS
    work_dir = args.work_dir or tempfile.mkdtemp(prefix="onnx2c-coverage-")

    started = time.monotonic()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    sys.stderr.write(f"running {len(selected)} cases with {args.jobs} jobs\n")
    results = runner.run_campaign(
        selected, tools, work_dir, jobs=args.jobs, timeout=args.timeout,
        accuracy=DEFAULT_ACCURACY, cc_flags=cc_flags,
        keep_work=args.keep_work, cross_check=args.cross_check,
        progress=_progress_factory(len(selected)))
    sys.stderr.write("\n")
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    cc_name, cc_version = _compiler_info(args.cc)
    repo_commit_short = _git(paths["repo_root"], "rev-parse",
                             "--short", "HEAD") or "unknown"
    onnx_commit = _git(os.path.join(paths["repo_root"], "onnx"),
                       "rev-parse", "--short", "HEAD") or "unknown"
    onnx_version = _git(os.path.join(paths["repo_root"], "onnx"),
                        "describe", "--tags") or "unknown"

    infra_errors = sum(1 for result in results
                       if result["result"] == runner.INFRASTRUCTURE_ERROR)
    report_operators, totals = report_module.aggregate(spec, registered, results)
    campaign = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(time.monotonic() - started, 2),
        "mode": "exhaustive" if complete_requested else "subset",
        "complete": False,
        "filters": {
            "operators": args.filter_op,
            "tests": args.filter_test,
            "quick": args.quick,
            "sample": args.sample,
            "max_tests": args.max_tests,
        },
        "cross_check": args.cross_check,
        "accuracy_override": args.accuracy is not None,
        "upstream_registrations": registration_summary,
        "onnx2c_commit": _git_commit(paths["repo_root"]),
        "onnx2c_commit_short": repo_commit_short,
        "onnx2c_dirty": _git_dirty(paths["repo_root"]),
        "onnx_submodule_commit": onnx_commit,
        "onnx_version": onnx_version,
        "compiler": cc_version,
        "compiler_command": cc_name,
        "onnx2c_binary": onnx2c,
        "testgen_binary": testgen,
        "testgen_singlefile_binary": testgen_singlefile,
        "cc_flags": list(cc_flags),
        "accuracy": DEFAULT_ACCURACY,
        "jobs": args.jobs,
        "timeout_seconds": args.timeout,
        "host": platform.node(),
        "python": platform.python_version(),
        "selected_cases": len(selected),
        "work_dir": work_dir,
    }
    report = report_module.build_report(campaign, spec, registered, results)
    report["campaign"]["complete"] = bool(
        complete_requested
        and len(results) == len(selected)
        and report["validation"]["ok"])
    report["campaign"]["clean"] = bool(infra_errors == 0)

    os.makedirs(paths["output_dir"], exist_ok=True)
    report_module.write_json(report, paths["json"])
    report_module.write_markdown(report, paths["markdown"])
    report_module.write_badges(report, paths["badges_dir"])

    if args.queue_report:
        _write_queue_status(report, paths["repo_root"])

    _print_summary(report, paths)

    if args.compare_baseline:
        return _compare_baseline(report, args.compare_baseline)

    if not report["validation"]["ok"]:
        sys.stderr.write("error: report failed self validation\n")
        return 3
    return 0


def _print_summary(report, paths):
    totals = report["totals"]
    campaign = report["campaign"]
    sys.stderr.write(
        "registered operators: %d/%d of ai.onnx\n"
        % (totals["registered_in_default_domain"],
           totals["spec_operators_default_domain"]))
    sys.stderr.write(
        "verified operators:   %d/%d\n"
        % (totals["default_domain_verified"],
           totals["spec_operators_default_domain"]))
    sys.stderr.write(
        "backend test cases:   %d/%d passed\n"
        % (totals["passed_cases"], totals["discovered_cases"]))
    sys.stderr.write(
        "test modes:           %d singlefile, %d split (upstream), "
        "cross-check %s\n"
        % (totals.get("cases_singlefile", 0),
           totals.get("cases_split", 0),
           "on" if campaign.get("cross_check") else "off"))
    if campaign.get("cross_check"):
        sys.stderr.write(
            "mode-sensitive cases: %d (primary and alternate mode disagree)\n"
            % totals.get("cross_check_mode_sensitive", 0))
    sys.stderr.write("report: %s\n" % paths["json"])
    sys.stderr.write("markdown: %s\n" % paths["markdown"])


def _write_queue_status(report, repo_root):
    directory = os.path.join(repo_root, ".github")
    os.makedirs(directory, exist_ok=True)
    totals = report["totals"]
    status = {
        "complete": report["campaign"]["complete"],
        "onnx2c_commit": report["campaign"]["onnx2c_commit"],
        "spec_operators": totals["spec_operators_default_domain"],
        "registered_operators": totals["registered_in_default_domain"],
        "verified_operators": totals["default_domain_verified"],
        "backend_tests_passed": totals["passed_cases"],
        "backend_tests_total": totals["discovered_cases"],
    }
    with open(os.path.join(directory, "operator-coverage-status.json"), "w",
              encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _compare_baseline(report, baseline_path):
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    base = baseline["totals"]
    now = report["totals"]
    regressions = []
    for key in ("default_domain_verified", "passed_cases",
                "registered_in_default_domain"):
        if now.get(key, 0) < base.get(key, 0):
            regressions.append(
                "%s dropped from %d to %d" % (key, base.get(key, 0),
                                              now.get(key, 0)))
    if regressions and report["campaign"]["complete"]:
        for line in regressions:
            sys.stderr.write("regression: %s\n" % line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
