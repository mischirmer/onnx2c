#!/usr/bin/env python3
"""Measure and publish onnx2c's ONNX operator coverage.

The tool answers three questions about the pinned ONNX release and reports
them as a Markdown file plus shields.io badges:

* **registered** - does onnx2c dispatch the operator (``Graph::createNode``)?
* **verified** - did at least one single-operator backend test pass end to end?
* **test cases** - how many individual upstream backend tests pass?

Only standard-library Python is used.  The operator set and its deprecations
are read straight from the pinned ``onnx`` submodule (``onnx/onnx/defs``), so
the tool never needs the ONNX Python package and can never go stale.  Cases
reuse the tool invocations from ``test/CMakeLists.txt``: ``onnx2c`` +
``testgen`` (*split* mode) or ``testgen_singlefile`` (the *singlefile* mode
upstream uses for a few tests).  Generated C is compiled with a
warning-tolerant flag set (``-Wall -Wno-unused-variable -lm``); unlike CTest
it drops ``-Werror`` and ``-Wno-stringop-overread``, so warning-only
differences never become failures.

Requires built ``onnx2c``, ``testgen`` and (for upstream singlefile tests)
``testgen_singlefile`` binaries:

    python3 test/operator_coverage/coverage.py --output-dir coverage-out
"""

import argparse
import datetime
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

DEFAULT_DOMAIN = "ai.onnx"

PASS = "pass"
ONNX_TO_C_FAILURE = "onnx_to_c_failure"
C_COMPILATION_FAILURE = "c_compilation_failure"
EXECUTION_FAILURE = "execution_failure"
NUMERICAL_MISMATCH = "numerical_mismatch"
TIMEOUT = "timeout"
INFRASTRUCTURE_ERROR = "infrastructure_error"

FAILURE_CATEGORIES = (ONNX_TO_C_FAILURE, C_COMPILATION_FAILURE,
                      EXECUTION_FAILURE, NUMERICAL_MISMATCH, TIMEOUT,
                      INFRASTRUCTURE_ERROR)
UNRUNNABLE_CATEGORIES = (TIMEOUT, INFRASTRUCTURE_ERROR)
FAILED_RESULTS = frozenset((ONNX_TO_C_FAILURE, C_COMPILATION_FAILURE,
                            EXECUTION_FAILURE, NUMERICAL_MISMATCH))
UNRUNNABLE_RESULTS = frozenset(UNRUNNABLE_CATEGORIES)

STAGE_TRANSLATE = "onnx_to_c"
STAGE_HARNESS = "harness_generation"
STAGE_COMPILE = "c_compilation"
STAGE_EXECUTE = "execution"

CAT_MISSING = "missing"
CAT_UNTESTED = "untested"
CAT_FAILING = "failing"
CAT_PARTIAL = "partial"
CAT_BLOCKED = "blocked"
CAT_PASSING = "passing"

CATEGORY_ORDER = (CAT_MISSING, CAT_FAILING, CAT_PARTIAL, CAT_BLOCKED,
                  CAT_UNTESTED, CAT_PASSING)
CATEGORY_LABEL = {
    CAT_MISSING: "Missing",
    CAT_UNTESTED: "Untested",
    CAT_FAILING: "Failing",
    CAT_PARTIAL: "Partial",
    CAT_BLOCKED: "Blocked",
    CAT_PASSING: "Passing",
}

MODE_SPLIT = "split"
MODE_SINGLEFILE = "singlefile"
DEFAULT_ACCURACY = 2e-05
DEFAULT_CC_FLAGS = ("-Wall", "-Wno-unused-variable")

SCHEMA_VERSION = 3

_DETAIL_LIMIT = 500

# Failed-stage output that describes the test *tooling*, not onnx2c.
_TOOLING_ERROR_MARKERS = (
    "unimplemented type",
    "unhandled tensor data type",
)
_ENVIRONMENT_ERROR_MARKERS = (
    "could not execute",
    "Error opening",
    "No such file",
)

FAILURE_COLUMNS = (
    ("ONNX\u2192C", (ONNX_TO_C_FAILURE,)),
    ("C compile", (C_COMPILATION_FAILURE,)),
    ("Numerical", (NUMERICAL_MISMATCH,)),
    ("Runtime", (EXECUTION_FAILURE, TIMEOUT)),
    ("Infrastructure", (INFRASTRUCTURE_ERROR,)),
)

_LEGEND = (
    ("Missing", "no implementation registered"),
    ("Untested", "registered, but no single-operator test exists"),
    ("Failing", "registered, tests exist, none pass"),
    ("Partial", "at least one test passes, at least one fails"),
    ("Blocked", "tests exist, but infrastructure errors leave the result "
                "unreliable"),
    ("Passing", "all applicable tests pass"),
)

_SPEC_RE = re.compile(
    r"ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME\(\s*Onnx\s*,\s*(\d+)\s*,\s*(\w+)\s*\)")
_SCHEMA_RE = re.compile(
    r"ONNX_OPERATOR_SET_SCHEMA\(\s*(\w+)\s*,\s*(\d+)\s*,", re.MULTILINE)
_REGISTERED_RE = re.compile(r'opName\s*==\s*"([A-Za-z0-9_]+)"')
_CREATE_NODE_RE = re.compile(r"Graph::createNode\b.*?(?=\n}\n)", re.DOTALL)
_REGISTRATION_RE = re.compile(
    r"^(ONNX_backend_node_test(?:_singlefile|_with_accuracy)?)"
    r"\s*\(\s*([A-Za-z0-9_]+)\s*(.*?)\)")
_ACCURACY_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")


class MalformedModel(ValueError):
    """Raised when bytes cannot be decoded as the ONNX protobuf subset."""


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Minimal protobuf reader for the ONNX model subset (op_type + domain).
# ---------------------------------------------------------------------------

def _varint(data, pos):
    shift = 0
    value = 0
    size = len(data)
    while True:
        if pos >= size:
            raise MalformedModel("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise MalformedModel("varint too long")


def _iter_fields(data):
    pos = 0
    size = len(data)
    while pos < size:
        tag, pos = _varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07
        if field == 0:
            raise MalformedModel("field number 0 is invalid")
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 2:
            length, pos = _varint(data, pos)
            end = pos + length
            if end > size:
                raise MalformedModel("truncated length-delimited field")
            value = data[pos:end]
            pos = end
        elif wire == 5:
            end = pos + 4
            if end > size:
                raise MalformedModel("truncated 32-bit field")
            value = data[pos:end]
            pos = end
        elif wire == 1:
            end = pos + 8
            if end > size:
                raise MalformedModel("truncated 64-bit field")
            value = data[pos:end]
            pos = end
        else:
            raise MalformedModel("unsupported wire type %d" % wire)
        yield field, wire, value


def _node_operator(data):
    domain = DEFAULT_DOMAIN
    op_type = ""
    for field, wire, value in _iter_fields(data):
        if wire != 2:
            continue
        text = value.decode("utf-8", "replace")
        if field == 4:
            op_type = text
        elif field == 7:
            domain = text or DEFAULT_DOMAIN
    return domain if domain != "ai.onnx" else DEFAULT_DOMAIN, op_type


def parse_model_operators(path):
    """Return the de-duplicated ``"domain:name"`` operators of an ONNX model.

    Only ``ModelProto.graph`` (7) / ``GraphProto.node`` (1) / ``NodeProto``
    ``op_type`` (4) and ``domain`` (7) are decoded; unknown fields are skipped
    so newer ONNX releases keep working.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    nodes = []
    for field, wire, value in _iter_fields(data):
        if field == 7 and wire == 2:
            nodes = [value for f, w, value in _iter_fields(value)
                     if f == 1 and w == 2]
            break
    seen = []
    for node in nodes:
        domain, op_type = _node_operator(node)
        if not op_type:
            continue
        key = "%s:%s" % (domain, op_type)
        if key not in seen:
            seen.append(key)
    return seen


# ---------------------------------------------------------------------------
# Discovery from the pinned ONNX submodule and onnx2c itself.
# ---------------------------------------------------------------------------

def read_spec(defs_dir):
    """Return ``{operator_name: max_version}`` for the default ``ai.onnx`` domain.

    ``operator_sets.h`` lists every schema in the pinned release, so its
    maximum version is also the active default-domain opset.
    """
    header = os.path.join(defs_dir, "operator_sets.h")
    if not os.path.exists(header):
        raise FileNotFoundError("ONNX operator set header not found: %s"
                                % header)
    table = {}
    for version, name in _SPEC_RE.findall(_read_text(header)):
        table[name] = max(table.get(name, 0), int(version))
    if not table:
        raise MalformedModel("no operators parsed from %s" % header)
    return table


def read_onnx_version(defs_dir):
    """Return the pinned ONNX version string, or ``"unknown"``.

    ``defs_dir`` is ``<repo>/onnx/onnx/defs``; ``VERSION_NUMBER`` lives at the
    submodule root, two directories up.
    """
    path = os.path.normpath(os.path.join(defs_dir, os.pardir, os.pardir,
                                         "VERSION_NUMBER"))
    if not os.path.exists(path):
        return "unknown"
    return _read_text(path).strip() or "unknown"


def read_deprecated(defs_dir, spec):
    """Return the default-domain operators deprecated at the pinned opset.

    An operator counts as deprecated when the ``.Deprecate()`` marker sits on
    its *latest* schema (e.g. ``Scatter`` 11, ``Upsample`` 10).  A marker on an
    older schema does not count: ``GroupNormalization`` 18 is deprecated, but
    version 21 is active.
    """
    deprecated = set()
    for path in glob.glob(os.path.join(defs_dir, "**", "*.cc"), recursive=True):
        text = _read_text(path)
        matches = list(_SCHEMA_RE.finditer(text))
        for index, match in enumerate(matches):
            name = match.group(1)
            version = int(match.group(2))
            if spec.get(name) != version:
                continue
            end = (matches[index + 1].start() if index + 1 < len(matches)
                   else len(text))
            if ".Deprecate()" in text[match.start():end]:
                deprecated.add(name)
    return deprecated


def read_registered_operators(graph_cc_path):
    """Return the ordered, de-duplicated operators dispatched by onnx2c.

    Registration is a flat ``if (opName == "...")`` chain inside
    ``Graph::createNode``; that body is isolated first so comparisons inside
    node implementations are never mistaken for registrations.
    """
    text = _read_text(graph_cc_path)
    match = _CREATE_NODE_RE.search(text)
    body = match.group(0) if match else text
    seen = []
    for name in _REGISTERED_RE.findall(body):
        if name not in seen:
            seen.append(name)
    return seen


def read_registrations(cmake_path):
    """Return ``{test_name: (mode, accuracy)}`` from ``test/CMakeLists.txt``.

    Both active and commented-out (known-broken upstream) node tests are
    recorded, because the report still executes the commented-out ones so
    their failures stay visible.
    """
    registrations = {}
    if not os.path.exists(cmake_path):
        return registrations
    for line in _read_text(cmake_path).splitlines():
        body = line.strip().lstrip("#").strip()
        match = _REGISTRATION_RE.match(body)
        if not match:
            continue
        helper, name, rest = match.groups()
        mode = MODE_SINGLEFILE if helper.endswith("_singlefile") else MODE_SPLIT
        accuracy = DEFAULT_ACCURACY
        if helper.endswith("_with_accuracy"):
            accuracy_match = _ACCURACY_RE.match(rest)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
        registrations[name] = (mode, accuracy)
    return registrations


def discover_cases(test_data_dir, registrations):
    """Return every ``(test, dataset)`` case with its operators and test mode."""
    cases = []
    for path in sorted(glob.glob(os.path.join(test_data_dir, "test_*"))):
        if not os.path.isdir(path):
            continue
        model = os.path.join(path, "model.onnx")
        if not os.path.exists(model):
            continue
        test = os.path.basename(path)
        short = test[len("test_"):] if test.startswith("test_") else test
        mode, accuracy = registrations.get(short,
                                           (MODE_SPLIT, DEFAULT_ACCURACY))
        try:
            operators = parse_model_operators(model)
        except (OSError, MalformedModel):
            operators = []
        datasets = sorted(name for name in os.listdir(path)
                          if name.startswith("test_data_set_")
                          and os.path.isdir(os.path.join(path, name)))
        for dataset in (datasets or ["test_data_set_0"]):
            index = (dataset[len("test_data_set_"):]
                     if dataset.startswith("test_data_set_") else dataset)
            cases.append({
                "id": "%s/%s" % (test, dataset),
                "test": test,
                "dataset": dataset,
                "dataset_index": index,
                "dir": path,
                "model": model,
                "operators": operators,
                "mode": mode,
                "accuracy": accuracy,
            })
    return cases


# ---------------------------------------------------------------------------
# Case execution.
# ---------------------------------------------------------------------------

def _tail(data):
    if not data:
        return ""
    text = data.decode("utf-8", "replace").strip()
    if len(text) > _DETAIL_LIMIT:
        text = "..." + text[-_DETAIL_LIMIT:]
    return text


def _run(command, stdout_path, timeout):
    """Run a command, streaming stdout to a file.

    Returns ``(returncode, detail, timed_out)``; ``returncode`` is ``None`` on
    timeout or when the command could not be started.
    """
    try:
        with open(stdout_path, "w") as stdout_handle:
            completed = subprocess.run(
                command, stdout=stdout_handle, stderr=subprocess.PIPE,
                timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return None, _tail(exc.stderr), True
    except OSError as exc:
        return None, "could not execute %s: %s" % (command[0], exc), False
    return completed.returncode, _tail(completed.stderr), False


def _matches(markers, detail):
    return any(marker in detail for marker in markers)


def _sanitize(name):
    return "".join(char if char.isalnum() or char in "-_." else "_"
                   for char in name)


def _compile_and_execute(tools, case_dir, sources, timeout, cc_flags):
    binary = os.path.join(case_dir, "case")
    command = [tools["cc"]] + list(cc_flags) + list(sources) + ["-o", binary,
                                                                "-lm"]
    returncode, detail, timed_out = _run(
        command, os.path.join(case_dir, "compile.log"), timeout)
    if timed_out:
        return TIMEOUT, STAGE_COMPILE, None, detail
    if returncode is None or returncode != 0:
        return C_COMPILATION_FAILURE, STAGE_COMPILE, returncode, detail

    returncode, detail, timed_out = _run(
        [binary], os.path.join(case_dir, "run.log"), timeout)
    if timed_out:
        return TIMEOUT, STAGE_EXECUTE, None, detail
    if returncode == 0:
        return PASS, STAGE_EXECUTE, 0, detail
    if returncode == 1:
        return (NUMERICAL_MISMATCH, STAGE_EXECUTE, 1,
                detail or "reference comparison failed")
    return EXECUTION_FAILURE, STAGE_EXECUTE, returncode, detail


def _run_split(case, tools, case_dir, timeout, accuracy, cc_flags):
    """onnx2c translation + testgen harness, as CTest's split mode does."""
    generated = os.path.join(case_dir, "generated.c")
    harness = os.path.join(case_dir, "harness.c")
    returncode, detail, timed_out = _run(
        [tools["onnx2c"], "-l", "0", case["model"]], generated, timeout)
    if timed_out:
        return TIMEOUT, STAGE_TRANSLATE, None, detail
    if returncode != 0:
        return ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode, detail
    if os.path.getsize(generated) == 0:
        return (ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode,
                "onnx2c produced no output")

    returncode, detail, timed_out = _run(
        [tools["testgen"], case["dir"], "%.10g" % accuracy,
         case["dataset_index"]], harness, timeout)
    if timed_out:
        return TIMEOUT, STAGE_HARNESS, None, detail
    if returncode != 0:
        return INFRASTRUCTURE_ERROR, STAGE_HARNESS, returncode, detail
    return _compile_and_execute(tools, case_dir, [harness, generated], timeout,
                                cc_flags)


def _run_singlefile(case, tools, case_dir, timeout, accuracy, cc_flags):
    """testgen_singlefile emits graph *and* harness, as singlefile mode does."""
    generator = tools.get("testgen_singlefile")
    if not generator:
        return (INFRASTRUCTURE_ERROR, STAGE_HARNESS, None,
                "testgen_singlefile binary not available")
    generated = os.path.join(case_dir, "generated.c")
    returncode, detail, timed_out = _run(
        [generator, case["dir"], "%.10g" % accuracy, case["dataset_index"]],
        generated, timeout)
    if timed_out:
        return TIMEOUT, STAGE_HARNESS, None, detail
    if returncode != 0:
        if (_matches(_TOOLING_ERROR_MARKERS, detail)
                or _matches(_ENVIRONMENT_ERROR_MARKERS, detail)):
            return INFRASTRUCTURE_ERROR, STAGE_HARNESS, returncode, detail
        return ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode, detail
    if os.path.getsize(generated) == 0:
        return (ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode,
                "testgen_singlefile produced no output")
    return _compile_and_execute(tools, case_dir, [generated], timeout, cc_flags)


def run_case(case, tools, work_root, timeout, cc_flags):
    """Run one case in its upstream-registered mode and return its record."""
    case_dir = os.path.join(work_root, _sanitize(case["test"]),
                            _sanitize(case["dataset"]))
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    if case["mode"] == MODE_SINGLEFILE:
        outcome, stage, returncode, detail = _run_singlefile(
            case, tools, case_dir, timeout, case["accuracy"], cc_flags)
    else:
        outcome, stage, returncode, detail = _run_split(
            case, tools, case_dir, timeout, case["accuracy"], cc_flags)
    return {
        "id": case["id"],
        "test": case["test"],
        "dataset": case["dataset"],
        "operators": case["operators"],
        "mode": case["mode"],
        "result": outcome,
        "stage": stage,
        "exit_code": returncode,
        "detail": "" if outcome == PASS else detail,
    }


def run_campaign(cases, tools, work_root, jobs=1, timeout=60.0,
                 cc_flags=DEFAULT_CC_FLAGS, keep_work=False, progress=None):
    """Run every case, optionally in parallel; return the ordered results."""
    os.makedirs(work_root, exist_ok=True)
    results = [None] * len(cases)

    def work(index):
        case = cases[index]
        result = run_case(case, tools, work_root, timeout, cc_flags)
        if not keep_work:
            case_dir = os.path.join(work_root, _sanitize(case["test"]),
                                    _sanitize(case["dataset"]))
            shutil.rmtree(case_dir, ignore_errors=True)
        if progress is not None:
            progress(index, result)
        return index, result

    if jobs <= 1:
        for index in range(len(cases)):
            _, results[index] = work(index)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for index, result in pool.map(work, range(len(cases))):
                results[index] = result
    return results


# ---------------------------------------------------------------------------
# Aggregation, validation, and rendering.
# ---------------------------------------------------------------------------

def classify(registered, applicable, passed, failed, unrunnable):
    """Map a single-operator's counters onto one mutually exclusive category.

    A registered operator is never *Passing* while any applicable test remains
    unevaluated - such uncertainty is *Blocked*.
    """
    if not registered:
        return CAT_MISSING
    if applicable == 0:
        return CAT_UNTESTED
    if failed > 0:
        return CAT_PARTIAL if passed > 0 else CAT_FAILING
    if unrunnable > 0:
        return CAT_BLOCKED
    if passed > 0:
        return CAT_PASSING
    return CAT_UNTESTED


def _is_deprecated_only(result, deprecated_keys):
    operators = set(result["operators"])
    return bool(operators) and operators <= deprecated_keys


def aggregate(spec, registered, results, deprecated):
    """Return ``(operator_rows, totals)`` derived from the raw records.

    Per-operator verdicts come from *single-operator* tests only: a test that
    exercises several operators at once cannot be attributed to any one of
    them, so it never changes an operator's status.
    """
    deprecated_set = set(deprecated)
    registered_set = set(registered)
    rows = {}
    for name, version in spec.items():
        rows[name] = {
            "operator": name,
            "version": version,
            "registered": name in registered_set,
            "deprecated": name in deprecated_set,
            "applicable_cases": 0,
            "solo_cases": 0,
            "solo_passed": 0,
            "solo_failed": 0,
            "solo_unrunnable": 0,
            "solo_per_category": {},
        }
    for result in results:
        keys = sorted(set(result["operators"]))
        solo = len(keys) == 1
        outcome = result["result"]
        for key in keys:
            domain, _, name = key.partition(":")
            row = rows.get(name)
            if row is None or domain != DEFAULT_DOMAIN:
                continue
            row["applicable_cases"] += 1
            if not solo:
                continue
            row["solo_cases"] += 1
            counts = row["solo_per_category"]
            counts[outcome] = counts.get(outcome, 0) + 1
            if outcome == PASS:
                row["solo_passed"] += 1
            elif outcome in FAILED_RESULTS:
                row["solo_failed"] += 1
            elif outcome in UNRUNNABLE_RESULTS:
                row["solo_unrunnable"] += 1
    for row in rows.values():
        row["status"] = classify(
            row["registered"], row["solo_cases"], row["solo_passed"],
            row["solo_failed"], row["solo_unrunnable"])
    ordered = sorted(rows.values(), key=lambda row: row["operator"])
    active = [row for row in ordered if not row["deprecated"]]
    deprecated_rows = [row for row in ordered if row["deprecated"]]
    deprecated_keys = set("%s:%s" % (DEFAULT_DOMAIN, name)
                          for name in deprecated_set)
    return ordered, _totals(active, deprecated_rows, results, deprecated_keys)


def _totals(active_rows, deprecated_rows, results, deprecated_keys):
    active_results = [result for result in results
                      if not _is_deprecated_only(result, deprecated_keys)]
    case_counts = Counter(result["result"] for result in results)
    active_counts = Counter(result["result"] for result in active_results)
    active_cat = Counter(row["status"] for row in active_rows)
    deprecated_cat = Counter(row["status"] for row in deprecated_rows)
    totals = {
        "spec_operators_active": len(active_rows),
        "spec_operators_deprecated": len(deprecated_rows),
        "registered_active": sum(1 for row in active_rows if row["registered"]),
        "active_verified": (active_cat.get(CAT_PASSING, 0)
                            + active_cat.get(CAT_PARTIAL, 0)),
        "deprecated_verified": (deprecated_cat.get(CAT_PASSING, 0)
                                + deprecated_cat.get(CAT_PARTIAL, 0)),
        "discovered_cases": len(results),
        "active_cases": len(active_results),
        "active_passed_cases": active_counts.get(PASS, 0),
        "active_infrastructure_errors": active_counts.get(
            INFRASTRUCTURE_ERROR, 0),
        "deprecated_only_cases": len(results) - len(active_results),
        "passed_cases": case_counts.get(PASS, 0),
        "cases_singlefile": sum(1 for result in results
                                if result["mode"] == MODE_SINGLEFILE),
        "cases_split": sum(1 for result in results
                           if result["mode"] == MODE_SPLIT),
    }
    for category in CATEGORY_ORDER:
        totals["active_" + category] = active_cat.get(category, 0)
        totals["deprecated_" + category] = deprecated_cat.get(category, 0)
    return totals


def validate(report):
    """Recompute the aggregates from the stored rows and cases."""
    errors = []
    rows = report["operators"]
    results = report["tests"]
    totals = report["totals"]
    deprecated_keys = set(report["campaign"].get("deprecated_operators", []))
    active_rows = [row for row in rows if not row["deprecated"]]
    deprecated_rows = [row for row in rows if row["deprecated"]]

    def check(key, value):
        if totals.get(key) != value:
            errors.append("%s: stored=%s recomputed=%s"
                          % (key, totals.get(key), value))

    check("spec_operators_active", len(active_rows))
    check("spec_operators_deprecated", len(deprecated_rows))
    check("registered_active", sum(1 for row in active_rows
                                   if row["registered"]))
    for prefix, group in (("active", active_rows),
                          ("deprecated", deprecated_rows)):
        counts = Counter(row["status"] for row in group)
        for category in CATEGORY_ORDER:
            check("%s_%s" % (prefix, category), counts.get(category, 0))
    check("active_verified", sum(1 for row in active_rows
                                 if row["status"] in (CAT_PASSING, CAT_PARTIAL)))
    check("deprecated_verified", sum(
        1 for row in deprecated_rows
        if row["status"] in (CAT_PASSING, CAT_PARTIAL)))
    case_counts = Counter(result["result"] for result in results)
    check("passed_cases", case_counts.get(PASS, 0))
    check("discovered_cases", len(results))
    active_results = [result for result in results
                      if not _is_deprecated_only(result, deprecated_keys)]
    active_counts = Counter(result["result"] for result in active_results)
    check("active_cases", len(active_results))
    check("active_passed_cases", active_counts.get(PASS, 0))
    check("active_infrastructure_errors",
          active_counts.get(INFRASTRUCTURE_ERROR, 0))
    check("deprecated_only_cases", len(results) - len(active_results))
    check("cases_singlefile", sum(1 for result in results
                                  if result["mode"] == MODE_SINGLEFILE))
    check("cases_split", sum(1 for result in results
                             if result["mode"] == MODE_SPLIT))
    return errors


def build_report(campaign, spec, registered, results, deprecated):
    rows, totals = aggregate(spec, registered, results, deprecated)
    report = {
        "schema_version": SCHEMA_VERSION,
        "campaign": campaign,
        "totals": totals,
        "operators": rows,
        "tests": sorted(results, key=lambda result: result["id"]),
    }
    errors = validate(report)
    report["validation"] = {"ok": not errors, "errors": errors}
    return report


def write_json(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _ratio(passed, total):
    return (passed / total) if total else 0.0


def _format_ratio(passed, total):
    if total == 0:
        return "%d/%d" % (passed, total)
    return "%d/%d (%.1f%%)" % (passed, total, 100.0 * passed / total)


def _badge_color(passed, total):
    ratio = _ratio(passed, total)
    if ratio >= 0.9:
        return "brightgreen"
    if ratio >= 0.75:
        return "green"
    if ratio >= 0.5:
        return "yellow"
    if ratio >= 0.25:
        return "orange"
    return "red"


def _badge(label, passed, total):
    return {
        "schemaVersion": 1,
        "label": label,
        "message": "%d/%d (%.1f%%)" % (passed, total, 100.0 * _ratio(passed,
                                                                    total)),
        "color": _badge_color(passed, total),
    }


def badge_payloads(report):
    """Shields.io endpoint payloads for the *active* operator denominators."""
    totals = report["totals"]
    denominator = totals["spec_operators_active"]
    return {
        "registered.json": _badge("onnx2c registered operators",
                                  totals["registered_active"], denominator),
        "verified.json": _badge("onnx2c verified operators",
                                totals["active_verified"], denominator),
        "tests.json": _badge("onnx2c backend tests",
                             totals["active_passed_cases"],
                             totals["active_cases"]),
    }


def write_badges(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for filename, payload in badge_payloads(report).items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written[filename] = path
    return written


def _failure_count(row, categories):
    counts = row.get("solo_per_category", {})
    return sum(counts.get(category, 0) for category in categories)


def _matrix_lines(rows):
    lines = []
    for row in rows:
        lines.append("| %s | %s | %d | %s |"
                     % (row["operator"], "yes" if row["registered"] else "no",
                        row["solo_passed"], CATEGORY_LABEL[row["status"]]))
    return lines


def write_markdown(report, path):
    """Render the compact, developer-facing compatibility overview."""
    campaign = report["campaign"]
    totals = report["totals"]
    active_rows = sorted(
        (row for row in report["operators"] if not row["deprecated"]),
        key=lambda row: (CATEGORY_ORDER.index(row["status"]), row["operator"]))
    deprecated_rows = sorted(
        (row for row in report["operators"] if row["deprecated"]),
        key=lambda row: row["operator"])
    denom = totals["spec_operators_active"]
    opset = campaign.get("spec", {}).get("opset", "?")
    revision = (campaign.get("onnx2c_commit_short")
                or campaign.get("onnx2c_commit", "unknown"))
    onnx_version = campaign.get("onnx_version", "unknown")

    lines = ["# ONNX operator compatibility", ""]
    if not campaign.get("complete", False):
        lines.extend([
            "> **Warning:** this is a partial campaign; the matrix is "
            "incomplete.", ""])
    lines.extend([
        "| Metric | Value |",
        "| --- | --- |",
        "| ONNX version | %s (`%s`) |"
        % (onnx_version, campaign.get("onnx_submodule_commit", "unknown")),
        "| onnx2c revision | `%s` |" % revision,
        "| Registered operators | %s |"
        % _format_ratio(totals["registered_active"], denom),
        "| Verified operators | %s |"
        % _format_ratio(totals["active_verified"], denom),
        "| Passing backend tests | %s |"
        % _format_ratio(totals["active_passed_cases"], totals["active_cases"]),
        "| Infrastructure errors | %s |"
        % totals["active_infrastructure_errors"],
        "",
        "Denominators count the %d non-deprecated `ai.onnx` operators active at "
        "opset %s (deprecated operators are listed separately below). Backend "
        "tests count the %d of %d discovered cases not exclusively "
        "attributable to a deprecated operator."
        % (denom, opset, totals["active_cases"], totals["discovered_cases"]),
        "",
        "* **Registered**: onnx2c dispatches the operator. "
        "**Verified**: at least one single-operator backend test passes. "
        "**Passing backend tests**: individual test cases that pass end to "
        "end.",
        "",
        "## Operator compatibility",
        "",
        "Every active (non-deprecated) operator of the pinned ONNX "
        "specification's default `ai.onnx` domain, classified from its "
        "single-operator tests.",
        "",
        "| Operator | Implemented | Tests passed | Status |",
        "| --- | --- | --- | --- |",
    ])
    lines.extend(_matrix_lines(active_rows))
    counts = Counter(row["status"] for row in active_rows)
    lines.extend([
        "",
        "Status counts: "
        + ", ".join("%s %d" % (CATEGORY_LABEL[category],
                               counts.get(category, 0))
                    for category in CATEGORY_ORDER)
        + ". Legend: "
        + "; ".join("**%s** \u2013 %s" % (name, text)
                    for name, text in _LEGEND)
        + ".",
        "",
    ])

    if deprecated_rows:
        lines.extend([
            "## Deprecated operators",
            "",
            "Deprecated in the pinned specification and excluded from the "
            "denominators above. Reported for completeness only.",
            "",
            "| Operator | Implemented | Tests passed | Status |",
            "| --- | --- | --- | --- |",
        ])
        lines.extend(_matrix_lines(deprecated_rows))
        lines.extend([
            "",
            "Deprecated tests excluded from the active backend-test counts: "
            "%d." % totals["deprecated_only_cases"],
            "",
        ])

    lines.extend([
        "## Failure details",
        "",
        "Only operators that are **Failing** or **Blocked**, counted from "
        "their single-operator tests.",
        "",
        "| Operator | " + " | ".join(name for name, _ in FAILURE_COLUMNS)
        + " |",
        "| --- |" + " ---: |" * len(FAILURE_COLUMNS),
    ])
    required = [row for row in active_rows
                if row["status"] in (CAT_FAILING, CAT_BLOCKED)]
    if required:
        for row in required:
            cells = " | ".join(str(_failure_count(row, categories))
                               for _name, categories in FAILURE_COLUMNS)
            lines.append("| %s | %s |" % (row["operator"], cells))
    else:
        lines.append("| _(none)_ | "
                     + " | ".join("0" for _ in FAILURE_COLUMNS) + " |")
    lines.extend([
        "",
        "## Coverage limitations",
        "",
        "- Passing the available backend tests does not prove full ONNX "
        "compatibility.",
        "- Operator versions, attributes, data types, and tensor shapes are "
        "not exhaustively tested.",
        "- Multi-operator tests cannot be attributed to a single "
        "implementation, so they never change an operator's status.",
        "- Deprecated operators are excluded from the denominators; only cases "
        "whose operators are all deprecated are dropped. Infrastructure errors "
        "are tooling limitations and never counted as passing tests.",
        "- Detailed per-case diagnostics live in `operator-coverage.json` and "
        "the CI artifacts.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Command line entry point.
# ---------------------------------------------------------------------------

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
        candidates = ([os.path.join(base, "onnx2c")] if kind == "onnx2c"
                      else [os.path.join(base, "test", kind),
                            os.path.join(base, kind)])
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


def _compiler_info(cc):
    try:
        out = subprocess.run([cc, "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return cc, "unknown"
    text = out.stdout.decode("utf-8", "replace").strip().splitlines()
    return cc, (text[0] if text else "unknown")


def _progress_factory(total):
    def progress(index, _result):
        if index % 50 == 0 or index + 1 == total:
            sys.stderr.write("\r  %d/%d cases" % (index + 1, total))
            sys.stderr.flush()
    return progress


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=None,
                        help="onnx2c checkout (default: auto-detected)")
    parser.add_argument("--onnx2c", default=None)
    parser.add_argument("--testgen", default=None)
    parser.add_argument("--testgen-singlefile", default=None)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--output-dir", default=None,
                        help="where to write the report and badges")
    parser.add_argument("--work-dir", default=None,
                        help="scratch directory (default: a temporary dir)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep-work", action="store_true",
                        help="keep per-case build directories")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    repo_root = os.path.abspath(
        args.repo_root or _repo_root(os.path.dirname(os.path.abspath(__file__))))
    defs_dir = os.path.join(repo_root, "onnx", "onnx", "defs")
    test_data_dir = os.path.join(repo_root, "onnx", "onnx", "backend", "test",
                                 "data", "node")
    graph_cc = os.path.join(repo_root, "src", "graph.cc")
    cmake = os.path.join(repo_root, "test", "CMakeLists.txt")
    if not os.path.exists(graph_cc):
        sys.stderr.write("error: %s not found; pass --repo-root\n" % graph_cc)
        return 2

    try:
        spec = read_spec(defs_dir)
    except (OSError, MalformedModel) as exc:
        sys.stderr.write("error: cannot read the pinned ONNX specification: "
                         "%s\n" % exc)
        return 2
    deprecated = read_deprecated(defs_dir, spec)
    registered = read_registered_operators(graph_cc)
    cases = discover_cases(test_data_dir, read_registrations(cmake))
    if not cases:
        sys.stderr.write("error: no backend test cases found under %s\n"
                         % test_data_dir)
        return 2

    tools = {
        "onnx2c": _find_binary(repo_root, args.onnx2c, "onnx2c"),
        "testgen": _find_binary(repo_root, args.testgen, "testgen"),
        "testgen_singlefile": _find_binary(repo_root, args.testgen_singlefile,
                                           "testgen_singlefile"),
        "cc": args.cc,
    }
    for kind in ("onnx2c", "testgen"):
        if not tools[kind]:
            sys.stderr.write("error: %s binary not found; build it or pass "
                             "--%s\n" % (kind, kind))
            return 2

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="onnx2c-coverage-")
    started = time.monotonic()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    sys.stderr.write("running %d cases with %d jobs\n" % (len(cases),
                                                          args.jobs))
    results = run_campaign(
        cases, tools, work_dir, jobs=args.jobs, timeout=args.timeout,
        cc_flags=DEFAULT_CC_FLAGS, keep_work=args.keep_work,
        progress=_progress_factory(len(cases)))
    sys.stderr.write("\n")
    onnx_version = read_onnx_version(defs_dir)

    campaign = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 2),
        "complete": False,
        "onnx2c_commit": _git(repo_root, "rev-parse", "HEAD") or "unknown",
        "onnx2c_commit_short": _git(repo_root, "rev-parse", "--short", "HEAD")
        or "unknown",
        "onnx_submodule_commit": _git(os.path.join(repo_root, "onnx"),
                                      "rev-parse", "--short", "HEAD")
        or "unknown",
        "onnx_version": onnx_version,
        "spec": {"opset": max(spec.values()), "operators": len(spec)},
        "deprecated_operators": sorted("%s:%s" % (DEFAULT_DOMAIN, name)
                                       for name in deprecated),
        "compiler": _compiler_info(args.cc)[1],
        "cc_flags": list(DEFAULT_CC_FLAGS),
        "jobs": args.jobs,
        "timeout_seconds": args.timeout,
        "host": platform.node(),
        "python": platform.python_version(),
        "selected_cases": len(cases),
    }
    report = build_report(campaign, spec, registered, results, deprecated)
    report["campaign"]["complete"] = bool(
        len(results) == len(cases) and report["validation"]["ok"])

    output_dir = args.output_dir or os.path.join(repo_root, "coverage-out")
    os.makedirs(output_dir, exist_ok=True)
    write_json(report, os.path.join(output_dir, "operator-coverage.json"))
    write_markdown(report, os.path.join(output_dir, "operator-coverage.md"))
    write_badges(report, os.path.join(output_dir, "badges"))

    totals = report["totals"]
    sys.stderr.write(
        "registered operators: %d/%d active ai.onnx\n"
        % (totals["registered_active"], totals["spec_operators_active"]))
    sys.stderr.write("verified operators:   %d/%d\n"
                     % (totals["active_verified"],
                        totals["spec_operators_active"]))
    sys.stderr.write("backend test cases:   %d/%d\n"
                     % (totals["active_passed_cases"], totals["active_cases"]))
    sys.stderr.write("report: %s\n"
                     % os.path.join(output_dir, "operator-coverage.md"))
    if not report["validation"]["ok"]:
        sys.stderr.write("error: report failed self validation\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
