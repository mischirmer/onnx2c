"""Execute the ONNX backend test corpus against onnx2c.

Each case is run through four stages, mirroring the existing CTest
infrastructure (``test/CMakeLists.txt``):

1. ``onnx2c`` translates ``model.onnx`` into C,
2. ``testgen`` builds a self-checking harness from the reference data,
3. the C compiler links harness and generated network into a binary,
4. the binary runs and compares its output against the reference.

Upstream registers some tests as *singlefile*: ``testgen_singlefile`` emits
graph and harness in one file and ``onnx2c`` is never involved.  Each case
therefore runs in the mode upstream registered it in, so campaign results
stay comparable with upstream's own CTest results.  ``cross_check`` re-runs
the same case in the other mode and records the outcome; divergence between
the two modes is reported, never used to reclassify the primary result.

The first stage that fails decides the case's outcome.
"""

import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from discovery import MODE_SINGLEFILE, MODE_SPLIT

PASS = "pass"
ONNX_TO_C_FAILURE = "onnx_to_c_failure"
C_COMPILATION_FAILURE = "c_compilation_failure"
EXECUTION_FAILURE = "execution_failure"
NUMERICAL_MISMATCH = "numerical_mismatch"
TIMEOUT = "timeout"
INFRASTRUCTURE_ERROR = "infrastructure_error"

FAILURE_CATEGORIES = (
    ONNX_TO_C_FAILURE,
    C_COMPILATION_FAILURE,
    EXECUTION_FAILURE,
    NUMERICAL_MISMATCH,
    TIMEOUT,
    INFRASTRUCTURE_ERROR,
)

UNRUNNABLE_CATEGORIES = (TIMEOUT, INFRASTRUCTURE_ERROR)

STAGE_TRANSLATE = "onnx_to_c"
STAGE_HARNESS = "harness_generation"
STAGE_COMPILE = "c_compilation"
STAGE_EXECUTE = "execution"

_DETAIL_LIMIT = 500

DEFAULT_CC_FLAGS = ("-Wall", "-Wno-unused-variable")

# Failure signatures that describe the test *tooling*, not onnx2c.
_TOOLING_ERROR_MARKERS = (
    "unimplemented type",
    "unhandled tensor data type",
)
_ENVIRONMENT_ERROR_MARKERS = (
    "could not execute",
    "Error opening",
    "No such file",
)

_ENTRY_RE = re.compile(r"void\s+entry\s*\((.*?)\)\s*\{", re.DOTALL)
_SHAPE_RE = re.compile(r"\[[0-9]+\]")
_GRAPHOUT_RE = re.compile(
    r"graphout_\w+\s*((?:\[[0-9]+\])+)", re.MULTILINE)


def _tail(data):
    if not data:
        return ""
    text = data.decode("utf-8", "replace").strip()
    if len(text) > _DETAIL_LIMIT:
        text = "..." + text[-_DETAIL_LIMIT:]
    return text


def _run(command, stdout_path, timeout):
    """Run a command, streaming stdout to a file.

    Returns ``(returncode, signal, stderr, timed_out)``.  ``returncode`` is
    ``None`` when the timeout elapsed.
    """
    try:
        with open(stdout_path, "w") as stdout_handle:
            completed = subprocess.run(
                command,
                stdout=stdout_handle,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        return None, None, _tail(exc.stderr), True
    except OSError as exc:
        return None, None, "could not execute %s: %s" % (command[0], exc), False
    returncode = completed.returncode
    signal = -returncode if returncode is not None and returncode < 0 else None
    return returncode, signal, _tail(completed.stderr), False


def _sanitize(name):
    return "".join(char if char.isalnum() or char in "-_." else "_"
                   for char in name)


def _result(case, outcome, stage, returncode=None, signal=None, detail="",
            duration=0.0, generated_paths=None, accuracy=None, mode=None):
    meta = case["meta"]
    operators = meta.get("operators", [])
    opsets = meta.get("opsets", {})
    default_opset = opsets.get("ai.onnx")
    upstream = case.get("upstream") or {}
    return {
        "id": "%s/%s" % (case["test"], case["dataset"]),
        "test": case["test"],
        "dataset": case["dataset"],
        "operators": ["%s:%s" % (domain, op) for domain, op in operators],
        "operator_names": sorted({op for _domain, op in operators}),
        "domains": sorted({domain for domain, _op in operators}),
        "node_count": meta.get("node_count", 0),
        "operator_count": meta.get("operator_count", 0),
        "multi_operator": meta.get("operator_count", 0) > 1,
        "opset": default_opset,
        "upstream_status": upstream.get("status", "absent"),
        "upstream_mode": upstream.get("mode", MODE_SPLIT),
        "upstream_accuracy": upstream.get("accuracy"),
        "mode": mode,
        "accuracy": accuracy,
        "result": outcome,
        "stage": stage,
        "exit_code": returncode,
        "signal": signal,
        "detail": detail,
        "root_cause": None,
        "cross_check": None,
        "duration_seconds": round(duration, 4),
        "artifacts": generated_paths or [],
    }


def _matches(markers, detail):
    return any(marker in detail for marker in markers)


def _diagnose(outcome, stage, detail):
    """Root cause from the failed stage and its recorded output.

    Only messages actually produced by the tools are turned into a root
    cause; anything else stays ``None`` so the raw detail remains the
    primary evidence.
    """
    if outcome == INFRASTRUCTURE_ERROR:
        if detail and "unimplemented type" in detail:
            return ("testgen cannot emit reference-comparison code for this "
                    "tensor data type")
        if detail and "unhandled tensor data type" in detail:
            return ("Tensor::parse_onnx_tensor has no loader for a data type "
                    "used by this test's tensors")
        if stage == STAGE_HARNESS:
            return "testgen failed to generate the test harness"
    if outcome == C_COMPILATION_FAILURE:
        return "generated C does not compile with the campaign compiler flags"
    return None


def _classify_singlefile_generation(detail):
    """Classify a ``testgen_singlefile`` failure (graph *and* harness)."""
    if _matches(_TOOLING_ERROR_MARKERS, detail) or \
            _matches(_ENVIRONMENT_ERROR_MARKERS, detail):
        return INFRASTRUCTURE_ERROR, STAGE_HARNESS
    return ONNX_TO_C_FAILURE, STAGE_TRANSLATE


def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _entry_output_shape(text):
    match = _ENTRY_RE.search(text)
    if not match:
        return None
    params = match.group(1).split(",")
    if not params or not params[-1].strip():
        return None
    return "".join(_SHAPE_RE.findall(params[-1])) or "[]"


def _reference_output_shape(text):
    match = _GRAPHOUT_RE.search(text)
    return match.group(1) if match else None


def _shape_mismatch(case_dir, mode):
    """Compare the generated network's output shape with the reference's."""
    generated = _read(os.path.join(case_dir, "generated.c"))
    harness = (_read(os.path.join(case_dir, "harness.c"))
               if mode == MODE_SPLIT else generated)
    generated_shape = _entry_output_shape(generated)
    reference_shape = _reference_output_shape(harness)
    if not generated_shape or not reference_shape:
        return None
    if generated_shape == reference_shape:
        return None
    return generated_shape, reference_shape


def _run_split(case, tools, case_dir, timeout, accuracy, cc_flags):
    """onnx2c translation + testgen harness, as CTest's split mode does."""
    generated_c = os.path.join(case_dir, "generated.c")
    harness_c = os.path.join(case_dir, "harness.c")
    binary = os.path.join(case_dir, "case")

    returncode, signal, detail, timed_out = _run(
        [tools["onnx2c"], "-l", "0", case["model_path"]], generated_c, timeout)
    if timed_out:
        return TIMEOUT, STAGE_TRANSLATE, None, None, detail
    if returncode != 0 or signal is not None:
        return ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode, signal, detail
    if os.path.getsize(generated_c) == 0:
        return (ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode, None,
                "onnx2c produced no output")

    returncode, signal, detail, timed_out = _run(
        [tools["testgen"], case["dir"], "%.10g" % accuracy,
         case.get("dataset_index", case["dataset"])],
        harness_c, timeout)
    if timed_out:
        return TIMEOUT, STAGE_HARNESS, None, None, detail
    if returncode != 0 or signal is not None:
        return INFRASTRUCTURE_ERROR, STAGE_HARNESS, returncode, signal, detail

    return _compile_and_execute(tools, case_dir, [harness_c, generated_c],
                                binary, timeout, cc_flags)


def _run_singlefile(case, tools, case_dir, timeout, accuracy, cc_flags):
    """testgen_singlefile emits graph and harness, as CTest's singlefile does."""
    binary = os.path.join(case_dir, "case")
    generator = tools.get("testgen_singlefile")
    if not generator:
        return (INFRASTRUCTURE_ERROR, STAGE_HARNESS, None, None,
                "testgen_singlefile binary not available")

    generated_c = os.path.join(case_dir, "generated.c")
    returncode, signal, detail, timed_out = _run(
        [generator, case["dir"], "%.10g" % accuracy,
         case.get("dataset_index", case["dataset"])],
        generated_c, timeout)
    if timed_out:
        return TIMEOUT, STAGE_HARNESS, None, None, detail
    if returncode != 0 or signal is not None:
        outcome, stage = _classify_singlefile_generation(detail)
        return outcome, stage, returncode, signal, detail
    if os.path.getsize(generated_c) == 0:
        return (ONNX_TO_C_FAILURE, STAGE_TRANSLATE, returncode, None,
                "testgen_singlefile produced no output")

    return _compile_and_execute(tools, case_dir, [generated_c], binary,
                                timeout, cc_flags)


def _compile_and_execute(tools, case_dir, sources, binary, timeout, cc_flags):
    compile_command = [tools["cc"]] + list(cc_flags) + list(sources) + [
        "-o", binary, "-lm"]
    returncode, signal, detail, timed_out = _run(
        compile_command, os.path.join(case_dir, "compile.log"), timeout)
    if timed_out:
        return TIMEOUT, STAGE_COMPILE, None, None, detail
    if returncode != 0 or signal is not None:
        return C_COMPILATION_FAILURE, STAGE_COMPILE, returncode, signal, detail

    returncode, signal, detail, timed_out = _run(
        [binary], os.path.join(case_dir, "run.log"), timeout)
    if timed_out:
        return TIMEOUT, STAGE_EXECUTE, None, None, detail
    if signal is not None:
        return EXECUTION_FAILURE, STAGE_EXECUTE, None, signal, detail
    if returncode == 0:
        return PASS, STAGE_EXECUTE, 0, None, detail
    if returncode == 1:
        return (NUMERICAL_MISMATCH, STAGE_EXECUTE, 1, None,
                detail or "reference comparison failed")
    return EXECUTION_FAILURE, STAGE_EXECUTE, returncode, None, detail


def _run_mode(case, tools, case_dir, mode, timeout, accuracy, cc_flags):
    if mode == MODE_SINGLEFILE:
        return _run_singlefile(case, tools, case_dir, timeout, accuracy,
                               cc_flags)
    return _run_split(case, tools, case_dir, timeout, accuracy, cc_flags)


def run_case(case, tools, work_root, timeout, accuracy, cc_flags,
             cross_check=False):
    """Run one ``(test, dataset)`` case in its upstream-registered mode.

    With ``cross_check`` the same case is additionally executed in the other
    mode; the outcome is stored under ``cross_check`` for reporting and never
    replaces the primary result.
    """
    case_dir = os.path.join(work_root, _sanitize(case["test"]),
                            _sanitize(case["dataset"]))
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    upstream = case.get("upstream") or {}
    mode = upstream.get("mode") or MODE_SPLIT
    case_accuracy = upstream.get("accuracy")
    if case_accuracy is None:
        case_accuracy = accuracy

    started = time.monotonic()
    outcome, stage, returncode, signal, detail = _run_mode(
        case, tools, case_dir, mode, timeout, case_accuracy, cc_flags)
    duration = time.monotonic() - started

    result = _result(case, outcome, stage, returncode=returncode,
                     signal=signal, detail=detail, duration=duration,
                     accuracy=case_accuracy, mode=mode)
    result["root_cause"] = _diagnose(outcome, stage, detail)
    if outcome == NUMERICAL_MISMATCH:
        shapes = _shape_mismatch(case_dir, mode)
        if shapes:
            result["root_cause"] = (
                "generated output shape %s differs from reference shape %s"
                % shapes)

    if cross_check and outcome != TIMEOUT:
        result["cross_check"] = _cross_check(
            case, tools, case_dir, mode, timeout, case_accuracy, cc_flags)
    return result


def _cross_check(case, tools, case_dir, mode, timeout, accuracy, cc_flags):
    other = MODE_SPLIT if mode == MODE_SINGLEFILE else MODE_SINGLEFILE
    other_dir = os.path.join(case_dir, "alt_" + other)
    os.makedirs(other_dir, exist_ok=True)
    started = time.monotonic()
    outcome, stage, returncode, signal, detail = _run_mode(
        case, tools, other_dir, other, timeout, accuracy, cc_flags)
    return {
        "mode": other,
        "result": outcome,
        "stage": stage,
        "exit_code": returncode,
        "signal": signal,
        "detail": detail,
        "duration_seconds": round(time.monotonic() - started, 4),
    }


def run_campaign(cases, tools, work_root, jobs=1, timeout=60.0, accuracy=2e-05,
                 cc_flags=DEFAULT_CC_FLAGS, keep_work=False, progress=None,
                 cross_check=False):
    """Run every case, optionally in parallel; return the ordered results."""
    os.makedirs(work_root, exist_ok=True)
    results = [None] * len(cases)
    keep_dir = {}

    def work(index):
        case = cases[index]
        result = run_case(case, tools, work_root, timeout, accuracy, cc_flags,
                          cross_check=cross_check)
        case_dir = os.path.join(work_root, _sanitize(case["test"]),
                                _sanitize(case["dataset"]))
        if not keep_work and result["result"] == PASS:
            shutil.rmtree(case_dir, ignore_errors=True)
        elif os.path.isdir(case_dir):
            keep_dir[index] = case_dir
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

    for index, case_dir in keep_dir.items():
        results[index]["artifacts"] = [case_dir]
    return results
