"""Discovery of the three inputs to the coverage report:

* the operators defined by the pinned ONNX specification (the denominator),
* the operators registered by onnx2c (from ``Graph::createNode``),
* the ONNX backend node test corpus (the individual test cases).
"""

import glob
import os
import re

import onnx_proto

SPEC_FILES = (
    ("operator_sets.h", "CLASS_NAME", "Onnx", onnx_proto.DEFAULT_DOMAIN),
    ("operator_sets_ml.h", "CLASS_NAME", "OnnxML", "ai.onnx.ml"),
    ("operator_sets_preview.h", "PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME", None,
     "ai.onnx.preview.training"),
)

_CLASS_NAME_RE = re.compile(
    r"CLASS_NAME\(\s*(Onnx|OnnxML)\s*,\s*(\d+)\s*,\s*([A-Za-z0-9_]+)\s*\)")
_PREVIEW_CLASS_NAME_RE = re.compile(
    r"PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME\(\s*(\d+)\s*,\s*([A-Za-z0-9_]+)\s*\)")
_REGISTERED_RE = re.compile(r'opName\s*==\s*"([A-Za-z0-9_]+)"')
_CREATE_NODE_RE = re.compile(r"Graph::createNode\b.*?(?=\n}\n)", re.DOTALL)

# How upstream wires a backend test into CTest (test/CMakeLists.txt).
MODE_SPLIT = "split"
MODE_SINGLEFILE = "singlefile"
STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"
STATUS_ABSENT = "absent"
DEFAULT_ACCURACY = 2e-05
DEFAULT_REGISTRATION = {
    "mode": MODE_SPLIT,
    "accuracy": DEFAULT_ACCURACY,
    "status": STATUS_ABSENT,
    "family": "node",
}

_REGISTRATION_RE = re.compile(
    r"^(ONNX_backend_node_test(?:_singlefile|_with_accuracy)?"
    r"|ONNX_backend_pytorch_converted_test|local_node_test)"
    r"\s*\(\s*([A-Za-z0-9_]+)\s*(.*?)\)")
_ACCURACY_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def read_spec_operators(defs_dir):
    """Return ``{domain: {operator_name: max_version}}`` from the ONNX headers."""
    domains = {}
    for filename, _macro, _prefix, domain in SPEC_FILES:
        path = os.path.join(defs_dir, filename)
        if not os.path.exists(path):
            continue
        text = _read_text(path)
        table = domains.setdefault(domain, {})
        if filename == "operator_sets_preview.h":
            matches = _PREVIEW_CLASS_NAME_RE.findall(text)
            for version, name in matches:
                table[name] = max(table.get(name, 0), int(version))
        else:
            for _prefix_match, version, name in _CLASS_NAME_RE.findall(text):
                table[name] = max(table.get(name, 0), int(version))
    return domains


def read_registered_operators(graph_cc_path):
    """Return the ordered, de-duplicated operator names dispatched by onnx2c.

    Registration in onnx2c is a flat ``if (opName == "...")`` chain inside
    ``Graph::createNode``; the function body is isolated first so that
    comparisons inside node implementations are never mistaken for
    registrations.
    """
    text = _read_text(graph_cc_path)
    match = _CREATE_NODE_RE.search(text)
    body = match.group(0) if match else text
    seen = []
    for name in _REGISTERED_RE.findall(body):
        if name not in seen:
            seen.append(name)
    return seen


def read_upstream_registrations(cmake_path):
    """Return ``{test_name: registration}`` from ``test/CMakeLists.txt``.

    Upstream runs a subset of the ONNX backend corpus through CTest, and it
    does so in two different ways:

    * ``ONNX_backend_node_test`` links an ``onnx2c`` generated translation
      with a ``testgen`` harness (the *split* mode used by real users),
    * ``ONNX_backend_node_test_singlefile`` asks ``testgen_singlefile`` for
      graph *and* harness in one file (the *singlefile* mode).

    Tests registered only as comments (known-broken upstream) come back with
    ``status == "disabled"``; tests that CMake never mentions are absent.

    Matching both modes and the per-test accuracy keeps the campaign
    comparable with upstream's own CTest results.
    """
    registrations = {}
    if not os.path.exists(cmake_path):
        return registrations
    for line in _read_text(cmake_path).splitlines():
        stripped = line.strip()
        active = not stripped.startswith("#")
        body = stripped.lstrip("#").strip() if not active else stripped
        match = _REGISTRATION_RE.match(body)
        if not match:
            continue
        helper, name, rest = match.groups()
        if helper == "ONNX_backend_node_test_singlefile":
            mode, family = MODE_SINGLEFILE, "node"
        elif helper == "ONNX_backend_pytorch_converted_test":
            mode, family = MODE_SINGLEFILE, "pytorch-converted"
        elif helper == "local_node_test":
            mode, family = MODE_SINGLEFILE, "local_ops"
        else:
            mode, family = MODE_SPLIT, "node"
        accuracy = DEFAULT_ACCURACY
        if helper == "ONNX_backend_node_test_with_accuracy":
            accuracy_match = _ACCURACY_RE.match(rest)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
        registrations[name] = {
            "mode": mode,
            "accuracy": accuracy,
            "status": STATUS_ACTIVE if active else STATUS_DISABLED,
            "family": family,
        }
    return registrations


def upstream_registration(registrations, test_dir_name):
    """Registration for a corpus directory such as ``test_round``."""
    name = test_dir_name
    if name.startswith("test_"):
        name = name[len("test_"):]
    registration = registrations.get(name)
    if registration is None:
        return dict(DEFAULT_REGISTRATION)
    return dict(registration)


def missing_upstream_cases(registrations, node_test_data_dir):
    """Active ``node`` registrations with no directory in the node corpus."""
    missing = []
    for name, registration in sorted(registrations.items()):
        if registration["status"] != STATUS_ACTIVE:
            continue
        if registration["family"] != "node":
            continue
        if not os.path.isdir(os.path.join(node_test_data_dir, "test_" + name)):
            missing.append(name)
    return missing


def discover_test_directories(node_test_data_dir):
    """Return sorted ``(name, path)`` pairs for every backend node test."""
    entries = []
    for path in glob.glob(os.path.join(node_test_data_dir, "test_*")):
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "model.onnx")):
            entries.append((os.path.basename(path), path))
    return sorted(entries)


def discover_datasets(test_dir):
    """Return the sorted ``test_data_set_*`` directory names inside a test."""
    names = []
    for path in glob.glob(os.path.join(test_dir, "test_data_set_*")):
        if os.path.isdir(path):
            names.append(os.path.basename(path))
    return sorted(names)


def analyse_model(model_path):
    """Return operator/opset metadata for one ONNX model file."""
    model = onnx_proto.parse_model_file(model_path)
    operators = [(domain, op_type) for domain, op_type in model.operators()
                 if op_type]
    opsets = {domain: version for domain, version in model.opset_imports}
    return {
        "operators": operators,
        "opsets": opsets,
        "node_count": len(model.nodes),
        "operator_count": len(operators),
    }


def build_test_cases(node_test_data_dir, registrations=None):
    """Return every ``(test_name, dataset)`` case plus its model metadata.

    ``registrations`` (see :func:`read_upstream_registrations`) records how
    upstream wires each test into CTest; cases without a registration fall
    back to :data:`DEFAULT_REGISTRATION` (split mode, default accuracy).
    """
    cases = []
    for name, path in discover_test_directories(node_test_data_dir):
        upstream = upstream_registration(registrations or {}, name)
        datasets = discover_datasets(path)
        if not datasets:
            datasets = ["test_data_set_0"]
        try:
            meta = analyse_model(os.path.join(path, "model.onnx"))
        except (OSError, onnx_proto.MalformedProtobuf) as exc:
            meta = {"error": str(exc), "operators": [], "opsets": {},
                    "node_count": 0, "operator_count": 0}
        for dataset in datasets:
            cases.append({
                "test": name,
                "dataset": dataset,
                "dataset_index": _dataset_index(dataset),
                "dir": path,
                "model_path": os.path.join(path, "model.onnx"),
                "upstream": upstream,
                "meta": meta,
            })
    return cases


def _dataset_index(dataset):
    """``test_data_set_3`` -> ``3`` (the index testgen expects)."""
    prefix = "test_data_set_"
    if dataset.startswith(prefix):
        return dataset[len(prefix):]
    return dataset
