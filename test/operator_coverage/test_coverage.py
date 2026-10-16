#!/usr/bin/env python3
"""Regression tests for the operator coverage tool.

Run with::

    python3 test/operator_coverage/test_coverage.py

The tests use hand-encoded protobuf messages and temporary ONNX/onnx2c
fixtures, so they run without the submodule, the compiler, or any third-party
package.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coverage as cov


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field, wire):
    return _varint((field << 3) | wire)


def _varint_field(field, value):
    return _tag(field, 0) + _varint(value)


def _bytes_field(field, data):
    return _tag(field, 2) + _varint(len(data)) + data


def _string_field(field, text):
    return _bytes_field(field, text.encode("utf-8"))


def _node(op_type, domain=""):
    body = _string_field(4, op_type)
    if domain:
        body += _string_field(7, domain)
    return body


def _model_bytes(nodes, extra=b""):
    graph = b"".join(_bytes_field(1, node) for node in nodes)
    return _bytes_field(7, graph) + extra


def _case(operators, outcome, mode="split"):
    keys = sorted(set("%s:%s" % (domain, op) for domain, op in operators))
    return {"id": "t/d", "test": "t", "dataset": "d", "operators": keys,
            "mode": mode, "result": outcome, "stage": "execution",
            "exit_code": 0, "detail": ""}


class ProtobufTest(unittest.TestCase):
    def _write(self, directory, name, data):
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_parses_operators_and_normalizes_default_domain(self):
        data = b"".join([
            _bytes_field(1, _bytes_field(4, b"Abs")),
            _bytes_field(1, _bytes_field(4, b"Abs")),
            _bytes_field(1, _bytes_field(4, b"Conv") + _bytes_field(7, b"")),
        ])
        model = _bytes_field(7, data)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "model.onnx", model)
            self.assertEqual(cov.parse_model_operators(path),
                             ["ai.onnx:Abs", "ai.onnx:Conv"])

    def test_keeps_contributed_domain(self):
        node = _bytes_field(4, b"QLinearAdd") + _bytes_field(
            7, b"com.microsoft")
        model = _bytes_field(7, _bytes_field(1, node))
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "model.onnx", model)
            self.assertEqual(cov.parse_model_operators(path),
                             ["com.microsoft:QLinearAdd"])

    def test_rejects_truncated_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "model.onnx", _bytes_field(7, b"\x0a\x05\x01"))
            with self.assertRaises(cov.MalformedModel):
                cov.parse_model_operators(path)


class SpecTest(unittest.TestCase):
    def _write(self, directory, name, text):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_reads_max_version_per_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(directory=tmp, name="operator_sets.h", text="\n".join([
                "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Abs);",
                "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Abs);",
                "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Add);",
                "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, TreeEnsemble);",
            ]))
            spec = cov.read_spec(tmp)
        self.assertEqual(spec, {"Abs": 13, "Add": 7})

    def test_missing_header_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                cov.read_spec(tmp)

    def test_only_the_latest_schema_can_deprecate_an_operator(self):
        source = "\n".join([
            "ONNX_OPERATOR_SET_SCHEMA(",
            "    Scatter,",
            "    11,",
            "    OpSchema().Deprecate());",
            "ONNX_OPERATOR_SET_SCHEMA(",
            "    GroupNormalization,",
            "    18,",
            "    OpSchema().Deprecate());",
            "ONNX_OPERATOR_SET_SCHEMA(",
            "    GroupNormalization,",
            "    21,",
            "    OpSchema());",
        ])
        spec = {"Scatter": 11, "GroupNormalization": 21, "Abs": 13}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "defs.cc"), "w",
                      encoding="utf-8") as handle:
                handle.write(source)
            self.assertEqual(cov.read_deprecated(tmp, spec), {"Scatter"})


class RegistrationTest(unittest.TestCase):
    def test_reads_registered_operators_from_create_node_only(self):
        source = "\n".join([
            "Node* Graph::createNode(const onnx::NodeProto& onnx_node) {",
            '\tif (opName == "Abs") return new Elementwise("Abs");',
            '\tif (opName == "Abs") return new Elementwise("Abs");',
            '\tif (opName == "Conv") return new Conv;',
            "}",
            "",
            "void helper() {",
            '\tif (opName == "NotRegistered") {}',
            "}",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "graph.cc")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            self.assertEqual(cov.read_registered_operators(path),
                             ["Abs", "Conv"])

    def test_reads_modes_and_accuracy(self):
        source = "\n".join([
            "ONNX_backend_node_test(add)",
            "ONNX_backend_node_test_singlefile(batchnorm_example)",
            "ONNX_backend_node_test_with_accuracy(lrn 0.005)",
            "# ONNX_backend_node_test(round)",
            "ONNX_backend_pytorch_converted_test(some_model)",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "CMakeLists.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            registrations = cov.read_registrations(path)
        self.assertEqual(registrations["add"], (cov.MODE_SPLIT,
                                                cov.DEFAULT_ACCURACY))
        self.assertEqual(registrations["batchnorm_example"],
                         (cov.MODE_SINGLEFILE, cov.DEFAULT_ACCURACY))
        self.assertEqual(registrations["lrn"],
                         (cov.MODE_SPLIT, 0.005))
        self.assertEqual(registrations["round"],
                         (cov.MODE_SPLIT, cov.DEFAULT_ACCURACY))
        self.assertNotIn("some_model", registrations)


class DiscoveryTest(unittest.TestCase):
    def test_builds_cases_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_dir = os.path.join(tmp, "test_abs")
            os.makedirs(os.path.join(test_dir, "test_data_set_0"))
            model = _bytes_field(7, _bytes_field(1, _bytes_field(4, b"Abs")))
            with open(os.path.join(test_dir, "model.onnx"), "wb") as handle:
                handle.write(model)
            os.makedirs(os.path.join(tmp, "test_relu"))
            with open(os.path.join(tmp, "test_relu", "model.onnx"),
                      "wb") as handle:
                handle.write(_bytes_field(7, _bytes_field(1, _bytes_field(
                    4, b"Relu"))))
            cases = cov.discover_cases(tmp, {"abs": (cov.MODE_SINGLEFILE,
                                                     0.005)})
        by_id = {case["id"]: case for case in cases}
        self.assertEqual(set(by_id),
                         {"test_abs/test_data_set_0",
                          "test_relu/test_data_set_0"})
        self.assertEqual(by_id["test_abs/test_data_set_0"]["mode"],
                         cov.MODE_SINGLEFILE)
        self.assertEqual(by_id["test_abs/test_data_set_0"]["accuracy"], 0.005)
        self.assertEqual(by_id["test_relu/test_data_set_0"]["mode"],
                         cov.MODE_SPLIT)
        self.assertEqual(by_id["test_abs/test_data_set_0"]["operators"],
                         ["ai.onnx:Abs"])


class ClassificationTest(unittest.TestCase):
    def test_rules_are_mutually_exclusive(self):
        classify = cov.classify
        self.assertEqual(classify(False, 0, 0, 0, 0), cov.CAT_MISSING)
        self.assertEqual(classify(True, 0, 0, 0, 0), cov.CAT_UNTESTED)
        self.assertEqual(classify(True, 3, 0, 2, 0), cov.CAT_FAILING)
        self.assertEqual(classify(True, 3, 1, 2, 0), cov.CAT_PARTIAL)
        self.assertEqual(classify(True, 3, 3, 0, 0), cov.CAT_PASSING)
        self.assertEqual(classify(True, 3, 0, 0, 2), cov.CAT_BLOCKED)


class AggregationTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"Abs": 13, "Conv": 11, "Relu": 14, "Scatter": 11,
                     "Upsample": 10}
        self.registered = ["Abs", "Conv", "Relu", "Upsample"]
        self.deprecated = {"Scatter", "Upsample"}

    def _aggregate(self, results):
        return cov.aggregate(self.spec, self.registered, results,
                             self.deprecated)

    def test_status_transitions(self):
        rows, totals = self._aggregate([
            _case([("ai.onnx", "Abs")], cov.PASS),
            _case([("ai.onnx", "Conv")], cov.PASS),
            _case([("ai.onnx", "Conv")], cov.NUMERICAL_MISMATCH),
            _case([("ai.onnx", "Relu")], cov.ONNX_TO_C_FAILURE),
            _case([("com.microsoft", "QLinearAdd")], cov.PASS),
        ])
        by_name = {row["operator"]: row for row in rows}
        self.assertEqual(by_name["Abs"]["status"], cov.CAT_PASSING)
        self.assertEqual(by_name["Conv"]["status"], cov.CAT_PARTIAL)
        self.assertEqual(by_name["Relu"]["status"], cov.CAT_FAILING)
        self.assertEqual(totals["spec_operators_active"], 3)
        self.assertEqual(totals["registered_active"], 3)
        self.assertEqual(totals["active_verified"], 2)

    def test_shared_failure_never_blames_an_operator(self):
        rows, _totals = self._aggregate([
            _case([("ai.onnx", "Abs")], cov.PASS),
            _case([("ai.onnx", "Abs"), ("ai.onnx", "Conv")],
                  cov.NUMERICAL_MISMATCH),
        ])
        by_name = {row["operator"]: row for row in rows}
        self.assertEqual(by_name["Abs"]["status"], cov.CAT_PASSING)
        self.assertEqual(by_name["Abs"]["applicable_cases"], 2)
        self.assertEqual(by_name["Abs"]["solo_cases"], 1)
        self.assertEqual(by_name["Conv"]["status"], cov.CAT_UNTESTED)

    def test_blocked_when_solo_test_is_unrunnable(self):
        rows, _totals = self._aggregate([
            _case([("ai.onnx", "Abs")], cov.INFRASTRUCTURE_ERROR)])
        by_name = {row["operator"]: row for row in rows}
        self.assertEqual(by_name["Abs"]["status"], cov.CAT_BLOCKED)

    def test_deprecated_operators_leave_denominators(self):
        _rows, totals = self._aggregate([
            _case([("ai.onnx", "Abs")], cov.PASS),
            _case([("ai.onnx", "Upsample")], cov.PASS),
            _case([("ai.onnx", "Scatter")], cov.ONNX_TO_C_FAILURE),
        ])
        self.assertEqual(totals["spec_operators_active"], 3)
        self.assertEqual(totals["spec_operators_deprecated"], 2)
        self.assertEqual(totals["registered_active"], 3)
        self.assertEqual(totals["deprecated_only_cases"], 2)
        self.assertEqual(totals["active_cases"], 1)
        self.assertEqual(totals["passed_cases"], 2)
        self.assertEqual(totals["active_passed_cases"], 1)
        self.assertEqual(totals["deprecated_verified"], 1)

    def test_mixed_deprecated_case_stays_active(self):
        _rows, totals = self._aggregate([
            _case([("ai.onnx", "Abs"), ("ai.onnx", "Scatter")], cov.PASS)])
        self.assertEqual(totals["active_cases"], 1)
        self.assertEqual(totals["deprecated_only_cases"], 0)


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"Abs": 13, "Conv": 11}
        self.registered = ["Abs"]
        self.deprecated = set()
        self.campaign = {"complete": True, "onnx2c_commit": "abc",
                         "onnx_version": "1.19.0", "spec": {"opset": 24},
                         "deprecated_operators": []}

    def _report(self, results):
        return cov.build_report(self.campaign, self.spec, self.registered,
                                results, self.deprecated)

    def test_self_validation_passes(self):
        report = self._report([_case([("ai.onnx", "Abs")], cov.PASS)])
        self.assertTrue(report["validation"]["ok"], report["validation"])

    def test_validation_detects_tampered_totals(self):
        report = self._report([_case([("ai.onnx", "Abs")], cov.PASS)])
        report["totals"]["passed_cases"] = 99
        self.assertTrue(cov.validate(report))

    def test_badges_are_shields_endpoints(self):
        report = self._report([_case([("ai.onnx", "Abs")], cov.PASS)])
        badges = cov.badge_payloads(report)
        self.assertEqual(set(badges),
                         {"registered.json", "verified.json", "tests.json"})
        self.assertIn("1/2", badges["registered.json"]["message"])
        for payload in badges.values():
            self.assertEqual(payload["schemaVersion"], 1)

    def test_markdown_renders_compact_overview(self):
        report = self._report([_case([("ai.onnx", "Conv")],
                                     cov.ONNX_TO_C_FAILURE)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            cov.write_markdown(report, path)
            with open(path, encoding="utf-8") as handle:
                markdown = handle.read()
        self.assertIn("# ONNX operator compatibility", markdown)
        self.assertIn("| Metric | Value |", markdown)
        self.assertIn("## Operator compatibility", markdown)
        self.assertIn("## Failure details", markdown)
        self.assertIn("ONNX\u2192C", markdown)
        self.assertIn("## Coverage limitations", markdown)
        self.assertIn("| Conv |", markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
