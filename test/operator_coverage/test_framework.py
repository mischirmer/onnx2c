"""Regression tests for the operator coverage framework itself.

Run with::

    python3 test/operator_coverage/test_framework.py

The tests exercise the protobuf reader against hand-encoded messages, the
three discovery sources, the runner's outcome mapping, and the report's
aggregation/validation.  An end-to-end case is run only when onnx2c and
testgen binaries are available.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import discovery
import onnx_proto
import report as report_module
import runner


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


def _node(op_type, domain="", name="", inputs=(), outputs=()):
    body = b""
    for value in inputs:
        body += _string_field(1, value)
    for value in outputs:
        body += _string_field(2, value)
    if name:
        body += _string_field(3, name)
    body += _string_field(4, op_type)
    if domain:
        body += _string_field(7, domain)
    return body


def _model_bytes(nodes, opsets=(("", 13),), extra=b""):
    graph = b"".join(_bytes_field(1, node) for node in nodes)
    model = _bytes_field(7, graph) + extra
    for domain, version in opsets:
        opset = _string_field(1, domain) + _varint_field(2, version)
        model += _bytes_field(8, opset)
    return model


class ProtobufReaderTest(unittest.TestCase):
    def test_parses_nodes_domains_and_opsets(self):
        data = _model_bytes([
            _node("Abs", outputs=("Y",)),
            _node("QLinearAdd", domain="com.microsoft", outputs=("Z",)),
        ], opsets=(("", 17), ("com.microsoft", 1)))
        model = onnx_proto.parse_model(data)
        self.assertEqual(model.opset(), 17)
        self.assertEqual([n.op_type for n in model.nodes],
                         ["Abs", "QLinearAdd"])
        self.assertEqual(model.nodes[1].domain, "com.microsoft")
        self.assertEqual(model.operators(),
                         [("ai.onnx", "Abs"), ("com.microsoft", "QLinearAdd")])

    def test_default_and_ai_onnx_domains_are_equivalent(self):
        self.assertEqual(onnx_proto.normalize_domain(""), "ai.onnx")
        self.assertEqual(onnx_proto.normalize_domain("ai.onnx"), "ai.onnx")
        self.assertEqual(onnx_proto.normalize_domain("ai.onnx.ml"),
                         "ai.onnx.ml")

    def test_skips_unknown_fields_of_every_wire_type(self):
        extra = (_varint_field(3, 99) + _bytes_field(5, b"\x01\x02")
                 + _tag(4, 5) + b"\x00\x00\x00\x00")
        model = onnx_proto.parse_model(_model_bytes([_node("Relu")],
                                                    extra=extra))
        self.assertEqual(model.opset(), 13)
        self.assertEqual(model.nodes[0].op_type, "Relu")

    def test_rejects_truncated_message(self):
        with self.assertRaises(onnx_proto.MalformedProtobuf):
            onnx_proto.parse_model(_bytes_field(7, b"\x0a\x05\x01"))


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_reads_spec_domains_and_max_versions(self):
        self._write("operator_sets.h", "\n".join([
            "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Abs);",
            "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Abs);",
            "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Add);",
        ]))
        self._write("operator_sets_ml.h",
                    "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, TreeEnsembleClassifier);")
        spec = discovery.read_spec_operators(self.tmp)
        self.assertEqual(spec["ai.onnx"]["Abs"], 13)
        self.assertEqual(spec["ai.onnx"]["Add"], 7)
        self.assertEqual(set(spec["ai.onnx.ml"]), {"TreeEnsembleClassifier"})

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
        path = self._write("graph.cc", source)
        self.assertEqual(discovery.read_registered_operators(path),
                         ["Abs", "Conv"])

    def test_discovers_test_directories_and_datasets(self):
        test_dir = os.path.join(self.tmp, "test_abs")
        os.makedirs(os.path.join(test_dir, "test_data_set_0"))
        with open(os.path.join(test_dir, "model.onnx"), "wb") as handle:
            handle.write(_model_bytes([_node("Abs")]))
        entries = discovery.discover_test_directories(self.tmp)
        self.assertEqual([name for name, _path in entries], ["test_abs"])
        self.assertEqual(discovery.discover_datasets(test_dir),
                         ["test_data_set_0"])

    def test_reads_upstream_registration_modes_and_accuracy(self):
        path = self._write("CMakeLists.txt", "\n".join([
            "ONNX_backend_node_test(add)",
            "ONNX_backend_node_test_singlefile(batchnorm_example)",
            "ONNX_backend_node_test_with_accuracy(lrn 0.005)",
            "# ONNX_backend_node_test(round)",
            "ONNX_backend_pytorch_converted_test(some_model)",
            "local_node_test(my_local)",
        ]))
        regs = discovery.read_upstream_registrations(path)
        self.assertEqual(regs["add"]["mode"], discovery.MODE_SPLIT)
        self.assertEqual(regs["add"]["status"], discovery.STATUS_ACTIVE)
        self.assertEqual(regs["batchnorm_example"]["mode"],
                         discovery.MODE_SINGLEFILE)
        self.assertEqual(regs["lrn"]["accuracy"], 0.005)
        self.assertEqual(regs["round"]["status"], discovery.STATUS_DISABLED)
        self.assertEqual(regs["some_model"]["family"], "pytorch-converted")
        self.assertEqual(regs["my_local"]["mode"], discovery.MODE_SINGLEFILE)

    def test_registration_lookup_defaults_and_prefix(self):
        regs = {"add": {"mode": discovery.MODE_SINGLEFILE,
                        "accuracy": 0.005, "status": discovery.STATUS_ACTIVE,
                        "family": "node"}}
        found = discovery.upstream_registration(regs, "test_add")
        self.assertEqual(found["mode"], discovery.MODE_SINGLEFILE)
        default = discovery.upstream_registration(regs, "test_other")
        self.assertEqual(default["status"], discovery.STATUS_ABSENT)
        self.assertEqual(default["mode"], discovery.MODE_SPLIT)

    def test_missing_active_node_registrations(self):
        path = self._write("CMakeLists.txt", "\n".join([
            "ONNX_backend_node_test(add)",
            "ONNX_backend_node_test_singlefile(batchnorm_example)",
            "# ONNX_backend_node_test(round)",
            "local_node_test(my_local)",
        ]))
        regs = discovery.read_upstream_registrations(path)
        for name in ("test_add", "test_round"):
            os.makedirs(os.path.join(self.tmp, name))
        missing = discovery.missing_upstream_cases(regs, self.tmp)
        self.assertEqual(missing, ["batchnorm_example"])

    def test_build_test_cases_attaches_upstream_registration(self):
        regs = {"abs": {"mode": discovery.MODE_SINGLEFILE,
                        "accuracy": 0.005, "status": discovery.STATUS_ACTIVE,
                        "family": "node"}}
        test_dir = os.path.join(self.tmp, "test_abs")
        os.makedirs(os.path.join(test_dir, "test_data_set_0"))
        with open(os.path.join(test_dir, "model.onnx"), "wb") as handle:
            handle.write(_model_bytes([_node("Abs")]))
        cases = discovery.build_test_cases(self.tmp, regs)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["upstream"]["mode"],
                         discovery.MODE_SINGLEFILE)
        self.assertEqual(cases[0]["upstream"]["accuracy"], 0.005)


def _case_result(operators, outcome, operator_count=None, mode="split",
                 upstream_status="active", cross_check=None):
    names = sorted({op for _domain, op in operators})
    return {
        "id": "case",
        "test": "case",
        "dataset": "test_data_set_0",
        "operators": ["%s:%s" % (domain, op) for domain, op in operators],
        "operator_names": names,
        "domains": sorted({domain for domain, _op in operators}),
        "node_count": len(operators),
        "operator_count": len(operators) if operator_count is None
        else operator_count,
        "multi_operator": False,
        "opset": 13,
        "mode": mode,
        "upstream_status": upstream_status,
        "result": outcome,
        "stage": "execution",
        "exit_code": 0,
        "signal": None,
        "detail": "",
        "root_cause": None,
        "cross_check": cross_check,
        "duration_seconds": 0.0,
        "artifacts": [],
    }


class AggregationTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"ai.onnx": {"Abs": 13, "Conv": 11, "Relu": 14}}
        self.registered = ["Abs", "Conv", "QLinearAdd"]

    def test_status_transitions(self):
        results = [
            _case_result([("ai.onnx", "Abs")], runner.PASS),
            _case_result([("ai.onnx", "Conv")], runner.PASS),
            _case_result([("ai.onnx", "Conv")], runner.NUMERICAL_MISMATCH),
            _case_result([("ai.onnx", "Relu")], runner.ONNX_TO_C_FAILURE),
            _case_result([("com.microsoft", "QLinearAdd")], runner.PASS),
        ]
        operators, totals = report_module.aggregate(
            self.spec, self.registered, results)
        by_name = {(row["domain"], row["operator"]): row for row in operators}
        self.assertEqual(by_name[("ai.onnx", "Abs")]["status"],
                         report_module.STATUS_FULLY)
        self.assertEqual(by_name[("ai.onnx", "Conv")]["status"],
                         report_module.STATUS_PARTIAL)
        self.assertEqual(by_name[("ai.onnx", "Relu")]["status"],
                         report_module.STATUS_NO_PASS)
        self.assertEqual(totals["spec_operators_default_domain"], 3)
        self.assertEqual(totals["registered_in_spec"], 2)
        self.assertEqual(totals["registered_non_spec"], 1)
        self.assertEqual(totals["default_domain_verified"], 2)

    def test_reports_separate_registered_non_spec_operators(self):
        operators, _totals = report_module.aggregate(
            self.spec, self.registered,
            [_case_result([("com.microsoft", "QLinearAdd")], runner.PASS)])
        qlinear = [row for row in operators
                   if row["operator"] == "QLinearAdd"][0]
        self.assertEqual(qlinear["domain"], report_module.CONTRIB_DOMAIN)
        self.assertTrue(qlinear["registered"])
        self.assertFalse(qlinear["spec"])

    def test_multiple_operators_count_for_each_operator(self):
        results = [_case_result([("ai.onnx", "Abs"), ("ai.onnx", "Relu")],
                                runner.PASS)]
        operators, _totals = report_module.aggregate(
            self.spec, self.registered, results)
        by_name = {row["operator"]: row for row in operators}
        self.assertEqual(by_name["Abs"]["applicable_cases"], 1)
        self.assertEqual(by_name["Relu"]["applicable_cases"], 1)
        self.assertEqual(by_name["Abs"]["solo_cases"], 0)

    def test_counts_modes_upstream_status_and_cross_check(self):
        results = [
            _case_result([("ai.onnx", "Abs")], runner.PASS,
                         mode="singlefile", upstream_status="active",
                         cross_check={"mode": "split",
                                      "result": runner.NUMERICAL_MISMATCH}),
            _case_result([("ai.onnx", "Conv")], runner.ONNX_TO_C_FAILURE,
                         mode="split", upstream_status="absent"),
            _case_result([("ai.onnx", "Relu")], runner.PASS,
                         mode="split", upstream_status="disabled",
                         cross_check={"mode": "singlefile",
                                      "result": runner.PASS}),
        ]
        _operators, totals = report_module.aggregate(
            self.spec, self.registered, results)
        self.assertEqual(totals["cases_singlefile"], 1)
        self.assertEqual(totals["cases_split"], 2)
        self.assertEqual(totals["cases_active"], 1)
        self.assertEqual(totals["cases_disabled"], 1)
        self.assertEqual(totals["cases_absent"], 1)
        self.assertEqual(totals["cross_check_ran"], 2)
        self.assertEqual(totals["cross_check_agree"], 1)
        self.assertEqual(totals["cross_check_mode_sensitive"], 1)


class ReportValidationTest(unittest.TestCase):
    def _report(self):
        spec = {"ai.onnx": {"Abs": 13, "Conv": 11}}
        registered = ["Abs"]
        results = [_case_result([("ai.onnx", "Abs")], runner.PASS),
                   _case_result([("ai.onnx", "Conv")],
                                runner.C_COMPILATION_FAILURE)]
        campaign = {"onnx2c_commit": "abc", "onnx_version": "v1.19.0"}
        return report_module.build_report(campaign, spec, registered, results)

    def test_self_validation_passes_for_consistent_report(self):
        report = self._report()
        self.assertTrue(report["validation"]["ok"], report["validation"])
        self.assertEqual(report["totals"]["passed_cases"], 1)
        self.assertEqual(report["totals"]["failed_cases"], 1)

    def test_validation_detects_tampered_totals(self):
        report = self._report()
        report["totals"]["passed_cases"] = 99
        self.assertFalse(report_module.validate(report) == [])

    def test_badges_are_shields_endpoints(self):
        report = self._report()
        badges = report_module.badge_payloads(report)
        self.assertEqual(set(badges), {"registered.json", "verified.json",
                                       "tests.json"})
        for payload in badges.values():
            self.assertEqual(payload["schemaVersion"], 1)
            self.assertIn("message", payload)

    def test_markdown_and_html_reports_render(self):
        report = self._report()
        with tempfile.TemporaryDirectory() as directory:
            markdown_path = os.path.join(directory, "report.md")
            html_path = os.path.join(directory, "report.html")
            report_module.write_markdown(report, markdown_path)
            report_module.write_html(report, html_path,
                                     pages_base_url="https://example.org/x")
            with open(markdown_path, encoding="utf-8") as handle:
                markdown = handle.read()
            with open(html_path, encoding="utf-8") as handle:
                document = handle.read()
        self.assertIn("# ONNX operator coverage", markdown)
        self.assertIn("| Operator | Domain |", markdown)
        self.assertIn("### Failing test cases", markdown)
        self.assertTrue(document.startswith("<!DOCTYPE html>"))
        self.assertIn("<table>", document)
        self.assertIn("img.shields.io/endpoint", document)
        self.assertIn("Conv", document)


class OutcomeMappingTest(unittest.TestCase):
    def test_result_groups_are_disjoint_and_total(self):
        categories = set(runner.FAILURE_CATEGORIES)
        self.assertNotIn(runner.PASS, categories)
        self.assertNotIn(runner.PASS,
                         set(runner.UNRUNNABLE_CATEGORIES))
        self.assertIn(runner.TIMEOUT, runner.UNRUNNABLE_CATEGORIES)
        self.assertIn(runner.INFRASTRUCTURE_ERROR,
                      runner.UNRUNNABLE_CATEGORIES)


INTEGRATION = os.environ.get("ONNX2C_COVERAGE_INTEGRATION") == "1"


@unittest.skipUnless(INTEGRATION, "set ONNX2C_COVERAGE_INTEGRATION=1")
class IntegrationTest(unittest.TestCase):
    def test_abs_case_passes(self):
        import cli
        repo_root = cli._repo_root(os.path.dirname(__file__))
        onnx2c = cli._find_binary(repo_root, None, "onnx2c")
        testgen = cli._find_binary(repo_root, None, "testgen")
        if not onnx2c or not testgen:
            self.skipTest("onnx2c/testgen not built")
        test_data_dir = os.path.join(repo_root, "onnx", "onnx", "backend",
                                     "test", "data", "node")
        case = None
        for candidate in discovery.build_test_cases(test_data_dir):
            if candidate["test"] == "test_abs":
                case = candidate
                break
        self.assertIsNotNone(case)
        with tempfile.TemporaryDirectory() as work:
            result = runner.run_case(
                case, {"onnx2c": onnx2c, "testgen": testgen, "cc": "cc"},
                work, timeout=60.0, accuracy=2e-05,
                cc_flags=runner.DEFAULT_CC_FLAGS)
        self.assertEqual(result["result"], runner.PASS, result["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
