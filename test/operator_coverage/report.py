"""Aggregation and rendering of coverage results.

The report separates three distinct questions:

* **registered** - do we dispatch this operator at all (``Graph::createNode``)?
* **verified** - did at least one applicable backend test pass end to end?
* **case pass rate** - how many individual test cases pass?

All aggregate numbers are derived from the per-operator and per-case records
in the same document, and :func:`validate` re-derives them so drift between
reference expectations and the report can never go unnoticed.
"""

import html
import json
import urllib.parse
from collections import Counter

import discovery
import runner

SCHEMA_VERSION = 1

DOMAIN_RANK = {"ai.onnx": 0, "ai.onnx.ml": 1}
CONTRIB_DOMAIN = "(non-spec)"

STATUS_FULLY = "fully_tested"
STATUS_PARTIAL = "partially_tested"
STATUS_NO_PASS = "no_passing_test"
STATUS_UNTESTED = "untested"

_FAILED_RESULTS = frozenset((
    runner.ONNX_TO_C_FAILURE,
    runner.C_COMPILATION_FAILURE,
    runner.EXECUTION_FAILURE,
    runner.NUMERICAL_MISMATCH,
))
_UNRUNNABLE_RESULTS = frozenset(runner.UNRUNNABLE_CATEGORIES)


def _domain_rank(domain):
    return DOMAIN_RANK.get(domain, 2)


def _spec_names_by_domain(spec):
    return {domain: set(table) for domain, table in spec.items()}


def aggregate(spec, registered, results):
    """Return ``(operator_rows, totals)`` derived from the raw records."""
    registered_set = set(registered)
    names_in_spec = set()
    for table in spec.values():
        names_in_spec.update(table)
    default_spec_names = set(spec.get("ai.onnx", {}))

    rows = {}

    def ensure(domain, name, spec_version=None):
        key = (domain, name)
        row = rows.get(key)
        if row is None:
            row = {
                "operator": name,
                "domain": domain,
                "spec": key in spec_entries,
                "spec_version": spec_version,
                "registered": name in registered_set,
                "applicable_cases": 0,
                "passed": 0,
                "failed": 0,
                "unrunnable": 0,
                "solo_cases": 0,
                "solo_passed": 0,
                "opsets": set(),
            }
            rows[key] = row
        if spec_version is not None and row["spec_version"] is None:
            row["spec_version"] = spec_version
        return row

    spec_entries = set()
    for domain, table in spec.items():
        for name, version in table.items():
            spec_entries.add((domain, name))
            ensure(domain, name, version)

    for name in registered:
        if name not in names_in_spec:
            ensure(CONTRIB_DOMAIN, name)

    for result in results:
        for operator in result["operators"]:
            domain, name = operator.split(":", 1)
            ensure(domain, name)
            row = rows[(domain, name)]
            row["applicable_cases"] += 1
            if result["result"] == runner.PASS:
                row["passed"] += 1
            elif result["result"] in _FAILED_RESULTS:
                row["failed"] += 1
            elif result["result"] in _UNRUNNABLE_RESULTS:
                row["unrunnable"] += 1
            if result["operator_count"] == 1:
                row["solo_cases"] += 1
                if result["result"] == runner.PASS:
                    row["solo_passed"] += 1
            if result.get("opset") is not None:
                row["opsets"].add(result["opset"])

    ordered = []
    for row in rows.values():
        if row["applicable_cases"] == 0:
            status = STATUS_UNTESTED
        elif row["passed"] == 0:
            status = STATUS_NO_PASS
        elif row["failed"] == 0 and row["unrunnable"] == 0:
            status = STATUS_FULLY
        else:
            status = STATUS_PARTIAL
        row["status"] = status
        row["opsets"] = sorted(row["opsets"])
        ordered.append(row)
    ordered.sort(key=lambda row: (_domain_rank(row["domain"]), row["operator"]))

    default_rows = [row for row in ordered if row["domain"] == "ai.onnx"]
    status_counts = Counter(row["status"] for row in ordered)

    case_counts = Counter(result["result"] for result in results)
    category_counts = {name: case_counts.get(name, 0)
                       for name in (runner.PASS,) + runner.FAILURE_CATEGORIES}
    mode_counts = Counter(result.get("mode") for result in results)
    upstream_status_counts = Counter(result.get("upstream_status")
                                     for result in results)

    cross_check_present = [result for result in results
                           if result.get("cross_check")]
    cross_check_ran = len(cross_check_present)
    cross_check_agree = sum(
        1 for result in cross_check_present
        if result["cross_check"]["result"] == result["result"])

    totals = {
        "spec_operators_default_domain": len(default_spec_names),
        "spec_operators_all_domains": sum(len(table) for table in spec.values()),
        "spec_operators_per_domain": {
            domain: len(table) for domain, table in sorted(spec.items())
        },
        "registered_operators": len(registered),
        "registered_in_default_domain": len(
            [name for name in registered if name in default_spec_names]),
        "registered_in_spec": len([name for name in registered
                                   if name in names_in_spec]),
        "registered_non_spec": len([name for name in registered
                                    if name not in names_in_spec]),
        "operators_fully_tested": status_counts.get(STATUS_FULLY, 0),
        "operators_partially_tested": status_counts.get(STATUS_PARTIAL, 0),
        "operators_no_passing_test": status_counts.get(STATUS_NO_PASS, 0),
        "operators_untested": status_counts.get(STATUS_UNTESTED, 0),
        "operators_with_passing_test": sum(
            1 for row in ordered
            if row["status"] in (STATUS_FULLY, STATUS_PARTIAL)),
        "default_domain_fully_tested": sum(
            1 for row in default_rows if row["status"] == STATUS_FULLY),
        "default_domain_partially_tested": sum(
            1 for row in default_rows if row["status"] == STATUS_PARTIAL),
        "default_domain_no_passing_test": sum(
            1 for row in default_rows if row["status"] == STATUS_NO_PASS),
        "default_domain_untested": sum(
            1 for row in default_rows if row["status"] == STATUS_UNTESTED),
        "default_domain_verified": sum(
            1 for row in default_rows
            if row["status"] in (STATUS_FULLY, STATUS_PARTIAL)),
        "discovered_cases": len(results),
        "passed_cases": case_counts.get(runner.PASS, 0),
        "failed_cases": sum(case_counts.get(category, 0)
                            for category in _FAILED_RESULTS),
        "unrunnable_cases": sum(case_counts.get(category, 0)
                                for category in _UNRUNNABLE_RESULTS),
        "cases_per_category": category_counts,
        "cases_singlefile": mode_counts.get(discovery.MODE_SINGLEFILE, 0),
        "cases_split": mode_counts.get(discovery.MODE_SPLIT, 0),
        "cases_active": upstream_status_counts.get(discovery.STATUS_ACTIVE, 0),
        "cases_disabled": upstream_status_counts.get(
            discovery.STATUS_DISABLED, 0),
        "cases_absent": upstream_status_counts.get(discovery.STATUS_ABSENT, 0),
        "cross_check_ran": cross_check_ran,
        "cross_check_agree": cross_check_agree,
        "cross_check_mode_sensitive": cross_check_ran - cross_check_agree,
    }
    return ordered, totals


def _ratio(passed, total):
    return (passed / total) if total else 0.0


def _format_ratio(passed, total):
    if total == 0:
        return f"{passed}/{total}"
    return f"{passed}/{total} ({100.0 * passed / total:.1f}%)"


def validate(report):
    """Re-derive every aggregate from the raw records; return error strings."""
    errors = []
    operators = report["operators"]
    results = report["tests"]
    totals = report["totals"]

    status_counts = Counter(row["status"] for row in operators)
    default_rows = [row for row in operators if row["domain"] == "ai.onnx"]
    checks = {
        "operators_untested": status_counts.get(STATUS_UNTESTED, 0),
        "operators_fully_tested": status_counts.get(STATUS_FULLY, 0),
        "operators_partially_tested": status_counts.get(STATUS_PARTIAL, 0),
        "operators_no_passing_test": status_counts.get(STATUS_NO_PASS, 0),
        "operators_with_passing_test": sum(
            1 for row in operators
            if row["status"] in (STATUS_FULLY, STATUS_PARTIAL)),
        "default_domain_verified": sum(
            1 for row in default_rows
            if row["status"] in (STATUS_FULLY, STATUS_PARTIAL)),
        "discovered_cases": len(results),
    }
    case_counts = Counter(result["result"] for result in results)
    checks["passed_cases"] = case_counts.get(runner.PASS, 0)
    checks["failed_cases"] = sum(case_counts.get(category, 0)
                                 for category in _FAILED_RESULTS)
    checks["unrunnable_cases"] = sum(case_counts.get(category, 0)
                                     for category in _UNRUNNABLE_RESULTS)
    mode_counts = Counter(result.get("mode") for result in results)
    checks["cases_singlefile"] = mode_counts.get(discovery.MODE_SINGLEFILE, 0)
    checks["cases_split"] = mode_counts.get(discovery.MODE_SPLIT, 0)
    status_counts = Counter(result.get("upstream_status") for result in results)
    checks["cases_active"] = status_counts.get(discovery.STATUS_ACTIVE, 0)
    checks["cases_disabled"] = status_counts.get(discovery.STATUS_DISABLED, 0)
    checks["cases_absent"] = status_counts.get(discovery.STATUS_ABSENT, 0)
    for key, value in checks.items():
        if totals.get(key) != value:
            errors.append(
                f"{key}: totals={totals.get(key)} recomputed={value}")

    for row in operators:
        recomputed_total = row["passed"] + row["failed"] + row["unrunnable"]
        if recomputed_total != row["applicable_cases"]:
            errors.append(
                f"operator {row['domain']}:{row['operator']} applicable "
                f"{row['applicable_cases']} != {recomputed_total}")
    return errors


def build_report(campaign, spec, registered, results):
    operators, totals = aggregate(spec, registered, results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "campaign": campaign,
        "totals": totals,
        "operators": operators,
        "tests": sorted(results, key=lambda result: result["id"]),
    }
    errors = validate(report)
    report["validation"] = {"ok": not errors, "errors": errors}
    return report


def write_json(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
        "message": f"{passed}/{total} ({100.0 * _ratio(passed, total):.1f}%)",
        "color": _badge_color(passed, total),
    }


def badge_payloads(report):
    totals = report["totals"]
    denominator = totals["spec_operators_default_domain"]
    return {
        "registered.json": _badge(
            "onnx2c registered operators",
            totals["registered_in_default_domain"], denominator),
        "verified.json": _badge(
            "onnx2c verified operators", totals["default_domain_verified"],
            denominator),
        "tests.json": _badge(
            "onnx2c backend tests", totals["passed_cases"],
            totals["discovered_cases"]),
    }


def write_badges(report, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for filename, payload in badge_payloads(report).items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written[filename] = path
    return written


_BADGE_SPECS = (
    ("registered.json", "registered ONNX operators"),
    ("verified.json", "verified ONNX operators"),
    ("tests.json", "passing ONNX backend tests"),
)


def _badge_markdown(report, pages_base_url):
    if not pages_base_url:
        return []
    base = pages_base_url.rstrip("/")
    lines = []
    for filename, label in _BADGE_SPECS:
        endpoint = f"{base}/badges/{filename}"
        lines.append(
            f"[![{label}](https://img.shields.io/endpoint?"
            f"url={urllib.parse.quote(endpoint, safe='')})]"
            f"({base}/index.html)")
    return lines


def _status_rows(report):
    totals = report["totals"]
    denominator = totals["spec_operators_default_domain"]
    return [
        ("Operators defined by the pinned specification (ai.onnx)",
         str(totals["spec_operators_default_domain"])),
        ("Operators registered by onnx2c", str(totals["registered_operators"])),
        ("Registered operators in the ai.onnx domain",
         _format_ratio(totals["registered_in_default_domain"], denominator)),
        ("Registered operators in any specification domain",
         str(totals["registered_in_spec"])),
        ("Registered operators outside the specification",
         str(totals["registered_non_spec"])),
        ("Operators with at least one passing backend test",
         _format_ratio(totals["default_domain_verified"], denominator)),
        ("Fully tested operators",
         _format_ratio(totals["default_domain_fully_tested"], denominator)),
        ("Partially tested operators",
         _format_ratio(totals["default_domain_partially_tested"], denominator)),
        ("Operators with no passing test",
         _format_ratio(totals["default_domain_no_passing_test"], denominator)),
        ("Operators with no applicable test",
         _format_ratio(totals["default_domain_untested"], denominator)),
        ("Backend test cases executed", str(totals["discovered_cases"])),
        ("Backend test cases passed",
         _format_ratio(totals["passed_cases"], totals["discovered_cases"])),
    ]


def _outcome_rows(report):
    categories = report["totals"]["cases_per_category"]
    order = [runner.PASS] + list(runner.FAILURE_CATEGORIES)
    return [(category, str(categories.get(category, 0))) for category in order]


def _basis_rows(report):
    campaign = report["campaign"]
    totals = report["totals"]
    registrations = campaign.get("upstream_registrations") or {}
    return [
        ("Upstream tests discovered (test/CMakeLists.txt)",
         "%s active, %s commented out"
         % (registrations.get("active", "?"),
            registrations.get("disabled", "?"))),
        ("Upstream active node tests",
         str(registrations.get("active_node", "?"))),
        ("Active tests registered singlefile",
         str(registrations.get("active_singlefile", "?"))),
        ("Active tests registered split",
         str(registrations.get("active_split", "?"))),
        ("Corpus cases executed in singlefile mode",
         str(totals.get("cases_singlefile", "?"))),
        ("Corpus cases executed in split mode",
         str(totals.get("cases_split", "?"))),
        ("Active upstream registrations missing from corpus",
         str(len(registrations.get("missing_from_corpus", [])))),
        ("Reference tolerance",
         "per-test from test/CMakeLists.txt"
         + (" (overridden)" if campaign.get("accuracy_override") else "")),
        ("Cross-check (alternate mode re-run)",
         ("%d run, %d agree, %d mode-sensitive"
          % (totals.get("cross_check_ran", 0),
             totals.get("cross_check_agree", 0),
             totals.get("cross_check_mode_sensitive", 0)))
         if campaign.get("cross_check") else "disabled"),
    ]


_OPERATOR_HEADERS = ("Operator", "Domain", "Registered", "Applicable",
                     "Passed", "Failed", "Unrunnable", "Status")


def _operator_rows(operators):
    return [
        (row["operator"], row["domain"],
         "yes" if row["registered"] else "no",
         str(row["applicable_cases"]), str(row["passed"]),
         str(row["failed"]), str(row["unrunnable"]), row["status"])
        for row in operators
    ]


_FAILURE_HEADERS = ("Test case", "Result", "Mode", "Upstream", "Stage",
                    "Cross-check", "Operators", "Root cause / detail")


def _failure_rows(report):
    rows = []
    for result in report["tests"]:
        if result["result"] == runner.PASS:
            continue
        operators = ", ".join(result["operators"]) or "-"
        mode = result.get("mode") or "-"
        upstream = "%s/%s" % (result.get("upstream_status", "-"),
                              result.get("upstream_mode", "-"))
        check = result.get("cross_check")
        if check:
            cross = ("%s (%s)" % (check["result"], check["mode"])
                     if check["result"] != result["result"]
                     else "%s (agree)" % check["result"])
        else:
            cross = "-"
        cause = result.get("root_cause")
        detail = ("%s: %s" % (cause, result["detail"] or "")
                  if cause else (result["detail"] or ""))
        detail = detail.replace("\n", " ")
        if len(detail) > 220:
            detail = detail[:217] + "..."
        rows.append((result["id"], result["result"], mode, upstream,
                     result["stage"], cross, operators, detail))
    return rows


_MODE_SENSITIVITY_HEADERS = ("Test case", "Upstream", "Registered result",
                             "Other mode", "Other result", "Operators")


def _mode_sensitivity_rows(report):
    rows = []
    for result in report["tests"]:
        check = result.get("cross_check")
        if not check or check["result"] == result["result"]:
            continue
        upstream = "%s/%s" % (result.get("upstream_status", "-"),
                              result.get("upstream_mode", "-"))
        operators = ", ".join(result["operators"]) or "-"
        rows.append((result["id"], upstream,
                     "%s (%s)" % (result["result"], result.get("mode", "-")),
                     check["mode"], check["result"], operators))
    return rows


_MODE_SENSITIVITY_NOTE = (
    "These cases produce different outcomes in the two test modes. Upstream "
    "registers a test in whichever mode it passes (test/CMakeLists.txt); the "
    "metric uses that registered mode, and the divergence is recorded here "
    "rather than hidden. A common cause is the split harness embedding input "
    "constants with only six significant digits, while testgen_singlefile "
    "uses full precision and makes inputs compile-time constants (upstream "
    "note: \"Accuracy breaks if not run in _singlefile\").")


def _provenance_rows(report):
    campaign = report["campaign"]
    totals = report["totals"]
    return [
        ("onnx2c commit", campaign.get("onnx2c_commit", "?")
         + (" (dirty working tree)" if campaign.get("onnx2c_dirty") else "")),
        ("ONNX version", "%s (submodule %s)"
         % (campaign.get("onnx_version", "?"),
            campaign.get("onnx_submodule_commit", "?"))),
        ("Compiler", campaign.get("compiler", "?")),
        ("Reference tolerance", "per-test from test/CMakeLists.txt (default %s%s)"
         % (campaign.get("accuracy", "?"),
            ", overridden" if campaign.get("accuracy_override") else "")),
        ("Campaign mode", "%s (complete: %s, infrastructure clean: %s)"
         % (campaign.get("mode", "?"), campaign.get("complete", False),
            campaign.get("clean", False))),
        ("Host", campaign.get("host", "?")),
        ("Specification denominator",
         "%d operators in ai.onnx; other domains: %s"
         % (totals["spec_operators_default_domain"],
            ", ".join("%s=%d" % (domain, count) for domain, count
                      in totals["spec_operators_per_domain"].items()))),
    ]


_NOTES = [
    "Operators that only appear in multi-operator graphs are counted for each "
    "operator in the graph; the solo_cases/solo_passed fields in the JSON "
    "report give the strict single-operator view.",
    "A registered operator proves dispatch exists, not that it is correct; "
    "only passing backend tests count as verified.",
    "Each case runs in the mode test/CMakeLists.txt registers it in; tests "
    "upstream does not register run in split mode (the onnx2c command-line "
    "path). Cases whose result differs in the other mode are listed under "
    "Mode-sensitive cases.",
    "Infrastructure errors (harness/tooling limitations) are reported "
    "separately and never counted as passing tests; a campaign can be "
    "complete while not being infrastructure clean.",
]


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ")
                 for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _html_table(headers, rows):
    parts = ["<table>", "<thead><tr>"]
    parts.extend("<th>%s</th>" % html.escape(str(header)) for header in headers)
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend("<td>%s</td>" % html.escape(str(cell)) for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _html_kv_table(rows):
    return _html_table(("Metric", "Value"), rows)


def _html_list(items):
    return ("<ul>\n"
            + "\n".join("<li>%s</li>" % html.escape(item) for item in items)
            + "\n</ul>")


def _badge_links(pages_base_url):
    if not pages_base_url:
        return []
    base = pages_base_url.rstrip("/")
    links = []
    for filename, label in _BADGE_SPECS:
        endpoint = "%s/badges/%s" % (base, filename)
        quoted = urllib.parse.quote(endpoint, safe="")
        links.append(
            '<a href="%s/index.html"><img alt="%s" '
            'src="https://img.shields.io/endpoint?url=%s"></a>'
            % (html.escape(base), html.escape(label), quoted))
    return links


_HTML_STYLE = """\
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 64rem; margin: 2rem auto; padding: 0 1rem; color: #1f2328; }
h1, h2, h3 { color: #0b3d63; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #d0d7de; padding: 0.3rem 0.6rem; text-align: left; }
th { background: #f6f8fa; }
code { background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 3px; }
.badges img { margin-right: 0.4rem; }
"""


def write_html(report, path, pages_base_url=None):
    campaign = report["campaign"]
    default_rows = [row for row in report["operators"]
                    if row["domain"] == "ai.onnx"]
    other_rows = [row for row in report["operators"]
                  if row["domain"] != "ai.onnx"]
    document = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        "<title>ONNX operator coverage</title>",
        "<style>",
        _HTML_STYLE,
        "</style>",
        "</head>",
        "<body>",
        "<h1>ONNX operator coverage</h1>",
        "<p>Generated for onnx2c <code>%s</code> with ONNX <code>%s</code> "
        "(submodule <code>%s</code>).</p>"
        % (html.escape(str(campaign.get("onnx2c_commit", "?"))),
           html.escape(str(campaign.get("onnx_version", "?"))),
           html.escape(str(campaign.get("onnx_submodule_commit", "?")))),
    ]
    badges = _badge_links(pages_base_url)
    if badges:
        document.append('<p class="badges">' + "\n".join(badges) + "</p>")
    document.extend(["<h2>Summary</h2>", _html_kv_table(_status_rows(report))])
    document.extend([
        "<h3>Outcome of individual test cases</h3>",
        _html_table(("Outcome", "Cases"), _outcome_rows(report)),
        "<h2>Measurement basis</h2>",
        "<p>Each case runs in the mode upstream registers it in (singlefile "
        "via <code>testgen_singlefile</code>, otherwise split: "
        "<code>onnx2c</code> + <code>testgen</code>), so results are directly "
        "comparable with upstream CTest.</p>",
        _html_kv_table(_basis_rows(report)),
        "<h2>Operator matrix (specification domain)</h2>",
        "<h3>ai.onnx</h3>",
        _html_table(_OPERATOR_HEADERS, _operator_rows(default_rows)),
    ])
    if other_rows:
        document.extend([
            "<h3>Other domains and registered-but-unspecified operators</h3>",
            _html_table(_OPERATOR_HEADERS, _operator_rows(other_rows)),
        ])
    failure_rows = _failure_rows(report)
    if failure_rows:
        document.extend([
            "<h2>Failing test cases</h2>",
            _html_table(_FAILURE_HEADERS, failure_rows),
        ])
    sensitivity_rows = _mode_sensitivity_rows(report)
    if sensitivity_rows:
        document.extend([
            "<h2>Mode-sensitive cases</h2>",
            "<p>%s</p>" % html.escape(_MODE_SENSITIVITY_NOTE),
            _html_table(_MODE_SENSITIVITY_HEADERS, sensitivity_rows),
        ])
    document.extend([
        "<h2>Provenance</h2>",
        _html_kv_table(_provenance_rows(report)),
        "<h2>Notes and limitations</h2>",
        _html_list(_NOTES),
        "</body>",
        "</html>",
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(document))



def write_markdown(report, path, pages_base_url=None):
    campaign = report["campaign"]
    default_rows = [row for row in report["operators"]
                    if row["domain"] == "ai.onnx"]
    other_rows = [row for row in report["operators"]
                  if row["domain"] != "ai.onnx"]
    lines = [
        "# ONNX operator coverage",
        "",
        f"Generated for onnx2c `{campaign.get('onnx2c_commit', '?')}` with "
        f"ONNX `{campaign.get('onnx_version', '?')}` "
        f"(`{campaign.get('onnx_submodule_commit', '?')}`).",
        "",
    ]
    badges = _badge_markdown(report, pages_base_url)
    if badges:
        lines.extend(badges)
        lines.append("")
    lines.extend(["## Summary", ""])
    lines.extend(_markdown_table(("Metric", "Value"), _status_rows(report)))
    lines.extend(["", "### Outcome of individual test cases", ""])
    lines.extend(_markdown_table(("Outcome", "Cases"), _outcome_rows(report)))
    lines.extend([
        "",
        "## Measurement basis",
        "",
        "Each case runs in the mode upstream registers it in (singlefile via "
        "`testgen_singlefile`, otherwise split: `onnx2c` + `testgen`), so "
        "results are directly comparable with upstream CTest.",
        "",
    ])
    lines.extend(_markdown_table(("Measurement", "Value"), _basis_rows(report)))
    lines.extend([
        "",
        "## Operator matrix (specification domain)",
        "",
        "### ai.onnx",
        "",
    ])
    lines.extend(_markdown_table(_OPERATOR_HEADERS, _operator_rows(default_rows)))
    if other_rows:
        lines.extend([
            "",
            "### Other domains and registered-but-unspecified operators",
            "",
        ])
        lines.extend(
            _markdown_table(_OPERATOR_HEADERS, _operator_rows(other_rows)))
    failure_rows = _failure_rows(report)
    if failure_rows:
        lines.extend([
            "",
            "### Failing test cases",
            "",
        ])
        lines.extend(_markdown_table(_FAILURE_HEADERS, failure_rows))
    sensitivity_rows = _mode_sensitivity_rows(report)
    if sensitivity_rows:
        lines.extend([
            "",
            "### Mode-sensitive cases",
            "",
            _MODE_SENSITIVITY_NOTE,
            "",
        ])
        lines.extend(_markdown_table(_MODE_SENSITIVITY_HEADERS,
                                     sensitivity_rows))
    lines.extend([
        "",
        "## Provenance",
        "",
    ])
    lines.extend(_markdown_table(("Field", "Value"), _provenance_rows(report)))
    lines.extend([
        "",
        "## Notes and limitations",
        "",
    ])
    lines.extend(f"- {note}" for note in _NOTES)
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
