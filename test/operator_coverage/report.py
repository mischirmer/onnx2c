"""Aggregation, classification, and rendering of operator coverage results.

The report answers three distinct questions:

* **registered** - does onnx2c dispatch the operator (``Graph::createNode``)?
* **verified** - did at least one backend test pass end to end?
* **case pass rate** - how many individual test cases pass?

Per-operator verdicts are derived from *single-operator* tests only.  A test
that exercises several operators at once cannot be attributed to any one of
them, so a failure there is reported separately and never blames an individual
implementation.  Every aggregate is re-derived from the per-case records by
:func:`validate`, so the rendering can never silently drift from the data.
"""

import json
import os
from collections import Counter

import discovery
import runner

SCHEMA_VERSION = 2

DOMAIN_RANK = {"ai.onnx": 0, "ai.onnx.ml": 1}
CONTRIB_DOMAIN = "(non-spec)"

CAT_MISSING = "missing"
CAT_UNTESTED = "untested"
CAT_FAILING = "failing"
CAT_PARTIAL = "partial"
CAT_BLOCKED = "blocked"
CAT_PASSING = "passing"

# Operators needing attention first; alphabetical inside a category.
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

# Columns of the compact failure table.  ``timeout`` counts as a runtime
# failure: the case ran but produced no usable result.
FAILURE_COLUMNS = (
    ("ONNX\u2192C", (runner.ONNX_TO_C_FAILURE,)),
    ("C compile", (runner.C_COMPILATION_FAILURE,)),
    ("Numerical", (runner.NUMERICAL_MISMATCH,)),
    ("Runtime", (runner.EXECUTION_FAILURE, runner.TIMEOUT)),
    ("Infrastructure", (runner.INFRASTRUCTURE_ERROR,)),
)

_FAILED_RESULTS = frozenset((
    runner.ONNX_TO_C_FAILURE,
    runner.C_COMPILATION_FAILURE,
    runner.EXECUTION_FAILURE,
    runner.NUMERICAL_MISMATCH,
))
_UNRUNNABLE_RESULTS = frozenset(runner.UNRUNNABLE_CATEGORIES)


def _domain_rank(domain):
    return DOMAIN_RANK.get(domain, 2)


def classify(registered, applicable, passed, failed, unrunnable):
    """Map a single-operator's counters onto one mutually exclusive category.

    ``failed`` counts tests that were evaluated and rejected; ``unrunnable``
    counts tests whose result could not be evaluated (harness/tooling errors,
    timeouts).  A registered operator is never reported as *Passing* while any
    applicable test remains unevaluated - such uncertainty is *Blocked*.
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


def _new_row(domain, name, spec, spec_version, registered):
    return {
        "operator": name,
        "domain": domain,
        "spec": spec,
        "spec_version": spec_version,
        "registered": registered,
        # Counters over every case that mentions the operator (informational).
        "applicable_cases": 0,
        "passed": 0,
        "failed": 0,
        "unrunnable": 0,
        # Counters over single-operator cases only (drive the verdict).
        "solo_cases": 0,
        "solo_passed": 0,
        "solo_failed": 0,
        "solo_unrunnable": 0,
        "shared_cases": 0,
        "solo_per_category": {},
        "opsets": set(),
    }


def aggregate(spec, registered, results):
    """Return ``(operator_rows, totals)`` derived from the raw records."""
    registered_set = set(registered)
    names_in_spec = set()
    for table in spec.values():
        names_in_spec.update(table)
    default_spec_names = set(spec.get("ai.onnx", {}))

    spec_entries = set()
    for domain, table in spec.items():
        for name in table:
            spec_entries.add((domain, name))

    rows = {}

    def ensure(domain, name, spec_version=None):
        key = (domain, name)
        row = rows.get(key)
        if row is None:
            row = _new_row(domain, name, key in spec_entries, spec_version,
                           name in registered_set)
            rows[key] = row
        if spec_version is not None and row["spec_version"] is None:
            row["spec_version"] = spec_version
        return row

    for domain, table in spec.items():
        for name, version in table.items():
            ensure(domain, name, version)

    for name in registered:
        if name not in names_in_spec:
            ensure(CONTRIB_DOMAIN, name)

    for result in results:
        keys = sorted(set(result["operators"]))
        solo = len(keys) == 1
        outcome = result["result"]
        for key in keys:
            domain, name = key.split(":", 1)
            row = ensure(domain, name)
            row["applicable_cases"] += 1
            if outcome == runner.PASS:
                row["passed"] += 1
            elif outcome in _FAILED_RESULTS:
                row["failed"] += 1
            elif outcome in _UNRUNNABLE_RESULTS:
                row["unrunnable"] += 1
            if solo:
                row["solo_cases"] += 1
                counts = row["solo_per_category"]
                counts[outcome] = counts.get(outcome, 0) + 1
                if outcome == runner.PASS:
                    row["solo_passed"] += 1
                elif outcome in _FAILED_RESULTS:
                    row["solo_failed"] += 1
                elif outcome in _UNRUNNABLE_RESULTS:
                    row["solo_unrunnable"] += 1
            else:
                row["shared_cases"] += 1
            if result.get("opset") is not None:
                row["opsets"].add(result["opset"])

    ordered = []
    for row in rows.values():
        row["status"] = classify(
            row["registered"], row["solo_cases"], row["solo_passed"],
            row["solo_failed"], row["solo_unrunnable"])
        row["opsets"] = sorted(row["opsets"])
        ordered.append(row)
    ordered.sort(key=lambda row: (_domain_rank(row["domain"]), row["operator"]))

    default_rows = [row for row in ordered
                    if row["domain"] == "ai.onnx" and row["spec"]]
    all_cat = Counter(row["status"] for row in ordered)
    default_cat = Counter(row["status"] for row in default_rows)

    case_counts = Counter(result["result"] for result in results)
    category_counts = {name: case_counts.get(name, 0)
                       for name in (runner.PASS,) + runner.FAILURE_CATEGORIES}
    mode_counts = Counter(result.get("mode") for result in results)
    upstream_status_counts = Counter(result.get("upstream_status")
                                     for result in results)

    def verified(cat_counts):
        return cat_counts.get(CAT_PASSING, 0) + cat_counts.get(CAT_PARTIAL, 0)

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
        "operators_missing": all_cat.get(CAT_MISSING, 0),
        "operators_failing": all_cat.get(CAT_FAILING, 0),
        "operators_partial": all_cat.get(CAT_PARTIAL, 0),
        "operators_blocked": all_cat.get(CAT_BLOCKED, 0),
        "operators_untested": all_cat.get(CAT_UNTESTED, 0),
        "operators_passing": all_cat.get(CAT_PASSING, 0),
        "operators_with_passing_test": verified(all_cat),
        "default_domain_missing": default_cat.get(CAT_MISSING, 0),
        "default_domain_failing": default_cat.get(CAT_FAILING, 0),
        "default_domain_partial": default_cat.get(CAT_PARTIAL, 0),
        "default_domain_blocked": default_cat.get(CAT_BLOCKED, 0),
        "default_domain_untested": default_cat.get(CAT_UNTESTED, 0),
        "default_domain_passing": default_cat.get(CAT_PASSING, 0),
        "default_domain_verified": verified(default_cat),
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
        "cross_check_ran": sum(1 for result in results
                               if result.get("cross_check")),
        "cross_check_agree": sum(
            1 for result in results if result.get("cross_check")
            and result["cross_check"]["result"] == result["result"]),
    }
    totals["cross_check_mode_sensitive"] = (
        totals["cross_check_ran"] - totals["cross_check_agree"])
    return ordered, totals


def _ratio(passed, total):
    return (passed / total) if total else 0.0


def _format_ratio(passed, total):
    if total == 0:
        return f"{passed}/{total}"
    return f"{passed}/{total} ({100.0 * passed / total:.1f}%)"


def _recategorize(report):
    """Re-derive every operator's counters and category from the raw cases."""
    applic = Counter()
    passed = Counter()
    failed = Counter()
    unrunnable = Counter()
    solo = Counter()
    solo_pass = Counter()
    solo_fail = Counter()
    solo_unr = Counter()
    for result in report["tests"]:
        keys = sorted(set(result["operators"]))
        solo_case = len(keys) == 1
        outcome = result["result"]
        for key in keys:
            applic[key] += 1
            if outcome == runner.PASS:
                passed[key] += 1
            elif outcome in _FAILED_RESULTS:
                failed[key] += 1
            elif outcome in _UNRUNNABLE_RESULTS:
                unrunnable[key] += 1
            if solo_case:
                solo[key] += 1
                if outcome == runner.PASS:
                    solo_pass[key] += 1
                elif outcome in _FAILED_RESULTS:
                    solo_fail[key] += 1
                elif outcome in _UNRUNNABLE_RESULTS:
                    solo_unr[key] += 1
    return {
        "applicable_cases": applic, "passed": passed, "failed": failed,
        "unrunnable": unrunnable, "solo_cases": solo, "solo_passed": solo_pass,
        "solo_failed": solo_fail, "solo_unrunnable": solo_unr,
    }


def validate(report):
    """Re-derive every aggregate from the raw records; return error strings."""
    errors = []
    operators = report["operators"]
    results = report["tests"]
    totals = report["totals"]

    derived = _recategorize(report)
    seen = set()
    for row in operators:
        key = "%s:%s" % (row["domain"], row["operator"])
        seen.add(key)
        for field, counter in derived.items():
            if row.get(field) != counter[key]:
                errors.append(
                    "%s %s: stored=%s recomputed=%s"
                    % (key, field, row.get(field), counter[key]))
        expected = classify(
            row["registered"], derived["solo_cases"][key],
            derived["solo_passed"][key], derived["solo_failed"][key],
            derived["solo_unrunnable"][key])
        if row["status"] != expected:
            errors.append(
                "%s status: stored=%s recomputed=%s"
                % (key, row["status"], expected))
    for key, count in derived["applicable_cases"].items():
        if count and key not in seen:
            errors.append("missing operator row for %s" % key)

    default_rows = [row for row in operators
                    if row["domain"] == "ai.onnx" and row["spec"]]
    all_cat = Counter(row["status"] for row in operators)
    default_cat = Counter(row["status"] for row in default_rows)

    def verified(cat_counts):
        return cat_counts.get(CAT_PASSING, 0) + cat_counts.get(CAT_PARTIAL, 0)

    checks = {
        "operators_missing": all_cat.get(CAT_MISSING, 0),
        "operators_failing": all_cat.get(CAT_FAILING, 0),
        "operators_partial": all_cat.get(CAT_PARTIAL, 0),
        "operators_blocked": all_cat.get(CAT_BLOCKED, 0),
        "operators_untested": all_cat.get(CAT_UNTESTED, 0),
        "operators_passing": all_cat.get(CAT_PASSING, 0),
        "operators_with_passing_test": verified(all_cat),
        "default_domain_missing": default_cat.get(CAT_MISSING, 0),
        "default_domain_failing": default_cat.get(CAT_FAILING, 0),
        "default_domain_partial": default_cat.get(CAT_PARTIAL, 0),
        "default_domain_blocked": default_cat.get(CAT_BLOCKED, 0),
        "default_domain_untested": default_cat.get(CAT_UNTESTED, 0),
        "default_domain_passing": default_cat.get(CAT_PASSING, 0),
        "default_domain_verified": verified(default_cat),
        "discovered_cases": len(results),
    }
    if len(default_rows) != totals.get("spec_operators_default_domain"):
        errors.append(
            "default-domain operator rows %d != spec operators %s"
            % (len(default_rows), totals.get("spec_operators_default_domain")))
    case_counts = Counter(result["result"] for result in results)
    checks["passed_cases"] = case_counts.get(runner.PASS, 0)
    checks["failed_cases"] = sum(case_counts.get(category, 0)
                                 for category in _FAILED_RESULTS)
    checks["unrunnable_cases"] = sum(case_counts.get(category, 0)
                                     for category in _UNRUNNABLE_RESULTS)
    mode_counts = Counter(result.get("mode") for result in results)
    checks["cases_singlefile"] = mode_counts.get(discovery.MODE_SINGLEFILE, 0)
    checks["cases_split"] = mode_counts.get(discovery.MODE_SPLIT, 0)
    upstream_counts = Counter(result.get("upstream_status") for result in results)
    checks["cases_active"] = upstream_counts.get(discovery.STATUS_ACTIVE, 0)
    checks["cases_disabled"] = upstream_counts.get(discovery.STATUS_DISABLED, 0)
    checks["cases_absent"] = upstream_counts.get(discovery.STATUS_ABSENT, 0)
    for key, value in checks.items():
        if totals.get(key) != value:
            errors.append(
                f"{key}: totals={totals.get(key)} recomputed={value}")
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
    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for filename, payload in badge_payloads(report).items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written[filename] = path
    return written


_LEGEND = (
    ("Missing", "no implementation registered"),
    ("Untested", "registered, but no single-operator test exists"),
    ("Failing", "registered, tests exist, none pass"),
    ("Partial", "at least one test passes, at least one fails"),
    ("Blocked", "tests exist, but infrastructure errors leave the result "
                "unreliable"),
    ("Passing", "all applicable tests pass"),
)


def _default_domain_rows(report):
    rows = [row for row in report["operators"]
            if row["domain"] == "ai.onnx" and row["spec"]]
    rows.sort(key=lambda row: (CATEGORY_ORDER.index(row["status"]),
                               row["operator"]))
    return rows


def _required(rows):
    return [row for row in rows if row["status"] in (CAT_FAILING, CAT_BLOCKED)]


def _failure_count(row, categories):
    counts = row.get("solo_per_category", {})
    return sum(counts.get(category, 0) for category in categories)


def write_markdown(report, path):
    """Render the compact, developer-facing compatibility overview."""
    campaign = report["campaign"]
    totals = report["totals"]
    rows = _default_domain_rows(report)
    denom = totals["spec_operators_default_domain"]
    revision = campaign.get("onnx2c_commit_short") or campaign.get(
        "onnx2c_commit", "unknown")
    onnx_version = campaign.get("onnx_version", "unknown")

    lines = ["# ONNX operator compatibility", ""]

    if not campaign.get("complete", False):
        lines.extend([
            "> **Warning:** this is a partial campaign "
            f"({campaign.get('mode', 'subset')}); the matrix is incomplete.",
            "",
        ])

    lines.extend([
        "| Metric | Value |",
        "| --- | --- |",
        f"| ONNX version | {onnx_version} "
        f"(`{campaign.get('onnx_submodule_commit', 'unknown')}`) |",
        f"| onnx2c revision | `{revision}` |",
        "| Registered operators | "
        f"{_format_ratio(totals['registered_in_default_domain'], denom)} |",
        "| Verified operators | "
        f"{_format_ratio(totals['default_domain_verified'], denom)} |",
        "| Passing backend tests | "
        f"{_format_ratio(totals['passed_cases'], totals['discovered_cases'])} |",
        "| Infrastructure errors | "
        f"{totals['cases_per_category'].get(runner.INFRASTRUCTURE_ERROR, 0)} |",
        "",
        "* **Registered**: onnx2c dispatches the operator.",
        "* **Verified**: at least one single-operator backend test passes.",
        "* **Passing backend tests**: individual test cases that pass end to "
        "end.",
        "",
    ])

    lines.extend([
        "## Operator compatibility",
        "",
        "Every operator of the pinned ONNX specification's default `ai.onnx` "
        "domain, classified from its single-operator tests.",
        "",
        "| Operator | Implemented | Tests passed | Status |",
        "| --- | --- | --- | --- |",
    ])
    for row in rows:
        lines.append(
            "| %s | %s | %d | %s |"
            % (row["operator"], "yes" if row["registered"] else "no",
               row["solo_passed"], CATEGORY_LABEL[row["status"]]))

    counts = {category: sum(1 for row in rows if row["status"] == category)
              for category in CATEGORY_ORDER}
    lines.extend([
        "",
        "Status counts: "
        + ", ".join(f"{CATEGORY_LABEL[c]} {counts[c]}"
                    for c in CATEGORY_ORDER)
        + ".",
        "",
        "Legend: "
        + "; ".join(f"**{name}** \u2013 {text}" for name, text in _LEGEND)
        + ".",
        "",
    ])

    shared_failures = sum(
        1 for result in report["tests"]
        if result["result"] in _FAILED_RESULTS
        and len(set(result["operators"])) > 1)
    if shared_failures:
        lines.extend([
            f"{shared_failures} failing test cases exercise several operators "
            "at once; they are not attributed to any single implementation.",
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
    required = _required(rows)
    if required:
        for row in required:
            cells = " | ".join(
                str(_failure_count(row, categories))
                for _name, categories in FAILURE_COLUMNS)
            lines.append(f"| {row['operator']} | {cells} |")
    else:
        lines.append("| _(none)_ | "
                     + " | ".join("0" for _ in FAILURE_COLUMNS) + " |")
    lines.append("")

    lines.extend([
        "## Coverage limitations",
        "",
        "- Passing the available backend tests does not prove full ONNX "
        "compatibility.",
        "- Operator versions, attributes, data types, and tensor shapes are "
        "not exhaustively tested.",
        "- Multi-operator tests cannot be attributed to a single "
        "implementation, so they never change an operator's status.",
        "- Infrastructure errors are tooling limitations, not compiler "
        "failures, and are never counted as passing tests.",
        "- Detailed per-case diagnostics live in `operator-coverage.json` and "
        "the CI artifacts.",
        "",
    ])

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
