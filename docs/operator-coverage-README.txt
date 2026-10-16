Operator coverage artifacts
==========================

- docs/operator-coverage.md — the compact ONNX operator compatibility report
  (committed snapshot; the workflow publishes the fresh copy to the
  `coverage-data` branch).
- test/operator_coverage/ — tool implementation (CLI, runner, discovery,
  aggregation, reporting). Run: `python3 test/operator_coverage/run_coverage.py --help`
- .github/workflows/operator-coverage.yml — builds, runs exhaustive coverage,
  validates, uploads the full artifact, and publishes the report + badge JSON to
  the `coverage-data` branch on schedule/manual dispatch. It never pushes
  generated data back to the source branch.

Report
------
- The report has one summary table (ONNX version, onnx2c revision, registered,
  verified, passing backend tests, infrastructure errors), one complete
  `ai.onnx` operator matrix, a compact failure table, and a limitations note.
- `docs/operator-coverage.md` is generated from `operator-coverage.json` by
  `report.write_markdown`; the JSON remains the machine-readable source of truth.

Classification
--------------
- Every operator of the pinned ONNX specification's default `ai.onnx` domain is
  classified exactly once, from its **single-operator** tests only:
  Missing (not registered) → Untested (registered, no such test) → Failing
  (tests exist, none pass) → Partial (some pass, some fail) → Blocked
  (infrastructure errors leave the result unreliable) → Passing (all pass).
- A multi-operator test cannot be attributed to a single implementation, so a
  failure there never changes an operator's status; the count of such failing
  cases is reported separately.

Notes
-----
- Exhaustive runs the entire ONNX backend node test corpus (v1.19.0) against
  onnx2c + onnx2c-generated C + reference comparison.
- Every case runs in the mode test/CMakeLists.txt registers it in (singlefile via
  testgen_singlefile, otherwise split via onnx2c + testgen). Tests upstream does
  not register run in split mode. A cross-check re-runs each case in the other
  mode and lists mode-sensitive cases; it never changes the primary result.
- Infrastructure errors (testgen/harness limitations, e.g. float16/castlike
  cases) are reported separately and never counted as passing tests. A campaign
  can be complete (all selected cases ran) while not infrastructure clean.
- Default CC flags for corpus runs: `-Wall -Wno-unused-variable` (no -Werror) to
  avoid warning-only misclassification.

README badge URLs
-----------------
- Shields.io endpoint badges require an absolute URL, so README.md points at
  `https://raw.githubusercontent.com/<owner>/<repo>/coverage-data/badges/*.json`
  and links each badge to
  `https://github.com/<owner>/<repo>/blob/coverage-data/operator-coverage.md`.
- These currently use the fork owner (`mischirmer`). When the feature lands
  upstream, change only the owner segment to `kraiskil` in the three badge URLs
  and the report link near the top of README.md. No implementation file embeds a
  fork name.
