Operator coverage artifacts
==========================

- docs/operator-coverage.md — example report (committed locally, not auto-pushed by CI on every run).
- test/operator_coverage/ — tool implementation (CLI, runner, discovery, aggregation, reporting). Run: `python3 test/operator_coverage/run_coverage.py --help`
- .github/workflows/operator-coverage.yml — builds, runs exhaustive coverage, validates, and publishes to GitHub Pages on schedule/manual dispatch (never pushes generated large logs back to the source branch).

Notes:
- Exhaustive runs the entire ONNX backend node test corpus (v1.19.0) against onnx2c + onnx2c-generated C + reference comparison.
- Every case runs in the mode test/CMakeLists.txt registers it in (singlefile via testgen_singlefile, otherwise split via onnx2c + testgen). Tests upstream does not register run in split mode. A cross-check re-runs each case in the other mode and lists mode-sensitive cases; it never changes the primary result.
- Infrastructure errors (testgen/harness limitations, e.g. float16/castlike cases) are reported separately and never counted as passing tests. A campaign can be complete (all selected cases ran) while not infrastructure clean.
- Default CC flags for corpus runs: `-Wall -Wno-unused-variable` (no -Werror) to avoid warning-only misclassification.
- Generated badges (shields endpoint JSON) are published to GitHub Pages under `badges/*.json`. The workflow derives the base URL from `$GITHUB_REPOSITORY` (owner + repo) so it is correct for schedule and manual dispatch alike.

README badge URLs:
- Shields.io endpoint badges require an absolute URL, so README.md points at `https://<owner>.github.io/<repo>/badges/*.json` and links to `https://<owner>.github.io/<repo>/`.
- These currently use the fork owner (`mischirmer`). When the feature lands upstream, change only the owner segment to `kraiskil` in the three badge URLs near the top of README.md. No implementation file embeds a fork name.
