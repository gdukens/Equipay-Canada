Proposal: Hardening `scripts/propagate_runmode.py`

Goal
----
Avoid accidental modification of SQL templates, f-strings, and triple-quoted SQL blocks when performing automated replacements across notebooks.

Plan
----
1. Implement heuristics to skip replacements in cells that appear to be SQL:
   - Skip cells containing SQL keywords (SELECT, FROM, HAVING) OR cells with triple-quoted strings that contain SQL patterns.
2. Use Python AST parsing to identify assignments and function call keyword arguments and apply replacements only in those contexts.
3. Add `--dry-run --show-diff` options that output patch-like diffs in a readable format for reviewers.
4. Add unit tests under `tests/test_propagate_runmode.py` to assert:
   - SQL cells are not modified
   - Code cells are modified as expected
   - Diff output is informative

Testing
-------
- Add unit tests and use a sample notebook fixture with separate SQL and code cells.
- Run tests in CI and on local dev before applying changes.

Notes
-----
This approach trades off a small amount of coverage for much lower risk of corrupting SQL/f-strings, which previously caused parser errors and runtime failures in notebooks. Implementing AST parsing reduces false positives significantly and is worth the investment.