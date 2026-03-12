<!--
PR Template for: Propagate runmode hardening
-->

## Summary

This PR hardens `scripts/propagate_runmode.py` to avoid accidental edits to SQL/f-strings and adds a robust `--dry-run --show-diff` mode and unit tests.

## What's included

- Heuristics to skip code cells that look like SQL when applying replacements
- AST-based detection for assignment/keyword contexts before substitution
- Unit tests covering SQL-preservation and dry-run output
- Documentation and example of the safer workflow

## Follow-up work
- Extend heuristics to handle multi-language notebooks
- Add integration test that runs the propagate script in dry-run and verifies no notebook diffs
