"""
Script to propagate RUN-MODE (EQUIPAY_MODE) and replace heavy numeric defaults in all notebooks.
- Inserts a RUN-MODE markdown + code cell at top if missing
- Inserts a RUN-MODE utilities code cell defining enforce_fast_sample, MIN_GROUP_N
- Replaces numeric defaults (n_bootstrap, n_estimators, max_iter, group-size thresholds) with variables

Use: python scripts/propagate_runmode.py --apply

NOTE: Run in a branch or with git to review changes; this script writes notebooks in-place.
"""
import re
import nbformat
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / 'notebooks'

def ensure_runmode_cells(nb):
    """Insert header and utility cells at top if missing."""
    # Check for a cell mentioning EQUIPAY_MODE
    if any('EQUIPAY_MODE' in (cell.get('source') or '') for cell in nb['cells']):
        return False

    header_md = nbformat.v4.new_markdown_cell("""# ⚡ Quick Run Mode: Fast vs Full\n\nThis notebook supports a global run mode (EQUIPAY_MODE) for FAST/FULL runs.\n""")

    header_code = nbformat.v4.new_code_cell(
        """# GLOBAL RUN MODE (inserted)
import os
EQUIPAY_MODE = os.environ.get('EQUIPAY_MODE', 'FAST')  # FAST | FULL
if EQUIPAY_MODE == 'FAST':
    N_SAMPLES = 1000
    N_BOOTSTRAP = 100
    MAX_ITER = 200
    N_ESTIMATORS = 50
    PLOT_INLINE = False
else:
    N_SAMPLES = None
    N_BOOTSTRAP = 1000
    MAX_ITER = 1000
    N_ESTIMATORS = 200
    PLOT_INLINE = True
print(f"EQUIPAY_MODE={EQUIPAY_MODE}; N_SAMPLES={N_SAMPLES}; N_BOOTSTRAP={N_BOOTSTRAP}")

# Central helpers: ensure a store and a small sample are available for FAST-mode smoke tests
try:
    from src.notebook_utils import ensure_store_and_sample, get_sample_from_store, safe_weight_col
    store, df_sample = ensure_store_and_sample()
    weight_col = safe_weight_col(df_sample)
except Exception as e:
    print(f"Could not import notebook helpers: {e}")
""")

    util_code = nbformat.v4.new_code_cell(
        """# RUN-MODE UTILITIES (inserted)
def enforce_fast_sample(df, n=None, seed=42):
    if EQUIPAY_MODE == 'FAST' and n is not None:
        return df.sample(n=min(len(df), n), random_state=seed)
    return df

# Conservative default: in FAST mode we skip small groups to avoid heavy per-group loops.
# In FULL mode we allow small groups to be processed for comprehensive analysis.
MIN_GROUP_N = 1000 if EQUIPAY_MODE == 'FAST' else 30
print(f"MIN_GROUP_N={MIN_GROUP_N}")
""")

    nb['cells'].insert(0, util_code)
    nb['cells'].insert(0, header_code)
    nb['cells'].insert(0, header_md)
    return True

# Replacement patterns (simple, conservative regexes)
REPLACEMENTS = [
    (re.compile(r"\bn_bootstrap\s*=\s*\d+\b"), "n_bootstrap = N_BOOTSTRAP"),
    (re.compile(r"\bn_bootstraps\s*=\s*\d+\b"), "n_bootstraps = N_BOOTSTRAP"),
    (re.compile(r"\bn_estimators\s*=\s*\d+\b"), "n_estimators=N_ESTIMATORS"),
    (re.compile(r"\bmax_iter\s*=\s*\d+\b"), "max_iter=MAX_ITER"),
    # Precise pattern: only replace if condition ends with ':' to avoid touching other expressions
    (re.compile(r"if\s+len\(([^)]+)\)\s*<\s*1000\s*:"), r"if len(\1) < MIN_GROUP_N:"),
    (re.compile(r"if\s+len\(([^)]+)\)\s*<=\s*1000\s*:"), r"if len(\1) <= MIN_GROUP_N:"),
    # Update existing MIN_GROUP_N definitions (old variant used 'FULL' in condition) to conservative FAST-mode defaults
    (re.compile(r"MIN_GROUP_N\s*=\s*1000\s*if\s*EQUIPAY_MODE\s*==\s*['\"]FULL['\"]\s*else[^\n]+"), "MIN_GROUP_N = 1000 if EQUIPAY_MODE == 'FAST' else 30"),
    # NOTE: Skipping direct replacement inside SQL HAVING clauses to avoid introducing placeholders.
]


def is_sql_cell(src: str) -> bool:
    """Heuristic: detect cells that are primarily SQL to avoid editing them.
    - If the cell contains triple-quoted string with SQL keywords, treat as SQL.
    - If the cell contains SQL keywords and does not look like Python code (no 'def'/'import'), treat as SQL.
    """
    if not src or not isinstance(src, str):
        return False
    low = src.lower()
    # Quick check for triple-quoted SQL blocks
    if "'''" in src or '"""' in src:
        # If inside triple quotes we see SELECT or FROM and the cell doesn't contain Python constructs, assume SQL
        if re.search(r"\bselect\b|\bfrom\b|\bwhere\b", low) and not re.search(r"\bdef\b|\bimport\b|\bclass\b", src):
            return True
    # Otherwise, if SQL keywords are present and it doesn't look like Python code, treat as SQL
    sql_kw = re.search(r"\b(select|from|where|group by|order by|having)\b", low)
    if sql_kw and not re.search(r"\bdef\b|\bimport\b|\bclass\b|\b=\b", src):
        return True
    return False


def apply_replacements(nb):
    """Apply replacements to code cells but skip cells that look like SQL. Returns a tuple:
    (changed: bool, diffs: list of (cell_index, old_src, new_src)).
    """
    changed = False
    diffs = []
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        # Skip SQL-like cells
        if is_sql_cell(src):
            continue
        new = src
        matched = []
        for pattern, repl in REPLACEMENTS:
            if pattern.search(src):
                matched.append(pattern.pattern)
            new = pattern.sub(repl, new)
        # Fix any small typos introduced by automated replacement passes
        new = new.replace('MIN_GROUP_N0', 'MIN_GROUP_N')
        if new != src:
            if matched:
                print(f"apply_replacements: matched patterns: {matched}")
            diffs.append((idx, src, new))
            cell['source'] = new
            changed = True
    return changed, diffs


def process_all(apply=False, backup=False, show_diff=False):
    notebooks = list(NOTEBOOKS_DIR.glob('*.ipynb'))
    modified = []
    backup_dir = NOTEBOOKS_DIR.parent / 'notebook_backups'
    if backup and apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
    for nb_file in notebooks:
        nb = nbformat.read(nb_file, as_version=4)
        updated = False
        updated |= ensure_runmode_cells(nb)
        # If the notebook already contained a run-mode header, ensure it imports central helpers
        if any('EQUIPAY_MODE' in (cell.get('source') or '') for cell in nb['cells']) and not any('from src.notebook_utils' in (cell.get('source') or '') for cell in nb['cells']):
            # Insert import cell immediately after the run-mode header cell
            for i, cell in enumerate(nb['cells']):
                if 'EQUIPAY_MODE' in (cell.get('source') or ''):
                    import_code = nbformat.v4.new_code_cell(
                        """# Central helpers for FAST-mode smoke tests
try:
    from src.notebook_utils import ensure_store_and_sample, get_sample_from_store, safe_weight_col
    store, df_sample = ensure_store_and_sample()
    weight_col = safe_weight_col(df_sample)
except Exception as e:
    print(f"Could not import notebook helpers: {e}")
""")
                    nb['cells'].insert(i+1, import_code)
                    updated = True
                    break
        # apply replacements and capture diffs
        rep_changed, rep_diffs = apply_replacements(nb)
        updated |= rep_changed
        if updated:
            modified.append(str(nb_file))
            # If show_diff requested, print diffs for reviewer
            if rep_diffs and show_diff:
                import difflib
                for idx, old, new in rep_diffs:
                    print(f"--- Notebook: {nb_file.name} | Cell index: {idx}")
                    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=''):
                        print(line)
            if apply:
                if backup:
                    # write backup copy first
                    dest = backup_dir / (nb_file.name + '.bak')
                    nbformat.write(nbformat.read(nb_file, as_version=4), dest)
                    print(f"Backup written: {dest}")
                nbformat.write(nb, nb_file)
    return modified

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Write changes to notebooks')
    parser.add_argument('--backup', action='store_true', help='Write a backup copy of notebooks before applying changes (only valid with --apply)')
    parser.add_argument('--show-diff', action='store_true', help='Show diffs for proposed changes (dry-run mode)')
    args = parser.parse_args()

    if args.backup and not args.apply:
        parser.error('--backup requires --apply')

    mods = process_all(apply=args.apply, backup=args.backup, show_diff=args.show_diff)
    print('Notebooks modified:')
    for m in mods:
        print(' -', m)
    if not mods:
        print('No changes required.')
