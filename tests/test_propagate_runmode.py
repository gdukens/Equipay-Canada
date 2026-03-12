import sys
from pathlib import Path
# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat
import pytest
from scripts.propagate_runmode import apply_replacements, is_sql_cell


def make_nb(sql_cell_src, code_cell_src):
    nb = nbformat.v4.new_notebook()
    nb['cells'] = [
        nbformat.v4.new_code_cell(sql_cell_src),
        nbformat.v4.new_code_cell(code_cell_src)
    ]
    return nb


def test_is_sql_cell_detects_triple_quoted_sql():
    src = """sql = '''
    SELECT PROV, SUM(FINALWT) as total_weight
    FROM df
    GROUP BY PROV
    '''"""
    assert is_sql_cell(src)


def test_apply_replacements_skips_sql_cells_and_replaces_code_cells():
    sql_src = """sql = '''
    SELECT PROV, SUM(FINALWT) as total_weight
    FROM df
    GROUP BY PROV
    '''"""
    code_src = "if len(df) < 1000:\n    print('small')"

    nb = make_nb(sql_src, code_src)
    changed, diffs = apply_replacements(nb)
    # Only the non-SQL cell should be changed
    assert changed is True
    assert len(diffs) == 1
    idx, old, new = diffs[0]
    assert 'MIN_GROUP_N' in new
    # SQL cell remains unchanged
    assert nb['cells'][0]['source'] == sql_src


def test_apply_replacements_no_change_for_sql_like_code():
    # If SQL keywords appear but cell looks like python (has import), we should not treat it as SQL
    src = "import pandas as pd\nsql = '''SELECT * FROM df'''\nif len(df) < 1000:\n    pass"
    nb = nbformat.v4.new_notebook()
    nb['cells'] = [nbformat.v4.new_code_cell(src)]
    changed, diffs = apply_replacements(nb)
    assert changed is True
    assert len(diffs) == 1
    assert 'MIN_GROUP_N' in diffs[0][2]