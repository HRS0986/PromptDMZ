"""Notebook-to-`src` wiring checks.

Notebooks contain no logic — they import and call `src/`. Nothing type-checks that boundary, so
it drifts silently: `01_scoring_kaggle.ipynb`'s P0.3 cell called `build_unified_splits(hf_token=…)`
returning an object with `.summary()` / `.assert_no_overlap()`, none of which the implemented
module has. That only surfaces mid-run on a paid GPU session, so it is asserted here instead.

Only modules whose task has landed are checked — see `IMPLEMENTED`. It is an allowlist rather
than a skip-list so that a newly implemented module is covered by adding one line, instead of
being silently exempt because nobody remembered to remove it from an exclusion set.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# Modules whose task has landed. Add one line per task; cells importing anything else are
# placeholders for unimplemented phases and are not checked yet.
IMPLEMENTED = {
    "src.templates",  # P0.2
    "src.eval.splits",  # P0.3
    "src.model_loader",  # P1.1
}

# Symbols a notebook already calls but a LATER task will provide. Named individually, with the
# task that owns them, so a forward reference stays visible instead of being hidden by a
# module-wide exemption. `test_pending_symbols_are_still_pending` fails once one lands, which is
# the prompt to delete the entry.
PENDING: dict[str, str] = {
    "src.model_loader.load_baseline_adapter": "P6.1 — load + score the merged/generalist adapter",
}


def _notebooks():
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _code_cells(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            yield i, "".join(cell["source"])


def _src_imports(source: str):
    """Yield (module, [names]) for every `from src... import ...` in a cell.

    IPython magics and shell escapes are stripped first — they are not valid Python.
    """
    cleaned = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith(("!", "%"))
    )
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            yield node.module, [alias.name for alias in node.names]


def test_notebooks_are_valid_json():
    for path in _notebooks():
        json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_every_imported_src_symbol_exists(path):
    missing = []
    for cell_index, source in _code_cells(path):
        for module_name, names in _src_imports(source):
            if module_name not in IMPLEMENTED:
                continue
            module = importlib.import_module(module_name)
            for name in names:
                if f"{module_name}.{name}" in PENDING:
                    continue
                if not hasattr(module, name):
                    missing.append(f"cell {cell_index}: {module_name}.{name}")
    assert not missing, "notebook imports symbols that do not exist:\n  " + "\n  ".join(missing)


def test_pending_symbols_are_still_pending():
    """Self-cleaning: once a forward reference lands, this fails so the entry gets removed."""
    landed = []
    for dotted, owner in PENDING.items():
        module_name, _, name = dotted.rpartition(".")
        if module_name not in IMPLEMENTED:
            continue
        if hasattr(importlib.import_module(module_name), name):
            landed.append(f"{dotted} (owned by {owner})")
    assert not landed, "PENDING entries now exist — remove them:\n  " + "\n  ".join(landed)


def test_scoring_notebook_loads_no_test_split_directly():
    """The seal at the notebook layer: notebook 01 may only reach TEST via splits.py's gate."""
    path = NOTEBOOK_DIR / "01_scoring_kaggle.ipynb"
    joined = "\n".join(source for _, source in _code_cells(path))
    assert "load_dataset" not in joined, "notebook 01 must reach Hub data through src/, not directly"
    assert 'split="test"' not in joined and "split='test'" not in joined


def test_p11_cell_asserts_its_acceptance_criteria():
    """The Kaggle run is the only place P1.1's AC can be checked, so the cell must gate on them."""
    path = NOTEBOOK_DIR / "01_scoring_kaggle.ipynb"
    joined = "\n".join(source for _, source in _code_cells(path))
    assert "report.adapter_names_loaded == sorted(ADAPTERS)" in joined
    assert "report.smoke_inference_ok" in joined
    assert "report.within_budget" in joined
    assert "p11_load_report.json" in joined
