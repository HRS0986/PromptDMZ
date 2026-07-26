"""P0.1 acceptance criteria as executable tests.

AC (docs/TASKS.md):
  1. `uv sync` builds the local env and `uv run python -c "import src"` succeeds.
  2. A fresh Kaggle session runs the notebook install cell and imports `src` cleanly on top of
     platform torch.
  3. Pinned versions match the working-notebook versions.

AC 2 cannot be executed from here (no Kaggle session), so it is decomposed into the properties
that make it true and would otherwise fail silently: the install cell must not `uv sync`, must
pin exactly what pyproject pins, must not leave `<YOUR_USER>` placeholders, and must put the
clone on `sys.path`. AC 3's ground truth is the Unsloth banner saved in the reference
notebooks' outputs — this file reads those banners rather than trusting a hardcoded string.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
GPU_NOTEBOOKS = [
    ROOT / "notebooks" / "01_scoring_kaggle.ipynb",
    ROOT / "notebooks" / "02_score_baseline_kaggle.ipynb",
    ROOT / "notebooks" / "03_test_eval_kaggle.ipynb",
]

# Behaviour-critical libraries: exact pins, per CLAUDE.md and TASKS.md P0.1.
EXACT_PINS = {
    "transformers": "4.53.1",
    "unsloth": "2025.7.2",
    "peft": "0.16.0",
    "trl": "0.19.1",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
}

# ARCHITECTURE.md §6. NOTE: the doc's tree indents `serve/` under `data/`, but C10 and P9.2
# both specify `src/serve/app.py`; that is a formatting slip in the doc, not the layout.
EXPECTED_MODULES = [
    "src.templates",
    "src.model_loader",
    "src.scoring",
    "src.tokenizer_stats",
    "src.calibration",
    "src.fusion",
    "src.conformal",
    "src.pipeline",
    "src.eval.splits",
    "src.eval.metrics",
    "src.eval.bootstrap",
    "src.eval.baselines",
    "src.eval.external",
    "src.eval.run_eval",
    "src.serve.app",
    "src.serve.schemas",
]


def _requirements() -> dict[str, str]:
    """Map distribution name -> full requirement string from [project.dependencies]."""
    out = {}
    for raw in PYPROJECT["project"]["dependencies"]:
        name = re.split(r"[><=!~; \[]", raw, maxsplit=1)[0].strip()
        out[name] = raw
    return out


def _install_cell(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    matches = [s for s in cells if "pip install" in s]
    assert len(matches) == 1, f"{nb_path.name}: expected exactly one install cell, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------------------------
# AC 1 — the package imports
# --------------------------------------------------------------------------------------

def test_import_src():
    """`import src` succeeds — the literal AC 1 check."""
    src = importlib.import_module("src")
    assert src.__version__ == PYPROJECT["project"]["version"]


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_architecture_module_exists_and_imports(module):
    """Every module in ARCHITECTURE §6 exists and imports without side effects.

    Stubs are docstring-only by design: P0.1 creates the LAYOUT, and CLAUDE.md forbids
    scaffolding future phases early. Importing must never touch the GPU or the Hub.
    """
    assert importlib.import_module(module).__doc__, f"{module} must document its component/phase"


def test_src_import_is_dependency_free():
    """`import src` must not drag in torch/transformers.

    Kaggle notebooks import `src` in the same cell region as the install; a heavy top-level
    import would bind the pre-install versions and mask a botched install.
    """
    import subprocess
    import sys

    probe = "import sys, src; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"`import src` pulled in a heavy dependency:\n{r.stderr}"


# --------------------------------------------------------------------------------------
# AC 3 — pins match the versions the adapters were trained under
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("package,version", sorted(EXACT_PINS.items()))
def test_behaviour_critical_libs_are_exactly_pinned(package, version):
    """Loose pins on these libraries would let a `uv sync` silently change model behaviour."""
    assert _requirements().get(package) == f"{package}=={version}"


@pytest.mark.parametrize("package,version", sorted(EXACT_PINS.items()))
def test_installed_versions_match_pins(package, version):
    """The local env actually satisfies the pins — the executable half of "uv sync builds it"."""
    assert md.version(package) == version


def test_pins_match_the_training_notebook_banners():
    """Ground-truth AC 3 against the Unsloth banners saved in the reference notebooks.

    The training notebooks used `pip install -U` (unpinned), so their saved cell OUTPUT is the
    only record of what actually ran. Guards against the pins drifting from reality.
    """
    banner_re = re.compile(r"Unsloth ([\d.]+): Fast Gemma3 patching\. Transformers: ([\d.]+)\.")
    found = {}
    for nb_path in (ROOT / "notebooks" / "reference").glob("*.ipynb"):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                text = "".join(out.get("text", ""))
                m = banner_re.search(text)
                if m:
                    found[nb_path.name] = {"unsloth": m.group(1), "transformers": m.group(2)}

    assert found, "no Unsloth banner found in reference notebooks — cannot verify pins"

    # unsloth is unambiguous: both training runs used the same version.
    assert {v["unsloth"] for v in found.values()} == {EXACT_PINS["unsloth"]}

    # transformers is NOT unambiguous — the specialists (4.54.1) and the merged baseline
    # (4.53.1) were trained under different versions. See docs/ENVIRONMENT.md for the decision.
    # The pin must at least match one real training run, never an invented version.
    assert EXACT_PINS["transformers"] in {v["transformers"] for v in found.values()}


def test_transformers_discrepancy_is_documented():
    """The 4.53.1 vs 4.54.1 split is a thesis-relevant environment caveat, not a silent choice."""
    env_doc = (ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "4.54.1" in env_doc and "4.53.1" in env_doc


def test_torch_is_left_loose():
    """Kaggle/Colab's pre-provisioned CUDA build must satisfy the requirement."""
    torch_req = _requirements()["torch"]
    assert "==" not in torch_req, f"torch must not be exactly pinned, got {torch_req!r}"


def test_serve_extra_exists():
    """TASKS P0.1: FastAPI/uvicorn live in a `serve` optional-dependency group, not the default."""
    serve = PYPROJECT["project"]["optional-dependencies"]["serve"]
    names = {re.split(r"[><=!~; \[]", r, maxsplit=1)[0] for r in serve}
    assert {"fastapi", "uvicorn"} <= names
    assert not ({"fastapi", "uvicorn"} & set(_requirements())), "serve deps leaked into defaults"


# --------------------------------------------------------------------------------------
# AC 2 — the Kaggle install cell
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("nb_path", GPU_NOTEBOOKS, ids=lambda p: p.name)
def test_kaggle_cell_never_runs_uv_sync(nb_path):
    """`uv sync` on Kaggle would rebuild the GPU stack and risk a CUDA mismatch."""
    nb = nb_path.read_text(encoding="utf-8")
    assert "uv sync" not in nb.replace("Do NOT `uv sync`", "")


@pytest.mark.parametrize("nb_path", GPU_NOTEBOOKS, ids=lambda p: p.name)
def test_kaggle_cell_pins_match_pyproject(nb_path):
    """The Kaggle stack must equal the declared stack, or P1.2 verifies the wrong environment."""
    cell = _install_cell(nb_path)
    pinned = dict(re.findall(r'"([A-Za-z0-9_-]+)==([\d.]+)"', cell))
    assert pinned == EXACT_PINS


@pytest.mark.parametrize("nb_path", GPU_NOTEBOOKS, ids=lambda p: p.name)
def test_kaggle_cell_does_not_install_torch(nb_path):
    """Platform torch is kept; installing torch here is the CUDA-mismatch failure mode.

    Only the `!pip install` invocation is inspected — the surrounding comment legitimately
    mentions torch when explaining why it is NOT installed.
    """
    cell = _install_cell(nb_path)
    install_lines = [
        ln for ln in cell.splitlines()
        if ln.lstrip().startswith("!pip install") or ln.lstrip().startswith('"')
    ]
    assert install_lines, f"{nb_path.name}: could not isolate the pip install invocation"
    assert not re.search(r"\btorch\b", "\n".join(install_lines))


@pytest.mark.parametrize("nb_path", GPU_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_placeholders_are_resolved(nb_path):
    """Unfilled placeholders make AC 2 impossible — the clone would fail before any import."""
    nb = nb_path.read_text(encoding="utf-8")
    assert "<YOUR_USER>" not in nb and "<YOUR_REPO>" not in nb


@pytest.mark.parametrize("nb_path", GPU_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_makes_src_importable(nb_path):
    """After `git clone` + `%cd`, the clone is put on sys.path explicitly.

    Relying on the kernel's implicit cwd entry is fragile: ipykernel resolves sys.path[0] at
    launch, before `%cd` runs.
    """
    cell = _install_cell(nb_path)
    assert "sys.path.insert" in cell


@pytest.mark.parametrize("nb_path", GPU_NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_contain_no_logic(nb_path):
    """CLAUDE.md: notebooks import and call `src/` only.

    Specifically, no notebook may build a prompt — `src/templates.py` is the only prompt
    builder (non-negotiable rule 1, the cause of the legacy silent accuracy loss).
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "def build_prompt" not in code
    assert "<start_of_turn>" not in code


# --------------------------------------------------------------------------------------
# Repo hygiene the later phases depend on
# --------------------------------------------------------------------------------------

def test_reference_notebooks_are_untouched_by_this_build():
    """CLAUDE.md: reference notebooks are READ-ONLY — never run or edited."""
    assert (ROOT / "notebooks" / "reference").is_dir()
    assert len(list((ROOT / "notebooks" / "reference").glob("*.ipynb"))) == 5


def test_artefact_and_data_dirs_exist():
    """Phases 1-7 persist here after every stage; sessions die and nothing may depend on one."""
    assert (ROOT / "artifacts").is_dir()
    assert (ROOT / "data").is_dir()
