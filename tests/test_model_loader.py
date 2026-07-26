"""P1.1 — model loader tests.

The acceptance criteria (T4 load, peak VRAM, PEFT adapter listing, smoke inference) are all
GPU-bound and are verified by the Kaggle run of `01_scoring_kaggle.ipynb`, which persists
`manifests/p11_load_report.json`. What is testable without a GPU is the part that silently
corrupts results rather than crashing: the constants. A wrong backbone id or a wrong adapter
name produces plausible numbers from the wrong model, so those are pinned here.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src import model_loader
from src.model_loader import (
    ADAPTER_REPOS,
    BACKBONE,
    BASELINE_ADAPTER_REPO,
    SMOKE_PROMPT,
    VRAM_BUDGET_GB,
    LoadReport,
    ModelLoadError,
    build_quant_config,
    load_model_with_adapters,
    load_tokenizer,
)
from src.templates import ADAPTERS, INSTRUCTION_PREFIX, build_prompt

# --- constants that must not drift -------------------------------------------------------


def test_backbone_is_the_4bit_training_checkpoint():
    """§5 pitfall 10: the legacy notebook loaded fp16, which is not what the adapters saw."""
    assert BACKBONE == "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
    assert "google/gemma-3-1b-it" != BACKBONE
    assert "4bit" in BACKBONE


def test_adapter_names_match_the_templates_module_exactly():
    """A name mismatch misroutes every prompt to the wrong specialist without raising."""
    assert sorted(ADAPTER_REPOS) == sorted(ADAPTERS)
    assert sorted(ADAPTER_REPOS) == sorted(INSTRUCTION_PREFIX)


def test_legacy_adapter_name_is_not_used():
    """`Final Inference Pipeline.ipynb` attaches this one as bare 'obfuscation'."""
    assert "obfuscation" not in ADAPTER_REPOS
    assert "obfuscation_evasion" in ADAPTER_REPOS


def test_adapter_repos_are_the_fyp_generation():
    """Both generations exist on the Hub, so only the spec distinguishes them."""
    for name, repo in ADAPTER_REPOS.items():
        assert repo.startswith("hirushafernando/fyp-gemma3-1b-slm-"), f"{name}: {repo}"
        assert "slm-shield-" not in repo, f"{name}: legacy repo {repo}"
    assert ADAPTER_REPOS["role_violation"].endswith("-a-qlora")
    assert ADAPTER_REPOS["privilege_escalation"].endswith("-b-qlora")
    assert ADAPTER_REPOS["obfuscation_evasion"].endswith("-c-qlora")


def test_baseline_adapter_is_never_attached_in_the_core_path():
    """The generalist is the experimental control (P6.1), not part of the running system."""
    assert BASELINE_ADAPTER_REPO not in ADAPTER_REPOS.values()
    assert "merged" in BASELINE_ADAPTER_REPO


def test_vram_budget_matches_the_architecture_expectation():
    assert VRAM_BUDGET_GB == 3.0


# --- quantisation config ------------------------------------------------------------------


def _call_kwargs(func, callee_attr: str) -> set[str]:
    """Keyword names passed to `<something>.<callee_attr>(...)` inside `func`'s source."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == callee_attr
        ):
            names |= {kw.arg for kw in node.keywords if kw.arg}
    return names


def test_backbone_load_uses_the_kwarg_spelling_the_pinned_transformers_accepts():
    """Regression: `dtype=` is silently forwarded to the model constructor and raises.

    transformers 4.53.1 — the version pinned in `pyproject.toml` and installed on Kaggle —
    recognises only `torch_dtype`. Passing `dtype` produces
    `Gemma3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'`, and it surfaces
    on the GPU session rather than here, which is exactly what this test is for.
    """
    kwargs = _call_kwargs(model_loader.load_backbone, "from_pretrained")
    assert "torch_dtype" in kwargs
    assert "dtype" not in kwargs


def test_installed_transformers_matches_the_pin():
    """The local API surface must equal Kaggle's, or these static checks prove nothing."""
    import re

    import transformers

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    pinned = re.search(r'"transformers==([\d.]+)"', pyproject)
    assert pinned, "transformers pin not found in pyproject.toml"
    assert transformers.__version__ == pinned.group(1), (
        f"installed transformers {transformers.__version__} != pinned {pinned.group(1)}; "
        "kwarg-compatibility checks in this file no longer reflect the Kaggle runtime"
    )


def test_quant_config_reproduces_the_training_quantisation():
    import torch

    cfg = build_quant_config()
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == torch.float16


# --- no CPU fallback, by design ------------------------------------------------------------


def test_load_raises_a_clear_error_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelLoadError, match="CUDA is not available"):
        load_model_with_adapters(hf_token=None)


def test_no_cpu_fallback_path_exists():
    """A silent CPU downgrade would break the 4-bit-everywhere standard."""
    source = inspect.getsource(model_loader)
    assert 'device_map="cpu"' not in source
    assert "load_in_4bit=True" in source


# --- rule 1: templates.py is the only prompt builder ---------------------------------------


def test_loader_builds_no_prompt_strings_of_its_own():
    """Non-negotiable rule 1 — the smoke test must go through templates.build_prompt."""
    source = inspect.getsource(model_loader)
    for fragment in (
        "<start_of_turn>",
        "<end_of_turn>",
        "User Prompt:",
        "Respond with exactly one word",
    ):
        assert fragment not in source, f"loader hardcodes template fragment {fragment!r}"
    assert "build_prompt" in source


def test_smoke_prompt_is_benign_and_templated():
    """The smoke prompt is raw text; templating happens in templates.build_prompt."""
    assert "<start_of_turn>" not in SMOKE_PROMPT
    templated = build_prompt(SMOKE_PROMPT, ADAPTERS[0])
    assert templated.endswith("<start_of_turn>model\n        ")
    assert SMOKE_PROMPT in templated


# --- tokenizer configuration ---------------------------------------------------------------


def test_tokenizer_uses_left_padding_and_has_a_pad_token():
    """C5 reads the last non-pad logit; right padding would read a pad slot instead."""
    tokenizer = load_tokenizer()
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token is not None


def test_tokenizer_is_the_backbone_tokenizer():
    tokenizer = load_tokenizer()
    assert BACKBONE.split("/")[-1] in str(tokenizer.name_or_path) or tokenizer.name_or_path == BACKBONE


# --- report shape ---------------------------------------------------------------------------


def test_load_report_carries_every_acceptance_field():
    fields = set(LoadReport.__dataclass_fields__)
    for required in (
        "adapter_names_loaded",
        "peak_vram_gb",
        "within_budget",
        "smoke_inference_ok",
        "backbone",
        "library_versions",
    ):
        assert required in fields
