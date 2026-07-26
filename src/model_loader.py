"""C1 — Model + adapter loading (once per session).

Loads the backbone `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` in 4-bit NF4 — the TRAINING
checkpoint, not plain `google/gemma-3-1b-it` in fp16, which is what the legacy Evaluation
notebook used. That train/eval discrepancy is why legacy metrics are not comparable to new
ones (ARCHITECTURE §0). All new scoring is 4-bit.

Attaches the three private Hub adapters as named adapters on one PeftModel, all
memory-resident for the whole session, never merged:

    role_violation        <- hirushafernando/fyp-gemma3-1b-slm-a-qlora
    privilege_escalation  <- hirushafernando/fyp-gemma3-1b-slm-b-qlora
    obfuscation_evasion   <- hirushafernando/fyp-gemma3-1b-slm-c-qlora

Requires HF_TOKEN (repos are private).

Two deliberate deviations from `Final Inference Pipeline.ipynb`
---------------------------------------------------------------
That notebook is the structural reference, but two of its constants are stale and copying them
would silently invalidate results:

1. It sets ``BASE_MODEL = "google/gemma-3-1b-it"`` and loads fp16. Every adapter's
   `adapter_config.json` records `base_model_name_or_path = unsloth/gemma-3-1b-it-unsloth-bnb-4bit`,
   so the 4-bit checkpoint is what they were actually trained against (§5 pitfall 10).
2. It attaches the older `slm-shield-*-qlora` adapters under the adapter name ``"obfuscation"``.
   Both generations of repo still exist on the Hub, so existence proves nothing — the `fyp-*`
   set is the one trained on the `fyp-slm-*` corpus. Adapter names are imported from
   `templates.ADAPTERS` rather than retyped, because a name that disagrees with the templates
   module misroutes every prompt to the wrong specialist without raising anything.

No CPU fallback, by design: 4-bit bnb requires CUDA, and a non-quantised CPU load would break
the "4-bit everywhere" standard that the deployability claims rest on. `load_model_with_adapters`
raises rather than silently degrading.

Implemented by P1.1. Notebook import surface: `load_model_with_adapters`, `report_load`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.templates import ADAPTERS, build_prompt

log = logging.getLogger(__name__)

# The training checkpoint. NOT google/gemma-3-1b-it — see module docstring.
BACKBONE = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"

ADAPTER_REPOS: dict[str, str] = {
    "role_violation": "hirushafernando/fyp-gemma3-1b-slm-a-qlora",
    "privilege_escalation": "hirushafernando/fyp-gemma3-1b-slm-b-qlora",
    "obfuscation_evasion": "hirushafernando/fyp-gemma3-1b-slm-c-qlora",
}

# The merged/generalist adapter is the experimental CONTROL, not part of the running system.
# Phase 6 (P6.1) loads and scores it separately; it is never attached here.
BASELINE_ADAPTER_REPO = "hirushafernando/fyp-gemma3-1b-slm-merged-qlora"

# ARCHITECTURE §2 C1: ~1B params at 4-bit plus three r=16 adapters.
VRAM_BUDGET_GB = 3.0

SMOKE_PROMPT = "What is the capital of France?"


class ModelLoadError(RuntimeError):
    """Raised when the runtime cannot support the loading contract."""


@dataclass
class LoadReport:
    """AC evidence for P1.1, emitted as JSON so a Kaggle run leaves reviewable proof."""

    backbone: str
    adapter_repos: dict[str, str]
    adapter_names_loaded: list[str]
    active_adapter: str
    peak_vram_gb: float
    vram_budget_gb: float
    within_budget: bool
    load_seconds: float
    smoke_inference_ok: bool
    smoke_logits_shape: list[int]
    training_mode: bool
    device: str
    gpu_name: str
    library_versions: dict[str, str] = field(default_factory=dict)


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise ModelLoadError(
            "CUDA is not available. C1 loads a 4-bit bnb checkpoint, which requires a GPU; "
            "there is no CPU fallback by design (a non-quantised CPU load would break the "
            "4-bit-everywhere standard). Run this on the Kaggle T4 session."
        )


def build_quant_config():
    """4-bit NF4 with double quantisation and fp16 compute — the training configuration."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer(hf_token: str | None = None):
    """Tokenizer for the backbone, configured for LEFT padding.

    Left padding is set here rather than in `scoring.py` because C5 extracts the label logit
    from the last non-pad position: with right padding that position is a pad token and every
    probability would be read off the wrong slot. Setting it on the handle this module returns
    means no caller can forget.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(hf_token: str | None = None):
    """Load the 4-bit backbone. Caller must have checked CUDA."""
    import torch
    from transformers import AutoModelForCausalLM

    # `torch_dtype`, NOT `dtype`. transformers 4.53.1 (the pinned version, and what Kaggle runs)
    # does not recognise `dtype` — it falls through to the model constructor and raises
    # `Gemma3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'`. Newer versions
    # accept both, so `torch_dtype` is the spelling that works across the range.
    # `Final Inference Pipeline.ipynb` uses `dtype=` and was evidently run on a newer stack.
    return AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        quantization_config=build_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )


def attach_adapters(base_model, hf_token: str | None = None):
    """Attach all three adapters as named adapters on one PeftModel. No merging."""
    from peft import PeftModel

    names = list(ADAPTERS)
    first, rest = names[0], names[1:]

    model = PeftModel.from_pretrained(
        base_model, ADAPTER_REPOS[first], adapter_name=first, token=hf_token
    )
    log.info("attached adapter %s <- %s", first, ADAPTER_REPOS[first])

    for name in rest:
        model.load_adapter(ADAPTER_REPOS[name], adapter_name=name, token=hf_token)
        log.info("attached adapter %s <- %s", name, ADAPTER_REPOS[name])

    return model


def loaded_adapter_names(model) -> list[str]:
    """Adapter names PEFT reports as attached, sorted for stable comparison."""
    return sorted(model.peft_config.keys())


def peak_vram_gb() -> float:
    import torch

    return torch.cuda.max_memory_allocated() / 1e9


def smoke_inference(model, tokenizer) -> tuple[bool, list[int]]:
    """One forward pass through a real templated prompt.

    A forward pass, not `generate`: C5 reads label logits from a single forward, so this
    exercises the path that actually matters. Prompt comes from `templates.build_prompt` —
    this module never builds prompt strings itself (non-negotiable rule 1).
    """
    import torch

    prompt = build_prompt(SMOKE_PROMPT, ADAPTERS[0])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model(**inputs)

    logits = out.logits
    ok = bool(torch.isfinite(logits).all())
    return ok, list(logits.shape)


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing optional lib must not fail the report
            versions[name] = "unavailable"
    return versions


def load_model_with_adapters(hf_token: str | None = None):
    """Load backbone + three resident adapters. Returns `(model, tokenizer)`.

    Raises `ModelLoadError` if CUDA is unavailable, or if PEFT does not report exactly the
    three expected adapter names after loading.
    """
    import torch

    _require_cuda()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    tokenizer = load_tokenizer(hf_token)
    base_model = load_backbone(hf_token)
    model = attach_adapters(base_model, hf_token)
    elapsed = time.perf_counter() - started

    # Explicit, not assumed: in training mode LoRA dropout is live, which makes every forward
    # pass nondeterministic and would surface as a batched-vs-sequential disagreement in P1.2.
    model.eval()

    names = loaded_adapter_names(model)
    if names != sorted(ADAPTERS):
        raise ModelLoadError(
            f"PEFT reports adapters {names}, expected {sorted(ADAPTERS)}. "
            "Scoring would misroute prompts to the wrong specialist."
        )

    log.info(
        "loaded %s + %d adapters in %.1fs, peak VRAM %.2f GB",
        BACKBONE,
        len(names),
        elapsed,
        peak_vram_gb(),
    )
    model._p11_load_seconds = elapsed  # carried into report_load
    return model, tokenizer


def report_load(model, tokenizer, out_path: Path | str | None = None) -> LoadReport:
    """Produce the P1.1 acceptance evidence and optionally persist it as JSON."""
    import torch

    ok, shape = smoke_inference(model, tokenizer)
    peak = peak_vram_gb()

    report = LoadReport(
        backbone=BACKBONE,
        adapter_repos=ADAPTER_REPOS,
        adapter_names_loaded=loaded_adapter_names(model),
        active_adapter=str(model.active_adapter),
        peak_vram_gb=round(peak, 3),
        vram_budget_gb=VRAM_BUDGET_GB,
        within_budget=peak < VRAM_BUDGET_GB,
        load_seconds=round(getattr(model, "_p11_load_seconds", float("nan")), 2),
        smoke_inference_ok=ok,
        smoke_logits_shape=shape,
        training_mode=bool(model.training),
        device=str(model.device),
        gpu_name=torch.cuda.get_device_name(0),
        library_versions=_library_versions(),
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
        log.info("wrote %s", out_path)

    return report
