"""P1.2 — scorer tests.

The headline AC (batched vs sequential within fp16 tolerance on >=200 prompts) needs the loaded
model and is verified by the Kaggle run, which writes `manifests/p12_gate_report.json`.

Everything else is provable here and is: the label-id derivation AC runs on the real Hub rows
with only a tokenizer, and the extraction plumbing is driven by a stub model with deterministic
logits. That plumbing is where a silent error lives — a wrong padding side or a transposed
index yields plausible probabilities from the wrong vocabulary slot, which no GPU run would
flag either.
"""

from __future__ import annotations

import pytest

from src.scoring import (
    GATE_MIN_PROMPTS,
    GATE_TOLERANCE,
    MIN_LABEL_ID_ROWS,
    LabelIds,
    ScoringError,
    build_batch,
    derive_label_ids,
    last_position_logits,
    logits_to_prob,
    score_batched,
    score_sequential,
    verify_batched_vs_sequential,
)
from src.templates import (
    ADAPTERS,
    DATASET_REPOS,
    LABEL_BENIGN,
    LABEL_INJECTION,
    build_formatted_text,
    build_prompt,
)
from tests.conftest import load_hf_token

pytest_plugins = ()

SEED = 42


# --- stub model ----------------------------------------------------------------------------


class StubModel:
    """Deterministic stand-in for the PeftModel.

    Emits logits that depend on the row's assigned adapter and on its real token content, so a
    batched call and a sequential call only agree if rows are assembled, padded, and indexed
    consistently. Records how it was called.
    """

    def __init__(self, vocab: int = 300):
        self.vocab = vocab
        self.device = "cpu"
        self.training = False  # the loader calls model.eval(); mirror that
        self.active = ADAPTERS[0]
        self.calls: list[dict] = []
        self.supports_adapter_names = True
        self.peft_config = dict.fromkeys(ADAPTERS, None)

    def set_adapter(self, name: str) -> None:
        self.active = name

    def __call__(
        self, input_ids=None, attention_mask=None, position_ids=None, adapter_names=None, **_
    ):
        import torch

        if adapter_names is not None and not self.supports_adapter_names:
            raise TypeError("got an unexpected keyword argument 'adapter_names'")

        names = adapter_names or [self.active] * input_ids.shape[0]
        self.calls.append(
            {
                "rows": input_ids.shape[0],
                "adapter_names": adapter_names,
                "position_ids": None if position_ids is None else position_ids.clone(),
            }
        )

        batch, length = input_ids.shape
        # Mirror the real model: absent position_ids default to arange for EVERY row, which is
        # what makes left padding shift RoPE. The stub must be position-sensitive or it cannot
        # reproduce the divergence the P1.2 gate found on the T4.
        if position_ids is None:
            position_ids = torch.arange(length).unsqueeze(0).expand(batch, -1)

        logits = torch.zeros(batch, length, self.vocab)
        for row in range(batch):
            offset = ADAPTERS.index(names[row]) + 1
            # Content-dependent: real tokens only. If padding leaked into the last position, or
            # rows were mis-paired with adapters, the two paths diverge.
            real = input_ids[row][attention_mask[row] == 1]
            # Centred on 0 so d spans both signs and p straddles 0.5. Without this every row
            # sits on one side of the boundary and decision-flip tests pass only by accident.
            signal = float(real.sum() % 97) / 97.0 - 0.5
            # Position-dependent, standing in for RoPE: the position of the FIRST real token.
            first_real_pos = float(position_ids[row][attention_mask[row] == 1][0])
            logits[row, -1, 10] = offset + signal + first_real_pos  # INJ slot
            logits[row, -1, 20] = offset - signal                   # BEN slot
        return type("Out", (), {"logits": logits})()


@pytest.fixture
def stub_tokenizer():
    """Real backbone tokenizer, left-padded — the same object the loader returns."""
    from src.model_loader import load_tokenizer

    return load_tokenizer()


# --- label-id derivation (the locally verifiable AC) ----------------------------------------


requires_hub = pytest.mark.skipif(
    load_hf_token() is None,
    reason="HF_TOKEN not available; label-id derivation needs the private Hub datasets",
)


@pytest.fixture(scope="session")
def hub_rows_by_adapter():
    """>=100 real stored rows per adapter, from TRAIN (VAL stays clean, TEST stays sealed)."""
    from datasets import load_dataset

    token = load_hf_token()
    out = {}
    for adapter, repo in DATASET_REPOS.items():
        ds = load_dataset(repo, split="train", token=token)
        sample = ds.shuffle(seed=SEED).select(range(200))
        out[adapter] = list(zip(sample["formatted_text"], sample["label"]))
    return out


@requires_hub
def test_ac_label_ids_derived_from_real_rows(stub_tokenizer, hub_rows_by_adapter):
    """AC: INJ_ID != BEN_ID, stable across >=100 real rows per adapter, identical across all."""
    ids = derive_label_ids(stub_tokenizer, hub_rows_by_adapter)

    assert ids.inj_id != ids.ben_id
    assert ids.identical_across_adapters
    for adapter in ADAPTERS:
        assert ids.rows_per_adapter[adapter] >= MIN_LABEL_ID_ROWS
        assert ids.per_adapter_ids[adapter] == {"inj_id": ids.inj_id, "ben_id": ids.ben_id}


@requires_hub
def test_derived_ids_match_in_context_tokenisation(stub_tokenizer, hub_rows_by_adapter):
    """The derived id must equal what the tokenizer emits after the real prompt prefix.

    Guards the SentencePiece merge hazard: measured on this tokenizer the trailing whitespace
    does NOT merge with the label word, but that is a result to re-verify, not an assumption.
    """
    ids = derive_label_ids(stub_tokenizer, hub_rows_by_adapter)

    for adapter in ADAPTERS:
        prefix = build_prompt("some prompt", adapter)
        prefix_ids = stub_tokenizer(prefix)["input_ids"]
        for label, expected in ((1, ids.inj_id), (0, ids.ben_id)):
            full = build_formatted_text("some prompt", adapter, label).removeprefix("<bos>")
            full_ids = stub_tokenizer(full)["input_ids"]
            assert full_ids[: len(prefix_ids)] == prefix_ids
            assert full_ids[len(prefix_ids)] == expected


def test_derivation_rejects_too_few_rows(stub_tokenizer):
    rows = {
        a: [(build_formatted_text(f"p{i}", a, i % 2), i % 2) for i in range(5)] for a in ADAPTERS
    }
    with pytest.raises(ScoringError, match="need >="):
        derive_label_ids(stub_tokenizer, rows)


def _reword_answer(row: str, replacement: str) -> str:
    """Swap the ANSWER word for a differently-cased spelling.

    Replaces the LAST occurrence only: the prompt prefix contains "INJECTION or BENIGN" in the
    response directive, and rewriting that would corrupt the prefix instead of the answer.
    """
    from src.templates import END_OF_TURN

    cut = row.rindex(LABEL_INJECTION)
    return row[:cut] + replacement + END_OF_TURN


def test_derivation_rejects_unstable_ids(stub_tokenizer):
    """A label token that varies row to row means the extraction position is wrong.

    Built from real rows whose answer is spelled two ways, rather than a mocked tokenizer —
    "INJECTION" and "Injection" genuinely tokenize to different first tokens.
    """
    assert (
        stub_tokenizer("INJECTION", add_special_tokens=False)["input_ids"][0]
        != stub_tokenizer("Injection", add_special_tokens=False)["input_ids"][0]
    ), "precondition: the two spellings must differ at the first token"

    rows = {}
    for adapter in ADAPTERS:
        built = []
        for i in range(MIN_LABEL_ID_ROWS):
            row = build_formatted_text(f"p{i}", adapter, 1)
            if i % 2:
                row = _reword_answer(row, "Injection")
            built.append((row, 1))
        rows[adapter] = built

    with pytest.raises(ScoringError, match="distinct first-token ids"):
        derive_label_ids(stub_tokenizer, rows)


# --- extraction plumbing --------------------------------------------------------------------


def test_build_batch_requires_left_padding(stub_tokenizer):
    stub_tokenizer.padding_side = "right"
    try:
        with pytest.raises(ScoringError, match="must be 'left'"):
            build_batch(stub_tokenizer, ["a"], [ADAPTERS[0]])
    finally:
        stub_tokenizer.padding_side = "left"


def test_last_position_is_never_padding_under_left_padding(stub_tokenizer):
    """Ragged lengths are the case that exposes a wrong padding side."""
    prompts = ["short", "a considerably longer prompt " * 20, "mid length prompt"]
    batch = build_batch(stub_tokenizer, prompts, [ADAPTERS[0]] * 3)
    assert bool((batch["attention_mask"][:, -1] == 1).all())


def test_position_ids_count_only_real_tokens():
    """Regression for the P1.2 gate failure: max |Δlogit| 0.28, max |Δp| 0.05 on the T4.

    Gemma3 defaults position_ids to arange for every row, so a left-padded row's first real
    token landed at position k instead of 0 and RoPE rotated the whole sequence relative to the
    same prompt scored unpadded.
    """
    import torch

    from src.scoring import position_ids_from_mask

    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])  # left-padded, then full
    pos = position_ids_from_mask(mask)

    assert pos[0][mask[0] == 1].tolist() == [0, 1, 2], "padded row must start counting at 0"
    assert pos[1].tolist() == [0, 1, 2, 3, 4], "unpadded row must be plain arange"
    assert pos[0][0].item() == 1 and pos[0][1].item() == 1, "pads get a dummy, never attended"


def test_build_batch_supplies_position_ids(stub_tokenizer):
    """Omitting them is the bug; they must be in the batch the model receives."""
    prompts = ["short", "a much longer prompt " * 15]
    batch = build_batch(stub_tokenizer, prompts, [ADAPTERS[0]] * 2)

    assert "position_ids" in batch
    for row in range(2):
        real = batch["position_ids"][row][batch["attention_mask"][row] == 1]
        assert real[0].item() == 0
        assert real.tolist() == list(range(len(real)))


def test_ragged_adapter_rows_are_the_reason_padding_exists(stub_tokenizer):
    """The three adapter templates differ in length, so every [3, L] batch is padded.

    This is why the gate exercises the padding path at batch_size=1 rather than only in bulk.
    """
    lengths = {
        a: len(stub_tokenizer(build_prompt("one prompt", a))["input_ids"]) for a in ADAPTERS
    }
    assert len(set(lengths.values())) > 1, f"expected ragged lengths, got {lengths}"


def test_last_position_logits_rejects_right_padding():
    import torch

    logits = torch.zeros(2, 4, 10)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])  # right-padded second row
    with pytest.raises(ScoringError, match="not left-padding"):
        last_position_logits(logits, mask)


def test_two_way_softmax_equals_sigmoid_of_the_difference():
    import torch

    last = torch.zeros(3, 50)
    last[:, 10] = torch.tensor([2.0, 0.0, -1.5])
    last[:, 20] = torch.tensor([0.5, 0.0, 1.0])

    p, d = logits_to_prob(last, inj_id=10, ben_id=20)
    expected = torch.softmax(torch.stack([last[:, 10], last[:, 20]], dim=1), dim=1)[:, 0]

    assert torch.allclose(p, expected, atol=1e-6)
    assert torch.allclose(d, last[:, 10] - last[:, 20], atol=1e-6)
    assert p[1].item() == pytest.approx(0.5)  # equal logits -> p = 0.5


def test_probabilities_are_in_range(stub_tokenizer):
    ids = LabelIds(10, 20, "IN", "BEN", dict.fromkeys(ADAPTERS, 100), True)
    p, _ = score_batched(StubModel(), stub_tokenizer, ["a", "b"], ids, batch_size=2)
    assert p.shape == (2, 3)
    assert ((p >= 0) & (p <= 1)).all()


# --- batched == sequential, on the stub -------------------------------------------------------


@pytest.fixture
def ids():
    return LabelIds(10, 20, "IN", "BEN", dict.fromkeys(ADAPTERS, 100), True)


def test_batched_matches_sequential_on_the_stub(stub_tokenizer, ids):
    """The gate's logic, exercised without a GPU.

    The stub's logits depend on each row's real tokens and assigned adapter, so this passes only
    if batching pairs rows with adapters correctly and padding never reaches the read position.
    """
    import numpy as np

    prompts = [f"prompt number {i} with varying length " * (i % 5 + 1) for i in range(12)]
    p_bat, d_bat = score_batched(StubModel(), stub_tokenizer, prompts, ids, batch_size=4)
    p_seq, d_seq = score_sequential(StubModel(), stub_tokenizer, prompts, ids)

    assert np.max(np.abs(p_bat - p_seq)) < 1e-6
    assert np.max(np.abs(d_bat - d_seq)) < 1e-6


def test_batch_size_one_reproduces_the_c4_shape(stub_tokenizer, ids):
    """C4 specifies a [3, L] batch — one row per adapter for a single prompt."""
    model = StubModel()
    score_batched(model, stub_tokenizer, ["one prompt"], ids, batch_size=1)
    assert len(model.calls) == 1
    assert model.calls[0]["rows"] == len(ADAPTERS)
    assert model.calls[0]["adapter_names"] == list(ADAPTERS)


def test_batching_assigns_one_adapter_per_row_in_order(stub_tokenizer, ids):
    model = StubModel()
    score_batched(model, stub_tokenizer, ["p1", "p2"], ids, batch_size=2)
    assert model.calls[0]["adapter_names"] == list(ADAPTERS) * 2


def test_sequential_path_passes_no_adapter_names(stub_tokenizer, ids):
    """The fallback must drive adapters via set_adapter, not the batched kwarg."""
    model = StubModel()
    score_sequential(model, stub_tokenizer, ["p1", "p2"], ids)
    assert all(call["adapter_names"] is None for call in model.calls)
    assert all(call["rows"] == 1 for call in model.calls)


def test_gate_detects_unsupported_adapter_names(stub_tokenizer, ids):
    """When the stack rejects adapter_names, the gate reports the fallback rather than failing."""
    model = StubModel()
    model.supports_adapter_names = False
    prompts = [f"p{i}" for i in range(GATE_MIN_PROMPTS)]

    report = verify_batched_vs_sequential(model, stub_tokenizer, prompts, ids)

    assert report.batched_supported is False
    assert report.passed is True
    assert "adapter_names" in (report.fallback_reason or "")


def test_gate_passes_when_paths_agree(stub_tokenizer, ids):
    prompts = [f"prompt {i}" for i in range(GATE_MIN_PROMPTS)]
    report = verify_batched_vs_sequential(StubModel(), stub_tokenizer, prompts, ids)

    assert report.batched_supported is True
    assert report.passed is True
    assert report.max_abs_prob_diff < 1e-6
    assert report.n_prompts == GATE_MIN_PROMPTS


def test_gate_fails_when_the_batched_path_changes_decisions(stub_tokenizer, ids):
    """A divergence large enough to flip predictions must surface as passed=False."""

    class DriftingModel(StubModel):
        def __call__(self, *args, adapter_names=None, **kw):
            out = super().__call__(*args, adapter_names=adapter_names, **kw)
            if adapter_names is not None:
                out.logits[:, -1, 10] += 5.0  # batched path only
            return out

    prompts = [f"p{i}" for i in range(GATE_MIN_PROMPTS)]
    report = verify_batched_vs_sequential(DriftingModel(), stub_tokenizer, prompts, ids)

    assert report.batched_supported is True
    assert report.passed is False
    assert report.n_decision_flips > 0
    assert report.max_abs_prob_diff > 1e-3


def test_numerical_gap_that_never_flips_a_decision_now_passes(stub_tokenizer, ids):
    """The re-specified criterion, stated as a test.

    A |Δp| far above the old 1e-3 rule passes IF no prediction changes — that is precisely the
    fp16/NF4 batch-shape situation measured on the T4. The magnitude is still reported, so the
    gap stays visible rather than being hidden by the boolean.
    """
    import numpy as np

    class TinyBiasModel(StubModel):
        def __call__(self, *args, adapter_names=None, **kw):
            out = super().__call__(*args, adapter_names=adapter_names, **kw)
            if adapter_names is not None:
                # Push both label logits together: p moves, sign of (p - 0.5) cannot.
                out.logits[:, -1, 10] *= 1.02
                out.logits[:, -1, 20] *= 1.02
            return out

    prompts = [f"prompt {i}" for i in range(GATE_MIN_PROMPTS)]
    report = verify_batched_vs_sequential(TinyBiasModel(), stub_tokenizer, prompts, ids)

    assert report.n_decision_flips == 0
    assert report.decision_agreement == 1.0
    assert report.passed is True
    assert report.max_abs_prob_diff > GATE_TOLERANCE, "gap must exceed the OLD rule to be a real test"
    assert not np.isnan(report.p95_abs_prob_diff)


def test_diagnosis_attributes_a_mixing_only_divergence(stub_tokenizer, ids):
    """A model that only misbehaves when serving multiple adapters must be named as such."""
    from src.scoring import diagnose_divergence

    class MixingOnlyModel(StubModel):
        def __call__(self, *args, adapter_names=None, **kw):
            out = super().__call__(*args, adapter_names=adapter_names, **kw)
            if adapter_names is not None and len(set(adapter_names)) > 1:
                out.logits[:, -1, 10] += 3.0
            return out

    diag = diagnose_divergence(MixingOnlyModel(), stub_tokenizer, ["a", "b"], ids)

    assert diag.dominant_cause == "adapter_mixing"
    assert diag.adapter_mixing_max_abs_d > 1.0
    assert diag.determinism_max_abs_d == 0.0


def test_diagnosis_reports_a_clean_model_as_deterministic(stub_tokenizer, ids):
    from src.scoring import diagnose_divergence

    diag = diagnose_divergence(StubModel(), stub_tokenizer, ["a", "b"], ids)

    assert diag.determinism_max_abs_d == 0.0
    assert diag.adapter_mixing_max_abs_d == 0.0
    assert diag.batch_vs_single_max_abs_d == 0.0


def test_loader_puts_the_model_in_eval_mode():
    """LoRA dropout in training mode would make every forward nondeterministic."""
    import inspect

    from src import model_loader

    assert "model.eval()" in inspect.getsource(model_loader.load_model_with_adapters)
    assert "training_mode" in model_loader.LoadReport.__dataclass_fields__


def test_gate_reports_decision_agreement_and_percentiles(stub_tokenizer, ids):
    """Distributional evidence must be present even when the max-|Δp| rule passes."""
    prompts = [f"prompt {i}" for i in range(GATE_MIN_PROMPTS)]
    report = verify_batched_vs_sequential(StubModel(), stub_tokenizer, prompts, ids)

    assert report.decision_agreement == 1.0
    assert report.n_decision_flips == 0
    assert report.p95_abs_prob_diff == 0.0
    assert report.flipped_prob_range == []


def test_decision_agreement_counts_real_flips(stub_tokenizer, ids):
    """A large batched-only perturbation must show up as flipped decisions, not just a big max."""

    class FlippingModel(StubModel):
        def __call__(self, *args, adapter_names=None, **kw):
            out = super().__call__(*args, adapter_names=adapter_names, **kw)
            if adapter_names is not None:
                out.logits[:, -1, 10] += 50.0  # batched path -> p ~ 1 everywhere
            return out

    prompts = [f"p{i}" for i in range(GATE_MIN_PROMPTS)]
    report = verify_batched_vs_sequential(FlippingModel(), stub_tokenizer, prompts, ids)

    assert report.n_decision_flips > 0
    assert report.decision_agreement < 1.0
    assert len(report.flipped_prob_range) == 2


def test_gate_refuses_too_few_prompts(stub_tokenizer, ids):
    with pytest.raises(ScoringError, match="gate needs >="):
        verify_batched_vs_sequential(StubModel(), stub_tokenizer, ["a", "b"], ids)


def test_gate_report_persists_label_ids(stub_tokenizer, ids, tmp_path):
    import json

    prompts = [f"p{i}" for i in range(GATE_MIN_PROMPTS)]
    out = tmp_path / "p12_gate_report.json"
    verify_batched_vs_sequential(StubModel(), stub_tokenizer, prompts, ids, out_path=out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["label_ids"]["inj_id"] == ids.inj_id
    assert payload["gate"]["passed"] is True


# --- rule 1 ------------------------------------------------------------------------------------


def test_scorer_builds_no_prompt_strings_of_its_own():
    import inspect

    from src import scoring

    source = inspect.getsource(scoring)
    for fragment in ("<start_of_turn>user", "User Prompt:", "Respond with exactly one word"):
        assert fragment not in source, f"scorer hardcodes template fragment {fragment!r}"
    assert "build_prompt" in source


def test_scorer_never_calls_generate():
    """C5 removes the generate-then-parse mechanism entirely."""
    import inspect

    from src import scoring

    assert ".generate(" not in inspect.getsource(scoring)
