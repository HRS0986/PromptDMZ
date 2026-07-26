"""C4 + C5 — Batched simultaneous forward pass and label-logit extraction.

C4: ONE forward pass (`model(**batch)`, never `generate`) with per-sample adapter assignment
via `adapter_names`, under `torch.inference_mode()`. A sequential `set_adapter` fallback must
also exist for stacks where `adapter_names` is unsupported — identical outputs, no early exit.

C5: read last-non-pad-position logits at the label-word token ids and take a two-way softmax.
Label words are exactly `INJECTION` and `BENIGN`. Derive INJ_ID / BEN_ID empirically from real
training rows (leading space/newline changes the id), assert they differ, log them.
Store `d_i = z_inj - z_ben` alongside `p_i` — C6 calibrates on `d`.

Gates and hazards:
  * P1.2 is the gate for everything downstream: batched vs sequential probabilities must agree
    within fp16 tolerance (max |Δp| < 1e-3) on >=200 prompts. No batched number is trusted until
    this passes.
  * Left-padding is mandatory for causal-LM last-position extraction (or explicit last-non-pad
    indexing). One silent mistake here corrupts every probability.
  * Bulk scoring must be resumable — skip already-scored ids; Kaggle sessions die.

Why the label ids are derived and not hardcoded
-----------------------------------------------
The scoring prompt ends in `\\n` + 8 spaces. SentencePiece may merge trailing whitespace with the
following word, so the first token of `INJECTION` in context need not equal its first token in
isolation. Measured on the deployed tokenizer the two happen to agree (`INJ_ID=1204` `'IN'`,
`BEN_ID=88980` `'BEN'`), but that is a *result*, not a licence to tokenize the bare word: a
tokenizer or template change would silently break the equality and every probability would be
read off the wrong vocabulary slot. `derive_label_ids` therefore always works by prefix-diff
against real stored rows.

Why `p = sigmoid(d)`
--------------------
`softmax([z_inj, z_ben])[0]` is algebraically `sigmoid(z_inj - z_ben)`. Computing it as a
sigmoid of the difference avoids forming the two-element softmax and is the numerically stable
form; `d` is needed by C6 anyway.

Implemented by P1.2 (scorer), P1.3 (legacy agreement), P1.4 (bulk runs). Notebook import
surface: `derive_label_ids`, `verify_batched_vs_sequential`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.templates import (
    ADAPTERS,
    LABEL_BENIGN,
    LABEL_INJECTION,
    build_prompt,
    get_prompt_without_answer,
)

log = logging.getLogger(__name__)

# AC: label ids stable across >=100 sampled real rows per adapter.
MIN_LABEL_ID_ROWS = 100

# AC: the batched-vs-sequential gate runs on >=200 prompts at max |Δp| < 1e-3.
GATE_MIN_PROMPTS = 200
GATE_TOLERANCE = 1e-3


class ScoringError(RuntimeError):
    """Raised when a scoring-path invariant fails. Never recoverable by retrying."""


@dataclass
class LabelIds:
    """Empirically derived label-word token ids, plus the evidence that derived them."""

    inj_id: int
    ben_id: int
    inj_token: str
    ben_token: str
    rows_per_adapter: dict[str, int]
    identical_across_adapters: bool
    per_adapter_ids: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_tuple(self) -> tuple[int, int]:
        return self.inj_id, self.ben_id


@dataclass
class GateReport:
    """P1.2 acceptance evidence."""

    n_prompts: int
    batched_supported: bool
    max_abs_prob_diff: float
    max_abs_logit_diff: float
    mean_abs_prob_diff: float
    tolerance: float
    passed: bool
    inj_id: int
    ben_id: int
    batch_size: int
    fallback_reason: str | None = None


def _next_token_after_prefix(tokenizer, full_text: str, prefix_text: str) -> int:
    """The single token id that follows `prefix_text` inside `full_text`.

    Requires the tokenization of the prefix to be a strict prefix of the full row's. If it is
    not, the template and the tokenizer disagree about a boundary and no downstream number can
    be trusted, so this raises rather than falling back to a heuristic.
    """
    prefix_ids = tokenizer(prefix_text)["input_ids"]
    full_ids = tokenizer(full_text)["input_ids"]

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ScoringError(
            "tokenized prompt is not a prefix of the tokenized full row; the label token "
            "cannot be located by diffing. Template/tokenizer boundary mismatch."
        )
    if len(full_ids) <= len(prefix_ids):
        raise ScoringError("full row has no tokens after the prompt prefix")

    return full_ids[len(prefix_ids)]


def load_label_id_rows(
    token: str | None = None,
    n_rows: int = 200,
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
    """Fetch real stored rows for label-id derivation, one sample per adapter.

    TRAIN deliberately: UNIFIED-VAL is reserved for fitting and UNIFIED-TEST is sealed until
    Phase 7. Templating is identical across splits, so train is the correct source for a
    tokenizer-level fact and costs nothing downstream.

    Lives here rather than in the notebook because notebooks carry no logic — and because a
    direct `load_dataset` in a notebook is exactly the shape of an accidental test-split read.
    """
    from datasets import load_dataset

    from src.templates import DATASET_REPOS

    rows: dict[str, list[tuple[str, int]]] = {}
    for adapter, repo in DATASET_REPOS.items():
        ds = load_dataset(repo, split="train", token=token)
        sample = ds.shuffle(seed=seed).select(range(min(n_rows, len(ds))))
        rows[adapter] = list(zip(sample["formatted_text"], sample["label"]))
        log.info("label-id rows: %s <- %s[train] (%d)", adapter, repo, len(rows[adapter]))
    return rows


def derive_label_ids(
    tokenizer,
    rows_by_adapter: dict[str, list[tuple[str, int]]],
    min_rows: int = MIN_LABEL_ID_ROWS,
) -> LabelIds:
    """Derive INJ_ID / BEN_ID by prefix-diffing real stored rows.

    `rows_by_adapter` maps adapter name -> list of `(formatted_text, label)` from the Hub.
    Every row of a given label must yield the same id, within and across adapters; anything
    else raises, because a per-row-varying label token means the extraction position is wrong.
    """
    per_adapter: dict[str, dict[str, int]] = {}
    counts: dict[str, int] = {}
    inj_ids: set[int] = set()
    ben_ids: set[int] = set()

    for adapter, rows in rows_by_adapter.items():
        if len(rows) < min_rows:
            raise ScoringError(
                f"{adapter}: {len(rows)} rows supplied, need >= {min_rows} to certify stability"
            )

        seen: dict[int, set[int]] = {0: set(), 1: set()}
        for formatted_text, label in rows:
            prefix = get_prompt_without_answer(formatted_text)
            token_id = _next_token_after_prefix(tokenizer, formatted_text.removeprefix("<bos>"), prefix)
            seen[int(label)].add(token_id)

        for label, ids in seen.items():
            word = LABEL_INJECTION if label == 1 else LABEL_BENIGN
            if len(ids) != 1:
                raise ScoringError(
                    f"{adapter}: label {word} produced {len(ids)} distinct first-token ids "
                    f"({sorted(ids)}); the extraction position is not stable"
                )

        adapter_inj = seen[1].pop()
        adapter_ben = seen[0].pop()
        if adapter_inj == adapter_ben:
            raise ScoringError(
                f"{adapter}: INJ_ID == BEN_ID == {adapter_inj}; the two classes are "
                "indistinguishable at the label position"
            )

        per_adapter[adapter] = {"inj_id": adapter_inj, "ben_id": adapter_ben}
        counts[adapter] = len(rows)
        inj_ids.add(adapter_inj)
        ben_ids.add(adapter_ben)

    identical = len(inj_ids) == 1 and len(ben_ids) == 1
    if not identical:
        raise ScoringError(
            f"label ids differ across adapters (INJ {sorted(inj_ids)}, BEN {sorted(ben_ids)}). "
            "The context preceding the label is byte-identical, so this should be impossible — "
            "stop and investigate rather than proceeding."
        )

    inj_id, ben_id = inj_ids.pop(), ben_ids.pop()
    ids = LabelIds(
        inj_id=inj_id,
        ben_id=ben_id,
        inj_token=tokenizer.decode([inj_id]),
        ben_token=tokenizer.decode([ben_id]),
        rows_per_adapter=counts,
        identical_across_adapters=identical,
        per_adapter_ids=per_adapter,
    )
    log.info(
        "derived label ids: INJ_ID=%d (%r), BEN_ID=%d (%r) over %s rows/adapter",
        ids.inj_id,
        ids.inj_token,
        ids.ben_id,
        ids.ben_token,
        counts,
    )
    return ids


def position_ids_from_mask(attention_mask):
    """Positions that count only real tokens, so padding cannot shift RoPE.

    `Gemma3TextModel.forward` defaults `position_ids` to `arange(0, L)` for EVERY row when the
    caller omits them. Under left padding that gives a row with `k` pads its first real token at
    position `k` rather than 0, rotating the whole sequence in RoPE relative to the same prompt
    scored unpadded. That is a silent, systematic batched-vs-sequential divergence proportional
    to the padding — it was measured at max |Δlogit| = 0.28, max |Δp| = 0.05 on the P1.2 gate,
    fifty times the fp16 tolerance.

    Real tokens therefore get 0,1,2,… by cumulative sum; pads get a dummy 1 (never attended).
    With no padding this reduces to `arange`, so the sequential path is numerically unchanged
    while both paths now compute positions the same explicit way.
    """
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 1)


def build_batch(tokenizer, prompts: list[str], adapters: list[str]):
    """Tokenize one batch of templated prompts. Requires LEFT padding.

    Returns a complete model input including explicit `position_ids` — see
    `position_ids_from_mask` for why omitting them corrupts batched scoring.

    Prompt strings come from `templates.build_prompt` — this module never assembles template
    text itself (non-negotiable rule 1).
    """
    if tokenizer.padding_side != "left":
        raise ScoringError(
            f"tokenizer.padding_side is {tokenizer.padding_side!r}, must be 'left'. With right "
            "padding the last position is a pad token and every probability is read off the "
            "wrong slot."
        )
    if len(prompts) != len(adapters):
        raise ScoringError("prompts and adapters must be the same length (one adapter per row)")

    texts = [build_prompt(p, a) for p, a in zip(prompts, adapters)]
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    batch["position_ids"] = position_ids_from_mask(batch["attention_mask"])
    return batch


def last_position_logits(logits, attention_mask):
    """Logits at each row's final real token.

    With left padding every row's last real token is the final column, so this is `[:, -1]`.
    The attention mask is checked rather than assumed: if any row's final column is padding,
    the padding side is wrong and the extraction would be silently meaningless.
    """
    if not bool((attention_mask[:, -1] == 1).all()):
        raise ScoringError(
            "final position is padding for at least one row — tokenizer is not left-padding"
        )
    return logits[:, -1, :]


def logits_to_prob(last_logits, inj_id: int, ben_id: int):
    """Two-way softmax over the label logits. Returns `(p, d)` as float32 tensors.

    `p = softmax([z_inj, z_ben])[0] = sigmoid(z_inj - z_ben)`. Cast to float32 before the
    subtraction: the forward runs in fp16, where differences of large logits lose precision.
    """
    import torch

    z_inj = last_logits[:, inj_id].float()
    z_ben = last_logits[:, ben_id].float()
    d = z_inj - z_ben
    return torch.sigmoid(d), d


def _forward(model, batch, adapter_names: list[str] | None):
    import torch

    with torch.inference_mode():
        if adapter_names is None:
            return model(**batch).logits
        return model(**batch, adapter_names=adapter_names).logits


def score_batched(
    model,
    tokenizer,
    prompts: list[str],
    label_ids: LabelIds,
    batch_size: int = 1,
    adapters: tuple[str, ...] = ADAPTERS,
):
    """Score every prompt with all three adapters in batched forward passes.

    `batch_size` counts PROMPTS; each contributes one row per adapter, so a forward pass sees
    `batch_size * len(adapters)` rows. `batch_size=1` reproduces C4's `[3, L]` shape exactly.

    Returns `(p, d)` arrays of shape `[n_prompts, n_adapters]`, adapter order as given.
    """
    import torch

    probs, diffs = [], []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        row_prompts = [p for p in chunk for _ in adapters]
        row_adapters = [a for _ in chunk for a in adapters]

        batch = build_batch(tokenizer, row_prompts, row_adapters).to(model.device)
        logits = _forward(model, batch, adapter_names=row_adapters)
        last = last_position_logits(logits, batch["attention_mask"])
        p, d = logits_to_prob(last, label_ids.inj_id, label_ids.ben_id)

        probs.append(p.reshape(len(chunk), len(adapters)))
        diffs.append(d.reshape(len(chunk), len(adapters)))

    return torch.cat(probs).cpu().numpy(), torch.cat(diffs).cpu().numpy()


def score_sequential(
    model,
    tokenizer,
    prompts: list[str],
    label_ids: LabelIds,
    adapters: tuple[str, ...] = ADAPTERS,
):
    """Reference path: `set_adapter` + one single-row forward per (prompt, adapter).

    No early exit — every prompt is scored by every adapter, so the outputs are directly
    comparable to `score_batched`. This is the fallback when `adapter_names` is unsupported,
    and the ground truth the P1.2 gate compares against.
    """
    import numpy as np

    p_out = np.zeros((len(prompts), len(adapters)), dtype=np.float32)
    d_out = np.zeros((len(prompts), len(adapters)), dtype=np.float32)

    for col, adapter in enumerate(adapters):
        model.set_adapter(adapter)
        for row, prompt in enumerate(prompts):
            batch = build_batch(tokenizer, [prompt], [adapter]).to(model.device)
            logits = _forward(model, batch, adapter_names=None)
            last = last_position_logits(logits, batch["attention_mask"])
            p, d = logits_to_prob(last, label_ids.inj_id, label_ids.ben_id)
            p_out[row, col] = float(p[0])
            d_out[row, col] = float(d[0])

    return p_out, d_out


def batched_adapter_names_supported(model, tokenizer, label_ids: LabelIds) -> tuple[bool, str | None]:
    """Probe whether the runtime stack accepts per-sample `adapter_names`.

    Deliberately broad: PEFT versions signal an unsupported `adapter_names` differently
    (TypeError on the kwarg, ValueError deeper in, or an assertion inside the layer). Any
    failure means the same thing operationally — use the sequential fallback. A `ScoringError`
    is re-raised, because that indicates a broken invariant (padding side, template boundary),
    not an unsupported feature.
    """
    try:
        score_batched(model, tokenizer, ["probe"], label_ids, batch_size=1)
    except ScoringError:
        raise
    except Exception as exc:  # noqa: BLE001 - any other failure means "use the fallback"
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def verify_batched_vs_sequential(
    model,
    tokenizer,
    prompts: list[str],
    label_ids: LabelIds,
    batch_size: int = 1,
    tolerance: float = GATE_TOLERANCE,
    out_path: Path | str | None = None,
) -> GateReport:
    """THE P1.2 GATE. Nothing downstream may be trusted until this passes.

    Runs both paths over the same prompts and compares probabilities elementwise. If
    `adapter_names` is unsupported the batched path is reported unsupported and the comparison
    is skipped — the fallback is then the only path, so there is nothing to disagree with.
    """
    import numpy as np

    if len(prompts) < GATE_MIN_PROMPTS:
        raise ScoringError(
            f"gate needs >= {GATE_MIN_PROMPTS} prompts, got {len(prompts)}"
        )

    supported, reason = batched_adapter_names_supported(model, tokenizer, label_ids)
    p_seq, d_seq = score_sequential(model, tokenizer, prompts, label_ids)

    if supported:
        p_bat, d_bat = score_batched(model, tokenizer, prompts, label_ids, batch_size=batch_size)
        max_p = float(np.max(np.abs(p_bat - p_seq)))
        mean_p = float(np.mean(np.abs(p_bat - p_seq)))
        max_d = float(np.max(np.abs(d_bat - d_seq)))
    else:
        log.warning("adapter_names unsupported (%s); sequential fallback is the only path", reason)
        max_p = mean_p = max_d = 0.0

    report = GateReport(
        n_prompts=len(prompts),
        batched_supported=supported,
        max_abs_prob_diff=max_p,
        max_abs_logit_diff=max_d,
        mean_abs_prob_diff=mean_p,
        tolerance=tolerance,
        passed=(not supported) or (max_p < tolerance),
        inj_id=label_ids.inj_id,
        ben_id=label_ids.ben_id,
        batch_size=batch_size,
        fallback_reason=reason,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"gate": asdict(report), "label_ids": asdict(label_ids)}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        log.info("wrote %s", out_path)

    log.info(
        "P1.2 gate: supported=%s max|dp|=%.3e (tol %.0e) -> %s",
        supported,
        max_p,
        tolerance,
        "PASS" if report.passed else "FAIL",
    )
    return report
