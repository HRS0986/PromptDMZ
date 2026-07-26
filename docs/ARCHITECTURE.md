# ARCHITECTURE.md — Prompt Injection Detection System (Build Target)

**Project:** FYP — Category-specialised QLoRA adapters on a shared quantized SLM backbone with calibrated learned fusion and conformal FPR control.
**Hardware target:** Single NVIDIA T4 (~15–16 GB VRAM), Colab/Kaggle. All GPU code must be memory-optimised. All decision-layer code (calibration, fusion, conformal) runs on CPU.
**Output:** Binary — `INJECTION` or `BENIGN` — plus per-category attribution scores.
**Excluded from this build:** spelling-correction channel (kept as a future extension; the fusion feature vector must remain extensible — see §C7), knowledge-base retrieval channel, escalate/abstain output band.

---

## 0. Existing assets — REUSE, do not rebuild

| Asset | Location | Reuse |
|---|---|---|
| 3 trained QLoRA adapters (r=16, α=16 — confirmed identical to the baseline; targets q/k/v/o/gate/up/down_proj) | HF Hub, private: `hirushafernando/fyp-gemma3-1b-slm-a-qlora` (role/instruction violation), `…-slm-b-qlora` (privilege escalation), `…-slm-c-qlora` (obfuscation/evasion) | Load as-is in C1. **Frozen for the entire core build** (Phases 0–7). These run SIMULTANEOUSLY at inference (batched, one per sample) — they ARE the system. |
| **Baseline "merged"/generalist adapter — ALREADY TRAINED** (r=16, α=16, same recipe as specialists; `FT_Merged_Adapter.ipynb`) | HF Hub, private: `hirushafernando/fyp-gemma3-1b-slm-merged-qlora` | Load as-is for the baseline comparison row (config (a) in §4). **NOT part of the running system** — it is the experimental control ("do 3 specialists beat 1 merged adapter?"). Never runs alongside the specialists. **P6.1 is therefore DONE — no retraining needed.** Quick-val already ~0.991 acc on 1000 rows (not the final metric — rescored properly in Phase 7). |
| 3 per-category datasets + the merged dataset, all with existing 80/10/10 stratified splits, columns `formatted_text` + `label` (0=BENIGN, 1=INJECTION); **no raw `text` column** | HF Hub, private: `hirushafernando/fyp-slm-a`, `hirushafernando/fyp-slm-b`, `hirushafernando/fyp-slm-c`, and `hirushafernando/fyp-slm-merged` (108,798 / 21,489 / 21,492 rows) | Splits reused verbatim (§3). Never re-split. Merged dataset already built (EDA/NB5 Step A). |
| `EDA.ipynb` | repo | Reused for Phase 8 augmentation only (the merged dataset it builds already exists). Its standardisation/dedup/formatting functions are the reference implementation. |
| `LoRA-Fine-Tuning.ipynb` / `FT_Merged_Adapter.ipynb` | repo | The recipe reference (both use r=16, α=16, confirmed identical). **The baseline is already trained via `FT_Merged_Adapter.ipynb`** — reuse these only for Phase 8 retrains if you do them. Note: packing auto-disabled by Unsloth, `adamw_8bit`, cosine schedule, 2 epochs. |
| `Final_Inference_Pipeline.ipynb` | repo | Superseded as the pipeline, but KEPT as the legacy baseline: source for `model_loader.py` loading code; the generation+parse comparator in P1.3; the sequential-early-exit latency row in §4. |
| Training checkpoint id | `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` | C1 must use this (see fp16 discrepancy note in C1). |

Facts confirmed from the notebooks that constrain the build:
- Label words are exactly **`INJECTION`** and **`BENIGN`** ("Respond with exactly one word: INJECTION or BENIGN") → C5 label-token derivation targets these two words.
- **Template variation** was deliberately applied during dataset creation (anti-overfitting). Consequence: there is NO single canonical template; see revised C2.
- `Evaluation.ipynb` loaded the backbone in **fp16** while training used the **4-bit Unsloth checkpoint** — a train/eval discrepancy. This build standardises on 4-bit everywhere (deployability claims depend on it); note the discrepancy in the thesis when comparing against old metrics.

## 1. System overview

```
User prompt (raw text)
   ├──> [C3] Tokenizer statistics (CPU, on RAW text)  ──────────────┐
   └──> [C2] Per-adapter templating ×3 → batch tokenize             │
            └──> [C4] ONE batched forward pass                      │
                     (3 resident LoRA adapters, per-sample assign)  │
                 └──> [C5] Label-logit extraction → p1, p2, p3      │
                      └──> [C6] Temperature scaling → p̂1, p̂2, p̂3    │
                           └──> [C7] Learned fusion  <──────────────┘
                                (feature vector: 3 probs + 4 stats)
                                └──> fused score S(x)
                                     └──> [C8] Conformal threshold τ̂
                                          └──> [C9] INJECTION / BENIGN
                                               + category attribution
```

Design principle: the three adapters do *semantic* work (language understanding); everything downstream is a small, cheap, auditable *statistical decision layer*. The execution layer (C1–C5) is fixed once built; the decision layer (C6–C8) is refit in minutes whenever any adapter changes.

---

## 2. Component specifications

### C1 — Model + adapter loading (once per session)

- **Input:** backbone checkpoint `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (the training checkpoint — NOT plain `google/gemma-3-1b-it` in fp16, which is what the legacy Evaluation notebook used; that discrepancy is documented in §0); the 3 Hub adapters `fyp-gemma3-1b-slm-a/b/c-qlora` (HF_TOKEN required, repos are private).
- **Process:** Load backbone 4-bit NF4 (as in training). Attach all three Hub adapters to the same PEFT model as named adapters (`role_violation`, `privilege_escalation`, `obfuscation_evasion` ← a/b/c). All adapters stay memory-resident for the whole session. No merging. Seed the loading code from `Final_Inference_Pipeline.ipynb`.
- **Output:** One `PeftModel` handle + tokenizer.
- **Justification:** ~1B params at 4-bit ≈ <1 GB weights; three r=16 adapters add only tens of MB. Resident adapters make "swap" a pointer flip (`set_adapter`) and enable per-sample batched assignment. This is the deployability core of the thesis.
- **Constraints:** Declare `peft`, `transformers`, `bitsandbytes`, `unsloth`, `trl`, `accelerate` in `pyproject.toml` and lock with `uv lock` (produces `uv.lock`) — this is the reproducible local/serve environment. Per-sample adapter assignment (`adapter_names`) requires a recent PEFT and must be verified against the runtime stack (see TASKS P1.2). NOTE: on Kaggle/Colab the platform's pre-provisioned torch+CUDA is kept; the notebooks `pip install` the behaviour-affecting libraries on top rather than letting uv rebuild the GPU stack (see §6 / TASKS P0.1).

### C2 — Canonical templating + batch tokenization

- **Input:** raw user prompt string.
- **Process:** For each adapter, wrap the prompt in that adapter's instruction template using Gemma chat format, ending exactly at the model turn (`<start_of_turn>model\n`) **with no answer text**. Tokenize the three strings as one padded batch (left-padding for causal LM; `max_length=2048`, truncation documented).
- **Output:** `input_ids [3, L]`, `attention_mask [3, L]`, plus the raw prompt passed through untouched.
- **Justification:** Templates prime the model so its next-token distribution concentrates on the label vocabulary the adapters were fine-tuned to emit.
- **CRITICAL — template variation reality:** The datasets were built with deliberate prompt-template variation (anti-overfitting, per EDA.ipynb), so there is NO single canonical template. Two scoring modes, both in `src/templates.py`:
  - **Dataset-row scoring (all split evaluations):** use each row's own stored `formatted_text`, stripped of the answer via `get_prompt_without_answer()` (split on `<start_of_turn>model`). This is the most faithful evaluation input — it is exactly the distribution the adapters were trained on. Do NOT regenerate templates for stored rows.
  - **Live inference (`pipeline.detect(prompt)`):** freeze ONE representative template variant per adapter in `templates.py` (choose the highest-frequency variant found in the training data; document the choice). Cross-adapter scoring of a raw prompt uses each adapter's own frozen variant.
  - Raw-prompt recovery: verify the datasets retain a raw `text` column; if only `formatted_text` exists, implement + unit-test an extractor (the user prompt sits between the instruction preamble and the "Respond with exactly one word" suffix). Needed for C3 (stats on raw text) and for unified cross-adapter scoring (§3).
- **Legacy divergence note:** the THREE inconsistent `build_prompt` implementations across `LoRA-Fine-Tuning.ipynb`, `Evaluation.ipynb`, and `Final_Inference_Pipeline.ipynb` (whitespace/punctuation/newline diffs) are exactly why no notebook may build prompts anymore — `src/templates.py` is the only prompt builder.
- **CRITICAL — no label leakage:** The template must end at `<start_of_turn>model\n`. The previous evaluation bug re-wrapped already-answered training examples; the fix pattern is a `get_prompt_without_answer()` that splits on `<start_of_turn>model`. Never feed a string containing the gold label.

### C3 — Tokenizer statistics (CPU side-channel)

- **Input:** the **raw** user prompt (pre-template — template tokens must not pollute the statistics).
- **Process:** Tokenize raw text once with the same tokenizer; compute 4 features:
  1. `fertility` = num_tokens / max(1, num_chars)
  2. `byte_fallback_rate` = fraction of tokens that are byte-fallback tokens (Gemma SentencePiece `<0xNN>` pieces)
  3. `rare_token_fraction` = fraction of tokens with corpus frequency below a threshold; implement as: fit token-frequency table on the benign portion of the fusion-train split, rare = not in top-K (K configurable, default 20 000)
  4. `token_length_entropy` = Shannon entropy of the distribution of token string-lengths in the prompt
- **Output:** `stats ∈ R^4` (standardised later by a scaler fitted on fusion-train only).
- **Justification:** Obfuscated/manipulated text (encodings, homoglyphs, TokenBreak-style perturbations) distorts segmentation shape even when semantics are hidden. Near-zero cost (CPU, runs concurrently with the GPU pass). Gives fusion a signal family orthogonal to the adapters' semantic evidence. Also the future gate for the correction channel.

### C4 — Batched simultaneous forward pass

- **Input:** the `[3, L]` batch from C2.
- **Process:** One forward pass (`model(**batch)` — **no** `generate`) with per-sample adapter assignment: `adapter_names=["role_violation", "privilege_escalation", "obfuscation_evasion"]`, one per batch row. `torch.inference_mode()`, fp16 autocast.
- **Output:** logits `[3, L, V]`; only the last non-pad position per row is needed downstream.
- **Justification:** Replaces up to three sequential generate calls with one pass; constant latency for all traffic (the sequential OR design paid worst-case 3× on benign traffic, the common case). Backbone dequantization cost is amortised across the three rows.
- **Fallback (must be implemented):** if `adapter_names` is unsupported/incorrect on the pinned stack, run three sequential single-row forward passes with `set_adapter` — identical outputs, no early exit. The architecture is unchanged; only latency differs. Verification requirement: batched probabilities must match sequential probabilities within fp16 tolerance (P1.2).

### C5 — Label-logit extraction

- **Input:** last-position logits `[3, V]`.
- **Process:** For each row, read logits at the token ids of the adapters' label words and take a two-way softmax:
  - Label words are confirmed from the training data: **`INJECTION`** and **`BENIGN`**. Determine `INJ_ID` and `BEN_ID` = the **first** tokenizer token of each label word exactly as it appears after `<start_of_turn>model\n` in the stored `formatted_text` (leading-space/newline variants change the id — derive empirically from actual training rows, assert uniqueness, log the ids).
  - `p_i = softmax([z_inj, z_ben])[0]` per adapter; equivalently store `d_i = z_inj − z_ben` (needed by C6).
- **Output:** `p_raw ∈ [0,1]^3` (+ `d ∈ R^3`).
- **Justification:** Removes the old generate-then-string-parse mechanism entirely: no parse failures (the old "unknown → INJECTION" fallback disappears), ~4–8× less decode compute, and yields a *proper* class probability — the old "confidence" was the emitted token's probability under the full 262k-vocab softmax, which is not a class probability and is incomparable across adapters.
- **Sanity requirement:** On a labelled sample, argmax of `p_raw` must agree with the legacy generation-parse pipeline ≥ ~99% (disagreements should be the legacy parser's failures — inspect and log them; this becomes an ablation datapoint).

### C6 — Per-adapter temperature calibration

- **Input:** logit differences `d_i` on the **T-split** (see §3); at inference, the live `d_i`.
- **Process:** For each adapter i, fit scalar `T_i > 0` minimising BCE of `σ(d_i / T_i)` against labels on the T-split (scipy `minimize_scalar`, bounds e.g. [0.05, 20]). At inference: `p̂_i = σ(d_i / T_i)`.
- **Output:** calibrated `p̂ ∈ [0,1]^3`; artefact: `temperatures.json`.
- **Justification:** Fine-tuned classifiers are systematically miscalibrated, each by a different amount; without per-adapter correction, "0.8" means different things from different adapters and the fusion layer wastes capacity undoing the distortion. Temperature scaling is monotone ⇒ provably cannot change any adapter's ranking/accuracy (state this in the thesis). Report ECE + reliability diagrams before/after, per adapter (P2.2).

### C7 — Learned fusion

- **Input:** feature vector from a **single assembly function** (extension seam — do not hardcode widths anywhere else):
  ```python
  def build_feature_vector(p_hat, stats_scaled) -> np.ndarray:
      # v1 layout: [p̂1, p̂2, p̂3, s1, s2, s3, s4]  → shape (7,)
      # future extensions append slots here ONLY (e.g. corrected-channel probs + flag)
  ```
- **Process:** Train and compare three fusers on the **F-split**:
  1. **Noisy-OR:** `S = 1 − Π_i (1 − q_i · p̂_i)` with learned reliabilities `q_i ∈ [0,1]` (fit by BCE, sigmoid-parameterised q; probability-features only). Collapses to the legacy OR-gate at `q_i=1` + hard thresholds — the principled generalisation of the baseline.
  2. **Logistic regression** (sklearn, all 7 features, standardised) — interpretable weights; expected to down-weight the noisier privilege-escalation adapter (report the weights).
  3. **MLP** (one hidden layer, 8–16 units, sklearn `MLPClassifier`, early stopping) — captures interactions (e.g. two moderate scores jointly suspicious).
- **Output:** fused score `S(x) ∈ [0,1]`; artefacts: `fusion_<variant>.pkl`, `scaler.pkl`.
- **Justification:** The OR-gate is a zero-parameter fusion assuming equal adapter reliability — empirically false. Learned fusion strictly generalises it (OR is representable in each family ⇒ learned fusion cannot be expressively worse). Heterogeneous features (semantic + surface-statistical) are the integration-novelty claim. All fusers are ≤ a few thousand params, CPU, seconds to train.
- **Selection rule:** pick the variant with best validation F1 at matched FPR on F-split internal CV; carry ALL variants into the final test table (they are the ablation).

### C8 — Conformal threshold (certified FPR)

- **Input:** fused scores of **benign** examples in the **C-split** (never used for any fitting); target level α (default 0.05).
- **Process:** Split-conformal quantile with finite-sample correction:
  ```python
  scores = sorted(S(x) for benign x in C_split)      # n values
  k = ceil((1 - alpha) * (n + 1))                    # NOT (1-alpha)*n
  tau = scores[k - 1]                                # k-th smallest
  ```
- **Output:** `tau` (+ metadata: n, α); artefact `conformal.json`.
- **Justification:** Converts an arbitrary 0.5 cut into a threshold with a distribution-free guarantee: for exchangeable benign inputs, `P(S(x) > τ̂) ≤ α` (up to O(1/n)). Upgrades the claim from "good empirical FPR" to "certified FPR bound" — qualitatively different. Costs a sort.
- **Constraints:** need n ≥ 100–200 benign in C-split for a practically tight bound (bare minimum 1/α). Guarantee is marginal, not per-instance — one honest sentence in the thesis. Report certified α vs empirically observed test FPR side by side (they should be consistent; that plot is the payoff).

### C9 — Decision + attribution

- **Input:** `S(x)`, `τ̂`, calibrated `p̂`.
- **Process:** `INJECTION` if `S(x) > τ̂` else `BENIGN`. Attribution = the per-category `p̂_i` (argmax + full vector), reported alongside every INJECTION verdict.
- **Output:** `{"verdict": "INJECTION"|"BENIGN", "score": S, "threshold": tau, "category_scores": {...}}`
- **Justification:** Binary output matches an autonomous inline guardrail (no human to escalate to). Attribution is preserved because fusion consumes, not destroys, the per-adapter probabilities — interpretability survives the upgrade.

### C10 — FastAPI demonstration server

- **Input:** HTTP `POST /detect` with `{"prompt": "<text>"}`.
- **Process:** At startup (once): load backbone + 3 resident adapters (C1) and all fitted decision-layer artefacts (temperatures, fusion model, scaler, τ̂). Per request: run the full `pipeline.detect()` path — tokenizer stats + batched forward + label-logit extraction + calibration + fusion + conformal decision.
- **Output:** `{"verdict": "INJECTION"|"BENIGN", "score": float, "threshold": float, "category_scores": {"role_violation": float, "privilege_escalation": float, "obfuscation_evasion": float}, "latency_ms": float}`.
- **Endpoints:** `POST /detect` (single), `POST /detect_batch` (list), `GET /health` (model + artefacts loaded), `GET /` (minimal HTML form for live demo).
- **Justification:** Demonstrates the deployability thesis concretely — one resident backbone, sub-second inline verdicts, category attribution surfaced. The server is a thin wrapper over `pipeline.detect()`; it contains no detection logic of its own (keeps the demo faithful to the evaluated system).
- **Constraints:** model loads once at startup, never per request (state on `app.state`); `torch.inference_mode()`; single-worker (one model in VRAM — do not fork workers on a single GPU); provide a CPU-fallback flag so the API is demonstrable on a laptop without a GPU (slower, same outputs). Not a hardened production service — CORS/local use, documented as a demo.

---

## 3. Data-split protocol (CRITICAL — violations invalidate results)

Source material = the three Hub datasets' **existing** splits (80/10/10, already stratified). Never re-split train/val/test boundaries.

**Unified sets (pipeline-level):** each adapter has its own dataset, but the pipeline scores every prompt with all three adapters. Build `UNIFIED-VAL` = union of the three datasets' validation splits, and `UNIFIED-TEST` = union of the three test splits; tag every row with its source category (A/B/C benign or attack family); deduplicate exact text across the union **and verify zero overlap between UNIFIED-VAL and UNIFIED-TEST after union** (benign pools were shared across datasets, so cross-dataset val/test collisions are possible — drop colliders from VAL, log counts).

Partition UNIFIED-VAL randomly (stratified by label × source category) into three disjoint parts; UNIFIED-TEST is untouched until the very end and used **exactly once**:

| Split | Name | Used by | Approx. share of UNIFIED-VAL |
|---|---|---|---|
| T-split | temperature fit | C6 | 30% |
| F-split | fusion training (+ scaler + rare-token table) | C7, C3 | 40% |
| C-split | conformal calibration | C8 | 30% (≥100–200 benign) |
| UNIFIED-TEST | final evaluation, single pass | §4 | the three Hub test splits, unioned + deduped |

Rules:
1. Fitting order is rigid: C6 → C7 → C8. Downstream artefacts depend on upstream ones; if any adapter is ever retrained, refit all three (minutes).
2. Nothing in T/F/C may overlap TEST. Re-verify by hashing prompts (exact-dup check across all four sets; log collisions).
3. All scoring uses the canonical template (C2) — the label-leakage class of bug is structurally prevented by `src/templates.py` being the only prompt builder.
4. Persist every artefact (scores parquet, temperatures, fusion models, scaler, tau) to Drive after each stage — Colab/Kaggle sessions die; nothing may depend on a live session.

---

## 4. Evaluation matrix (final test-split run)

Configurations (rows):
- (a) **Merged/generalist baseline adapter** (`fyp-gemma3-1b-slm-merged-qlora`, ALREADY TRAINED, same recipe: r=16, α=16) — the specialist-vs-generalist control for the thesis hypothesis. Load and score; do not retrain.
- (b) 3 specialists + hard OR-gate on raw `p_i > 0.5` — the legacy decision logic, re-implemented on the new scoring path.
- (c) 3 specialists + learned fusion, **uncalibrated** (ablates calibration).
- (d) 3 specialists + **calibrated** learned fusion (each variant: noisy-OR / LR / MLP), probs-only.
- (e) (d) + tokenizer stats (full feature vector) — ablates the stats channel.
- (f) best of (d/e) + conformal threshold — the headline system.

Metrics (columns): Accuracy, Precision, Recall, F1, FPR, AUROC, ECE (pre/post calibration, per adapter), per-category recall, cross-category leakage matrix (which adapter fires on which attack family), parse-failure rate of legacy pipeline vs 0 for new (motivates C5), latency ms/prompt (batched vs sequential; report both), peak VRAM (`torch.cuda.max_memory_allocated`), adapter + decision-layer parameter counts.

---

## 5. Known pitfalls (encode as tests, not tribal knowledge)

1. **Template divergence** — three inconsistent `build_prompt`s exist in the repo today. Only `src/templates.py` may build prompts; add a unit test freezing the exact template string (golden test).
2. **Label leakage** — never score a string containing the answer; golden test: templated output must not contain the label tokens after `<start_of_turn>model`.
3. **Label token ids** — leading space/newline changes the first-token id; derive empirically, assert the two ids differ, log them.
4. **`adapter_names` support** — verify batched == sequential numerically before trusting any batched numbers (P1.2 acceptance test). Fall back to sequential-no-early-exit if needed.
5. **Padding side** — causal LM last-position extraction requires left-padding (or explicit last-non-pad indexing); one silent mistake here corrupts every probability.
6. **Stats on raw text only** — template/marking tokens must never enter C3.
7. **Identical val/test metrics** — previous results files showed identical numbers for val and test (suspected duplicated run). The new harness must label every output file with split name + prompt-set hash to make this class of error impossible.
8. **Session death** — checkpoint scores/artefacts to Drive after every stage; scoring runs must be resumable (skip already-scored ids).
9. **Template variation** — dataset rows carry varied templates; scoring stored rows must use their own `formatted_text` (answer-stripped), never a regenerated template. Live inference uses the frozen variant. Mixing these modes silently shifts the input distribution.
10. **Backbone dtype mismatch** — training used the 4-bit Unsloth checkpoint; the legacy Evaluation notebook loaded fp16. All new scoring uses 4-bit. Old saved metrics are not directly comparable; do not mix them into new tables.
11. **Cross-dataset benign overlap** — benign pools were drawn from shared sources across the three datasets; the UNIFIED-VAL/UNIFIED-TEST dedup + overlap check (§3) is mandatory, not optional.

---

## 6. Module layout

```
src/
  templates.py        # canonical per-adapter build_prompt + get_prompt_without_answer (+ golden tests)
  model_loader.py     # C1: backbone 4-bit + 3 resident adapters; version pinning
  scoring.py          # C4+C5: batched forward, label-logit extraction, sequential fallback,
                      #        batched-vs-sequential verifier, resumable bulk scorer → parquet
  tokenizer_stats.py  # C3: 4 features + rare-token table fitting
  calibration.py      # C6: temperature fit, ECE, reliability-diagram data
  fusion.py           # C7: build_feature_vector (THE extension seam), noisy-OR / LR / MLP, scaler
  conformal.py        # C8: quantile threshold + guarantee metadata
  pipeline.py         # C9: end-to-end detect(prompt) using persisted artefacts
  eval/
    splits.py         # T/F/C partition + overlap hashing + split manifests
    metrics.py        # all §4 metrics incl. leakage matrix, latency/VRAM harness
    run_eval.py       # the single-pass test evaluation producing the §4 table
  serve/
    app.py            # FastAPI demo: load model+adapters+artefacts once at startup, POST /detect
    schemas.py        # DetectRequest / DetectResponse pydantic models
notebooks/            # thin Kaggle entry points (01 scoring, 02 baseline, 03 test) — call src/ only
artifacts/            # temperatures.json, fusion_*.pkl, scaler.pkl, conformal.json, scores/*.parquet
pyproject.toml        # dependency declarations (managed by uv)
uv.lock               # locked resolution for the reproducible local/serve environment
```

Notebooks contain no logic — all logic lives in `src/` so it is testable and reusable across sessions.

**Package management (uv), two environments:**
- **Local / FastAPI serve (reproducible):** `uv` owns it. `uv sync` installs from `uv.lock`; run things with `uv run …`. This is where the decision-layer phases (2–5), evaluation, and the demo server run — all CPU, no CUDA-driver entanglement, so uv's clean resolution is ideal.
- **Kaggle / Colab (GPU, pre-provisioned):** keep the platform's existing torch+CUDA; do NOT have uv rebuild the GPU stack. Notebooks `pip install` the behaviour-affecting libraries (`transformers`, `peft`, `bitsandbytes`, `unsloth`, `trl`, `accelerate`) on top of the platform torch, pinned to the versions declared in `pyproject.toml`. The P1.2 batched-vs-sequential gate is what verifies this runtime stack is correct regardless of platform torch.
- `src/` must import cleanly under both; keep `torch` as an unpinned/loosely-pinned peer so the Kaggle platform version satisfies it, while behaviour-critical libs are exactly pinned.
