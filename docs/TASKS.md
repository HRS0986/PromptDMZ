# TASKS.md — Implementation Plan

Companion to ARCHITECTURE.md. Phases are strictly ordered; do not start a phase before its
predecessor's acceptance criteria pass. Each task lists: deliverable, acceptance criteria (AC),
and rough effort. GPU tasks assume a single T4; everything else is CPU.

Global rules:
- All prompt construction goes through `src/templates.py`. No other file may build prompts.
- After every GPU stage, persist outputs (parquet/json/pkl) to Drive; scoring must be resumable.
- The TEST split is not read by any code until Phase 7.

---

## Phase 0 — Foundations (0.5–1 day)

- [x] **P0.1 Repo scaffold + uv**: create `src/` layout per ARCHITECTURE §6; init `uv`
      (`uv init`), declare deps in `pyproject.toml` — behaviour-critical libs pinned to the
      versions from the working notebooks (transformers==4.53.1, unsloth==2025.7.2, plus peft,
      bitsandbytes, trl, accelerate at their known-good versions), `torch` left loose so platform
      GPU builds satisfy it; `uv lock` to produce `uv.lock`. Add a `[project.optional-dependencies]`
      `serve` group for FastAPI/uvicorn. Provide the Kaggle install snippet (plain `pip install`
      of the pinned behaviour libs on top of platform torch — NOT `uv sync`, which must not rebuild
      the GPU stack).
      AC: `uv sync` builds the local env and `uv run python -c "import src"` succeeds; a fresh
      Kaggle session runs the notebook install cell and imports `src` cleanly on top of platform
      torch; pinned versions match the working-notebook versions.
- [x] **P0.2 Template handling** (`src/templates.py`) — NOTE: datasets use deliberate template
      VARIATION (EDA.ipynb), so there is no single canonical template. Implement:
      (i) `get_prompt_without_answer(formatted_text)` — answer-strip stored rows by splitting on
      `<start_of_turn>model` (the scoring path for ALL split evaluations);
      (ii) `extract_raw_prompt(formatted_text)` — recover the raw user prompt IF the Hub datasets
      lack a raw `text` column (verify first; needed for tokenizer stats + cross-adapter scoring);
      (iii) `build_prompt(prompt, adapter)` — ONE frozen representative variant per adapter
      (highest-frequency variant in the training split; document the choice) for live inference.
      AC: golden tests freeze the three chosen variants; answer-strip test asserts no label token
      after the model-turn marker on 1 000 sampled rows; raw-prompt extractor round-trips on
      1 000 sampled rows (or the raw column is confirmed and the extractor is skipped).
- [ ] **P0.3 Unified splits + manifests**: pull the three Hub datasets (`hirushafernando/fyp-slm-a`,
      `fyp-slm-b`, `fyp-slm-c`); build UNIFIED-VAL (union of validation splits) and UNIFIED-TEST (union of test
      splits) with source-category tags; exact-text dedup within each union AND across the
      VAL/TEST boundary (drop colliders from VAL, log counts — shared benign pools make
      collisions likely); partition UNIFIED-VAL into T/F/C (stratified by label × category, fixed
      seed); write manifests with prompt hashes.
      AC: pairwise overlap between T/F/C/UNIFIED-TEST = 0 exact-hash collisions; C-split contains
      ≥100 benign examples (hard fail otherwise); manifests persisted; collision counts logged.

## Phase 1 — Scoring core (1.5–2 days, GPU)

- [ ] **P1.1 Model loader** (`model_loader.py`): backbone `unsloth/gemma-3-1b-it-unsloth-bnb-4bit`
      (the training checkpoint — NOT plain google/gemma-3-1b-it fp16 as the legacy Evaluation
      notebook did) + the 3 Hub adapters `fyp-gemma3-1b-slm-a/b/c-qlora` (private; HF_TOKEN),
      all memory-resident. Seed loading code from `Final_Inference_Pipeline.ipynb`.
      AC: loads on T4 with peak VRAM logged (<~3 GB expected incl. overhead); all three adapter
      names listed by PEFT; smoke inference runs.
- [ ] **P1.2 Batched scorer** (`scoring.py`): label-token id derivation (empirical, logged),
      batched forward with `adapter_names`, last-non-pad logit extraction (left padding),
      two-way softmax → (p_raw, d) per adapter; sequential `set_adapter` fallback path.
      AC: **batched vs sequential probabilities agree within fp16 tolerance (max |Δp| < 1e-3)
      on ≥200 prompts** — this test gates everything downstream. Label ids: INJ_ID ≠ BEN_ID,
      both asserted stable across 3 template variants of the same prompt (per adapter).
- [ ] **P1.3 Legacy agreement check**: run new scorer + the legacy generate-parse pipeline
      (reuse `Final_Inference_Pipeline.ipynb`'s predict functions verbatim as the comparator)
      on the same labelled sample (n≈500), both on the 4-bit backbone.
      AC: argmax agreement ≥ 99%; disagreement cases dumped for inspection with legacy parse
      output — record the legacy parse-failure rate (this number goes in the thesis).
- [ ] **P1.4 Bulk scoring runs**: score T, F, C splits (NOT test) with resumable writer. Input =
      each row's own answer-stripped `formatted_text` for the row's home adapter, plus the frozen
      variants (P0.2 iii) wrapping the raw prompt for the other two adapters (cross-adapter
      scoring — every row gets all three probabilities).
      AC: parquet per split with columns [prompt_hash, split, label, category, d1..d3, p1..p3];
      resume-after-kill test passes; files on Drive.

## Phase 2 — Calibration (1 day, CPU)

- [ ] **P2.1 Temperature fitting** (`calibration.py`): fit T_i per adapter on T-split by BCE on
      σ(d_i/T_i); persist `temperatures.json`.
      AC: all T_i in [0.05, 20]; fitted BCE ≤ uncalibrated BCE per adapter.
- [ ] **P2.2 ECE + reliability diagrams**: 10–15 bin ECE per adapter, before/after, on F-split
      (not T-split — measure on data the temperatures were not fitted to).
      AC: post-calibration ECE ≤ pre-calibration ECE for each adapter; reliability-diagram data
      saved (plots generated in Phase 7); accuracy identical pre/post (monotonicity check).

## Phase 3 — Tokenizer statistics (0.5–1 day, CPU)

- [ ] **P3.1 Stats module** (`tokenizer_stats.py`): 4 features on RAW text; rare-token table
      fitted on F-split benign portion; scaler fitted on F-split.
      AC: unit tests — base64 blob yields fertility & rare-fraction above benign-English
      medians; homoglyph string yields byte_fallback_rate > 0; plain English near medians.
      Stats computed pre-template (test asserts template tokens absent from the tokenization
      being measured).

## Phase 4 — Fusion (1 day, CPU)

- [ ] **P4.1 Feature assembly seam** (`fusion.py`): `build_feature_vector(p_hat, stats_scaled)`
      → shape (7,); versioned layout constant; NOTHING else hardcodes feature width.
      AC: unit test on shape/ordering; grep-level check that no other module indexes features
      positionally.
- [ ] **P4.2 Three fusers**: noisy-OR (learned q, probs-only — 3 params), LR (all 7 feats,
      standardised — 8 params), MLP (8–16 hidden, early stopping) trained on F-split; internal CV
      for hyperparams. **Variant selection is by TPR@1%FPR** under F-split CV (the headline metric),
      tie-broken on AUROC — never select on a different metric than the one reported.
      AC: each trains in <60 s CPU; artefacts persisted; noisy-OR with q=1 + 0.5 thresholds
      reproduces conventional rule-based (probabilistic-OR) fusion exactly (regression test — this
      IS the baseline link: the baseline is the constrained q=1 case, not a separate system);
      LR weights logged (expect privilege-escalation down-weighted — note either way);
      parameter count of each fuser recorded for the efficiency table.
- [ ] **P4.3 Uncalibrated ablation twin**: same fusers trained on raw p (config (c) of the
      evaluation matrix).
      AC: artefacts saved under distinct names; no shared scaler with the calibrated variant.

## Phase 5 — Conformal (0.5 day, CPU)

- [ ] **P5.1 Threshold** (`conformal.py`): benign C-split scores through the selected fuser;
      k = ceil((1−α)(n+1)); τ̂ = k-th smallest; **headline α=0.01** (aligns with the TPR@1%FPR
      headline metric), α sweep {0.01, 0.05, 0.10}.
      AC: unit test of the quantile rule on synthetic data (analytic check); `conformal.json`
      records n, α, τ̂, fuser id + artefact hashes (threshold invalid if fuser changes).
- [ ] **P5.2 Bound sanity**: empirical FPR of τ̂ on C-split itself ≤ α by construction (test);
      document that the real check is TEST FPR vs α in Phase 7.

## Phase 6 — Baselines & evaluation assets (1.5–2 days; GPU for scoring only — NO adapter training)

- [ ] **P6.1 Baseline adapter — ALREADY TRAINED, just load + score**: the merged/generalist adapter
      `hirushafernando/fyp-gemma3-1b-slm-merged-qlora` exists (built by `FT_Merged_Adapter.ipynb`,
      r=16, α=16, same recipe as specialists). Load it and score F-split via the SAME scoring path
      (C5 label-logit extraction). Its own template is the merged-instruction template used in the
      merged dataset — freeze that variant in templates.py; verify label token ids match.
      AC: baseline scored on F-split with the new scoring path; NO retraining performed.
      Hyperparameters (r=16, α=16) are confirmed identical to the specialists, so the
      specialist-vs-generalist comparison is properly controlled.
- [ ] **P6.2 Conventional rule-based fusion baseline** (config (b), literature-standard disjunctive
      fusion — cite MoJE / WAInjectBench; do NOT describe it as "the previous architecture"):
      (b1) probabilistic-OR `S = 1 − Π(1 − p_i)` on uncalibrated probs — natively continuous;
      (b2) hard rule `any p_i > τ` with **shared τ swept 0→1** for its ROC curve;
      (b3) optional strong variant with per-adapter τ optimised on F-split.
      Plus the sequential-with-early-exit latency variant for the efficiency table.
      AC: (b1) reproduced via the noisy-OR q=1 regression test (P4.2); (b2) produces a full ROC
      curve — a single fixed τ=0.5 point is NOT acceptable as a matched-FPR comparison; operating
      point for the headline number selected on F/C-split, not on test.
- [ ] **P6.3 Perplexity + LightGBM baseline** (config (g), `eval/baselines.py`): GPT-2 (small)
      windowed perplexity features → LightGBM, fitted on F-split ONLY.
      AC: fits on F-split, scores F-split held-out portion; GPT-2 load does not exceed T4 budget
      alongside nothing else (run standalone); artefact persisted; cite Alon & Kamfonas.
- [ ] **P6.4 TF-IDF + RF/LR baseline** (config (h), `eval/baselines.py`): word+char TF-IDF →
      RandomForest and LogisticRegression, fitted on F-split ONLY. **First thing to cut if time is short.**
      AC: CPU-only; fits in <5 min; artefacts persisted; cite Shaheer et al.
- [ ] **P6.5 Benign stress set** (`data/benign_stress_set.jsonl`, tier 3): hand-author ~100–300
      all-benign prompts containing trigger vocabulary in innocuous contexts (e.g. "ignore the
      previous paragraph", RBAC/permission discussions, base64 explained in a tutorial, code with
      `sudo`/`admin` identifiers, security-course questions). Freeze BEFORE Phase 7.
      AC: file committed with a provenance note (author, date, design rationale per item category);
      every row labelled benign; no overlap with any split (hash check); NOT used for any fitting.
- [ ] **P6.6 External benchmark loader** (`eval/external.py`, tier 2): load Open-Prompt-Injection
      (preferred) or deepset `prompt-injections`; map to binary INJECTION/BENIGN.
      AC: mapping documented in the module docstring incl. any task types dropped and why; row
      counts + class balance logged; loader is read-only and fits nothing.

## Phase 7 — Final evaluation (1.5–2 days; ALL test-tier data touched here ONLY)

- [ ] **P7.1 Test scoring (all three tiers)**: single bulk run through the scorer for
      UNIFIED-TEST (tier 1), the external benchmark (tier 2), and the benign stress set (tier 3),
      for the specialists AND the merged baseline. Decision-layer artefacts applied READ-ONLY —
      no re-fitting, no re-thresholding on any tier.
      AC: outputs tagged with tier + manifest hash; **val-vs-test distinguishability check**:
      assert test metrics are not bit-identical to any F/C-split metrics file (guards the
      duplicated-run failure mode seen previously); tier-3 rows are all benign as authored.
- [ ] **P7.2 Full matrix** (`run_eval.py`): configs (a)–(h) × §4.3 metrics, per tier.
      Primary: Precision, Recall, F1, AUROC, **TPR@1%FPR (headline)**, benign FPR reported
      separately. Accuracy appendix-only. Plus ECE (pre/post, per adapter) + reliability diagrams,
      conformal empirical coverage (does α hold on tiers 1 and 3?), per-category recall,
      cross-category leakage matrix, legacy parse-failure rate.
      AC: one machine-readable results.json + rendered markdown tables + reliability/ROC plots;
      every number traceable to an artefact hash; TPR@1%FPR computed by threshold sweep on the
      score distribution (document the interpolation rule).
- [ ] **P7.3 Bootstrap confidence intervals** (`eval/bootstrap.py`): 1,000 **stratified** resamples
      of the saved test predictions; 95% CIs on F1 and TPR@1%FPR for every configuration and tier.
      CPU only — no GPU, no re-scoring. 3 seeds additionally if compute allows.
      Two mandatory implementation details: (i) resample benign and attack pools SEPARATELY, each
      to original size (fixes the FPR denominator); (ii) RECOMPUTE the 1%-FPR threshold inside each
      replicate — the threshold's tail sensitivity is the dominant variance source, and fixing it
      understates the CI.
      AC: CI reported alongside every headline number; explicit statement of whether the
      specialist-vs-generalist gap and the learned-fusion-vs-rule-based gap exceed their CIs.
- [ ] **P7.4 Efficiency benchmark**: peak VRAM (`torch.cuda.max_memory_allocated`), **median and
      p95** latency per prompt, batched-vs-sequential throughput, parameter counts (adapters vs
      decision layer). All in ONE T4 session so numbers are mutually comparable.
      AC: single results table; session/GPU model recorded; batched and sequential measured on the
      same prompt set.
- [ ] **P7.5 Results narrative check**: verify the headline claims are supported or honestly
      reported as negative — (i) calibrated fusion vs conventional rule-based fusion at 1% FPR,
      (ii) specialists+fusion
      vs merged baseline, (iii) observed FPR ≤ certified α on tiers 1 and 3, (iv) generalisation
      on tier 2 vs tier 1 (expect a drop — report it, don't hide it).
      AC: a short RESULTS_SUMMARY.md stating each claim with its number AND its bootstrap CI;
      negative or inconclusive results are recorded, not hidden. Include the scoping statement
      that DataSentinel / Attention Tracker were cited but not reimplemented (T4 constraints).

## Phase 8 — Optional extensions (only after Phase 7 is green)

- [ ] **P8.1 Gap 3 — prompt-leaking coverage** (~1–1.5 d): add leaking examples to role-violation
      train+test; retrain that adapter; **refit Phases 2→5 artefacts**; re-run Phase 7 delta table.
- [ ] **P8.2 Gap 1 — semantic obfuscation** (~1.5–2 d, data-dependent): source paraphrase/AutoDAN-style
      examples; retrain obfuscation adapter; refit 2→5; delta table. If no usable public data →
      document as Further Work instead.
- [ ] **P8.3 TokenBreak vulnerability probe** (~1 d): perturbation generator over TEST attacks;
      measure recall drop of the final system. (Defence/correction channel remains out of scope;
      this measurement alone is a reportable robustness finding and motivates the Further Work.)
- [ ] **P8.4 Gap 2 — provenance marking** (~1.5 d): regenerate privilege-escalation data with
      marking, retrain, refit 2→5. Lowest priority; acceptable as Further Work.

Refit rule for ALL of Phase 8: any adapter retrain invalidates temperatures, fusers, scaler and
τ̂ — rerun Phases 2–5 (minutes of CPU) before any new evaluation.

## Phase 9 — FastAPI demonstration server (0.5–1 day, CPU-buildable, GPU-optional)

- [ ] **P9.1 Pipeline wrapper** (`src/pipeline.py`): `detect(prompt) -> dict` applying all persisted
      artefacts read-only (C1→C9). Pure function of prompt + artefacts; no fitting.
      AC: on 20 known prompts, `detect()` verdict matches the Phase 7 scored verdict for the same
      prompts (the API cannot diverge from the evaluated system).
- [ ] **P9.2 Server** (`src/serve/app.py`, `schemas.py`): load model+artefacts once at startup on
      `app.state`; `POST /detect`, `POST /detect_batch`, `GET /health`, `GET /` demo form;
      `--cpu` fallback flag for laptop demos.
      AC: `GET /health` returns ready; `POST /detect` returns the full response schema incl.
      category_scores and latency_ms; loading happens exactly once (log line at startup only);
      README documents `uv run uvicorn src.serve.app:app --reload` (and `--cpu` flag) and a sample curl.

---

## GPU vs CPU split (execution guide)

**GPU (Kaggle T4) — use the three provided notebooks, in order:**
- `01_scoring_kaggle.ipynb` — Phases 0.3, 1.1–1.4 (splits + verification + bulk score T/F/C)
- `02_score_baseline_kaggle.ipynb` — Phase 6.1 (LOAD + score the already-trained baseline; no training)
- `03_test_eval_kaggle.ipynb` — Phase 7 (sealed test scoring + latency/VRAM), run last

The original plan's heaviest GPU job (training the generalist) is ALREADY DONE via
`FT_Merged_Adapter.ipynb`, so notebook 02 is now load-and-score only — a short GPU session.

**CPU (laptop / Colab CPU / Kaggle CPU) — everything else:**
- Phases 2 (calibration), 3 (tokenizer stats), 4 (fusion), 5 (conformal): operate on the parquet
  score files produced by notebook 01 — no GPU needed, seconds of compute.
- Phase 9 (FastAPI) builds and runs on CPU; use GPU only for a fast live demo.

Data hand-off between GPU and CPU work is via the private HF **artefact dataset repo**
(`<user>/slm-shield-artifacts`): notebook 01 pushes scores+manifests there; CPU phases pull them,
fit the decision layer, push artefacts back; notebook 03 pulls the fitted artefacts for the sealed run.

---

## Effort summary

| Block | Phases | Est. focused days |
|---|---|---|
| Core (must-have) | 0–5 | 5.5–7 |
| Baselines: existing adapter + rule-based fusion + classical + stress set + external loader | 6 | 1.5–2 |
| Final evaluation: 3 tiers × 8 configs + bootstrap + efficiency | 7 | 2–2.5 |
| FastAPI demo (must-have for demonstration) | 9 | 0.5–1 |
| Extensions (optional, ordered) | 8 | 1–6 |

Note: the merged/generalist baseline adapter is ALREADY TRAINED (`FT_Merged_Adapter.ipynb`), so the
single largest GPU job in the original plan is already done — Phase 6 trains nothing.

Calendar multiplier for Kaggle session limits and life: ×~1.5.

**Cut-line if time runs short (in order):** P6.4 (TF-IDF baseline) first — it is the classical floor
and least informative. Then P8.4, P8.2, P8.1. Never cut: P7.3 (bootstrap CIs — zero GPU cost, high
viva value), P6.5 (stress set — it is the FPR argument), or the tier-1 evaluation.
