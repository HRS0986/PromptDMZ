# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A prompt-injection detection system for a final-year thesis. Three category-specialised QLoRA
adapters share one frozen 4-bit Gemma3-1B-it backbone and run **simultaneously** (batched, one
adapter per sample). Their calibrated probabilities plus tokenizer-statistics features are combined
by a small **learned fusion** layer, and a **conformal threshold** gives a certified false-positive
bound. Output is binary: `INJECTION` or `BENIGN`, with per-category attribution.

The novelty is the *integration*, not any single mechanism: calibrated learned fusion over
specialist adapters on a shared quantized backbone, evaluated against conventional rule-based
(disjunctive) fusion and a single merged/generalist adapter, under T4 VRAM constraints.

Terminology: never call the rule-based baseline "the previous architecture" — it is the
literature-standard disjunctive fusion approach (MoJE, WAInjectBench), and formally the `q_i = 1`
constrained case of the noisy-OR fuser.

## Read these first (authoritative specs)

- `docs/ARCHITECTURE.md` — components C1–C10 (input / process / output / justification), the
  data-split protocol, evaluation matrix, known pitfalls, module layout. This is the design contract.
- `docs/TASKS.md` — phased implementation plan (Phase 0–9) with per-task acceptance criteria, and
  the GPU-vs-CPU execution guide.

Treat both as the source of truth. If a request conflicts with them, say so and ask before diverging.

## How to work here

- **One phase at a time.** Do not start a phase until the user confirms the previous phase's
  acceptance criteria (in `docs/TASKS.md`) pass. Do not jump ahead or scaffold future phases early.
- **Acceptance criteria are the definition of done.** Every task in `docs/TASKS.md` has AC — implement
  to satisfy them, and write the test that proves each one.
- **Prefer editing over rewriting.** Reuse the user's existing notebooks and Hub assets (see below);
  do not regenerate things that already exist.
- **Ask before destructive or irreversible actions** (deleting artefacts, force-pushing, retraining
  an adapter). Retraining is almost never needed in the core build.

## Reuse — do NOT rebuild (assets that already exist)

**Notebooks in `notebooks/` fall into two groups — treat them differently:**
- **Reference only (READ, never run or edit):** `FT_Merged_Adapter.ipynb`, `LoRA-Fine-Tuning.ipynb`,
  `EDA.ipynb`, `Final_Inference_Pipeline.ipynb`, `Evaluation.ipynb`. These are the training recipe,
  dataset-building logic, and legacy comparator — read them to extract templates/recipe/comparator
  behaviour, but never execute or modify them.
- **Runnable entry points (created for this build):** `01_scoring_kaggle.ipynb`,
  `02_score_baseline_kaggle.ipynb`, `03_test_eval_kaggle.ipynb` — the Kaggle GPU notebooks; they
  import and call `src/` only.


- **3 specialist adapters** (HF, private): `hirushafernando/fyp-gemma3-1b-slm-a-qlora` (role/instruction
  violation), `-b-` (privilege escalation), `-c-` (obfuscation/evasion). Frozen for the whole core build.
- **Baseline (merged/generalist) adapter — ALREADY TRAINED**: `hirushafernando/fyp-gemma3-1b-slm-merged-qlora`.
  It is the experimental control, NOT part of the running system. Phase 6 loads and scores it — never retrain it.
- **Datasets** (HF, private): `hirushafernando/fyp-slm-merged` (108,798 / 21,489 / 21,492) and the
  three per-category `hirushafernando/fyp-slm-a`, `fyp-slm-b`, `fyp-slm-c`. Columns are
  `formatted_text` + `label` (0=BENIGN, 1=INJECTION).
  Existing 80/10/10 splits are reused verbatim; never re-split.
- **Notebooks** (repo): `FT_Merged_Adapter.ipynb` / `LoRA-Fine-Tuning.ipynb` are the training recipe
  reference (r=16, α=16 — confirmed identical for specialists and baseline); `EDA.ipynb` built the
  merged dataset; `Final_Inference_Pipeline.ipynb` is the legacy generate+parse comparator for P1.3.
  `Evaluation.ipynb` is the source of the label-leakage bug being fixed — do not copy its scoring path.

## Non-negotiable rules (violating these invalidates thesis results)

1. **`src/templates.py` is the only place prompts are built.** Three divergent legacy `build_prompt`
   implementations caused silent accuracy loss; never reintroduce ad-hoc templating in notebooks or
   elsewhere.
2. **No label leakage.** Never score a string containing the answer. Score stored rows via
   `get_prompt_without_answer()` (split on `<start_of_turn>model`); templates end at
   `<start_of_turn>model\n` with no answer text.
3. **The test split (UNIFIED-TEST) is sealed until Phase 7** and used exactly once. No fitting,
   tuning, or peeking on test data. Fitting order is rigid: calibration (C6) → fusion (C7) →
   conformal (C8).
4. **Batched must equal sequential.** The P1.2 gate — batched per-sample adapter scoring agrees with
   sequential `set_adapter` scoring within fp16 tolerance — must pass before any downstream work is
   trusted. If `adapter_names` is unsupported on the runtime stack, use the sequential-no-early-exit
   fallback; outputs must be identical.
5. **Tokenizer stats are computed on RAW prompt text**, before templating/marking.
6. **The extension seam.** All fusion inputs come from `build_feature_vector(...)`; no other module
   hardcodes feature width. This is what lets future signals (e.g. a correction channel) be added by
   appending slots + refitting.
7. **Standardise on the 4-bit backbone everywhere** (`unsloth/gemma-3-1b-it-unsloth-bnb-4bit`). The
   legacy fp16 evaluation numbers are not comparable to new results; do not mix them.
8. **Evaluation rules.** The headline metric is **TPR @ 1% FPR**, not accuracy — accuracy is
   appendix-only (imbalanced classes). Every headline number carries a **bootstrap 95% CI**
   (1,000 resamples of saved predictions; CPU, no re-scoring). All three test tiers (UNIFIED-TEST,
   external benchmark, benign stress set) are scored with the decision-layer artefacts applied
   **read-only** — never re-fit or re-threshold on them. The benign stress set is authored and
   frozen BEFORE any Phase 7 results are seen. Do NOT attempt to reimplement DataSentinel or
   Attention Tracker — they are cited and compared qualitatively only.

## Environment & package management (uv)

- **Local / FastAPI serve (reproducible):** managed by `uv`. Use `uv sync` to install, `uv run …` to
  execute. Declare deps in `pyproject.toml`; commit `uv.lock`.
- **Kaggle / Colab (GPU):** keep the platform's pre-provisioned torch+CUDA. Do NOT `uv sync` there.
  Notebooks `pip install` the behaviour-critical libs (`transformers==4.53.1`, `unsloth==2025.7.2`,
  `peft`, `bitsandbytes`, `trl`, `accelerate`) on top of platform torch. `torch` stays loosely pinned
  so the platform build satisfies it; behaviour-critical libs are exactly pinned to the versions the
  adapters were trained under.
- The P1.2 gate doubles as the cross-environment sanity check.

## GPU vs CPU (where work runs)

- **GPU (Kaggle T4), three notebooks in order:** `01_scoring_kaggle.ipynb` (splits + verification +
  bulk-score T/F/C), `02_score_baseline_kaggle.ipynb` (load + score the existing baseline — no
  training), `03_test_eval_kaggle.ipynb` (sealed test + latency/VRAM, run last).
- **CPU (local / laptop):** everything else — calibration, tokenizer stats, fusion, conformal, the
  evaluation tables, and the FastAPI server. These operate on the parquet score files, in seconds.
- **Hand-off** between GPU and CPU work is via the private HF artefact dataset repo
  `hirushafernando/slm-shield-artifacts` (scores/manifests out, fitted artefacts back).
- Notebooks contain NO logic — they import and call `src/`.

## Coding conventions

- Optimise for low VRAM (T4 ~15–16 GB); `torch.inference_mode()` for all inference; never fork GPU
  workers. Bulk scoring must be resumable (skip already-scored ids) — sessions die.
- Small, testable functions in `src/`; pytest tests alongside, especially the golden tests
  (frozen templates, no-leakage assertion, quantile-rule check, batched==sequential).
- Persist every artefact after each stage (parquet/json/pkl) to the artefact repo; nothing may
  depend on a live session.
- Do not add ML/heuristic components beyond what `docs/ARCHITECTURE.md` specifies without asking.

## Definition of done for a phase

The phase's acceptance criteria in `docs/TASKS.md` pass, the relevant tests are green, artefacts are
persisted, and the user has reviewed. Then — and only then — proceed to the next phase.
