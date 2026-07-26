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
    within fp16 tolerance (max |Δp| < 1e-3) on ≥200 prompts. No batched number is trusted until
    this passes.
  * Left-padding is mandatory for causal-LM last-position extraction (or explicit last-non-pad
    indexing). One silent mistake here corrupts every probability.
  * Bulk scoring must be resumable — skip already-scored ids; Kaggle sessions die.

Implemented by P1.2 (scorer), P1.3 (legacy agreement), P1.4 (bulk runs). Notebook import
surface: `verify_batched_vs_sequential`, `legacy_agreement_check`, `bulk_score_split`.
"""
