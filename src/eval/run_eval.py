"""The single-pass final evaluation producing the §4 tables. PHASE 7 ONLY.

Configs (a)-(h) × §4.3 metrics × three tiers:

    (a) merged/generalist baseline adapter — the specialist-vs-generalist control the thesis
        hypothesis lives or dies on. ALREADY TRAINED; load and score, never retrain.
    (b) 3 specialists + conventional rule-based disjunctive fusion (MoJE, WAInjectBench):
        (b1) probabilistic-OR on uncalibrated probs — natively continuous, has its own ROC;
        (b2) hard rule `any p_i > τ` with SHARED τ swept 0->1 for a full ROC curve. A fixed
             τ=0.5 is a single ROC point and CANNOT be compared at matched FPR — never report
             it as the baseline's TPR@1%FPR;
        (b3) optional strong variant, per-adapter τ optimised on F-split.
    (c) specialists + learned fusion, UNCALIBRATED  (ablates calibration)
    (d) specialists + CALIBRATED learned fusion, probs only
    (e) (d) + tokenizer stats, full feature vector   (ablates the stats channel)
    (f) best of (d/e) + conformal threshold          -> the headline system
    (g) perplexity + LightGBM       (h) TF-IDF + RF/LR

Tiers: 1 UNIFIED-TEST, 2 external benchmark, 3 benign stress set. All three scored ONCE, with
decision-layer artefacts applied read-only.

Guard (P7.1): assert test metrics are not bit-identical to any F/C-split metrics file. A
previous results set showed identical val and test numbers from a suspected duplicated run
(§5 pitfall 7); this check makes that class of error loud.

Output: one machine-readable results.json + rendered markdown tables + reliability/ROC plots,
every number traceable to an artefact hash, and the TPR@1%FPR interpolation rule documented.

Implemented by P7.2. Notebook import surface: `run_full_matrix`.
"""
