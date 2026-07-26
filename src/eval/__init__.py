"""Evaluation harness — the data-split protocol (§3) and the evaluation protocol (§4).

    splits.py     T/F/C partition of UNIFIED-VAL, overlap hashing, split manifests
    metrics.py    P/R/F1/AUROC, TPR@1%FPR (headline), ECE, leakage matrix, latency/VRAM
    bootstrap.py  1,000-resample stratified CIs on F1 and TPR@1%FPR (CPU, no GPU)
    baselines.py  configs (g) perplexity+LightGBM, (h) TF-IDF+RF/LR
    external.py   tier-2 benchmark loading + binary mapping
    run_eval.py   the single-pass test evaluation producing the §4 tables

UNIFIED-TEST is sealed until Phase 7 and used exactly once. No module here may read it before
then, and tiers 1-3 are all scored with the decision-layer artefacts applied read-only.
"""
