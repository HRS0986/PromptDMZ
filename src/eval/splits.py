"""Data-split protocol (ARCHITECTURE §3). Violations here invalidate every downstream result.

Source material is the three Hub datasets' EXISTING 80/10/10 stratified splits. Never
re-split.

    UNIFIED-VAL  = union of the three validation splits, source-category tagged
    UNIFIED-TEST = union of the three test splits — SEALED until Phase 7, used exactly once

Because benign pools were drawn from shared sources across the three datasets, cross-dataset
collisions are likely: exact-text dedup within each union AND across the VAL/TEST boundary is
mandatory, not optional (§5 pitfall 11). Drop colliders from VAL and log the counts.

UNIFIED-VAL is partitioned (stratified by label × source category, fixed seed) into three
disjoint parts, consumed in this rigid order — C6 -> C7 -> C8:

    T-split  30%   temperature fit                              (C6)
    F-split  40%   fusion training + scaler + rare-token table  (C7, C3)
    C-split  30%   conformal calibration, ≥100-200 benign       (C8)

Hard gates (P0.3): pairwise overlap between T/F/C/UNIFIED-TEST must be 0 exact-hash
collisions, and the C-split must contain ≥100 benign examples or the run fails.

Every output file is labelled with split name + prompt-set hash. That is what makes the
previously-observed failure mode — identical val and test metrics from a duplicated run
(§5 pitfall 7) — structurally impossible rather than merely unlikely.

Implemented by P0.3. Notebook import surface: `build_unified_splits`, `load_manifests`.
"""
