"""Metrics (ARCHITECTURE §4.3).

Headline metric is **TPR @ 1% FPR**, not accuracy. Classes are imbalanced and accuracy
conceals the interesting behaviour — it is appendix-only, if reported at all. The low-FPR
operating point is where a real detector is judged, where rule-based fusion degrades most
(compounded false positives) and calibrated fusion gains most.

    Detection    Precision, Recall, F1, AUROC, TPR@1%FPR (headline); benign FPR reported
                 separately as the utility-preservation number.
    Calibration  10-bin ECE + reliability-diagram data per adapter, before vs after
                 temperature scaling; empirical coverage of the conformal threshold.
    Error        per-category recall, cross-category leakage matrix (which adapter fires on
                 which attack family), legacy parse-failure rate vs 0 for the new path.
    Efficiency   peak VRAM (`torch.cuda.max_memory_allocated`), MEDIAN and P95 latency,
                 batched-vs-sequential throughput, parameter counts. All measured in ONE T4
                 session so the numbers are mutually comparable.

Operating-point discipline: ROC/AUROC are threshold-free and computed on test, but the
SPECIFIC threshold achieving 1% FPR is selected on F- or C-split and then applied to test.
Selecting it on test is tuning on test.

Implemented by P7.2 (matrix) and P7.4 (efficiency). Notebook import surface:
`benchmark_runtime`.
"""
