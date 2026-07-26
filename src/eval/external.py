"""Tier-2 external benchmark loader (ARCHITECTURE §4.1) — cross-dataset generalisation.

Open-Prompt-Injection (Liu et al.) preferred; deepset `prompt-injections` (HF) as the lighter
fallback if OPI's task structure does not map cleanly to binary.

The binary INJECTION/BENIGN mapping — and every task type dropped, with the reason — must be
documented in this module's docstring once chosen, so the thesis can state exactly what was
evaluated. Row counts and class balance are logged.

This loader is READ-ONLY and fits nothing. Tier 2 is scored through the identical pipeline and
the identical decision-layer artefacts as tier 1: no re-fitting, no re-thresholding. Expect a
generalisation drop versus tier 1 — report it, do not hide it (P7.5).

Implemented by P6.6. Notebook import surface: `load_external_benchmark`.
"""
