"""Stratified bootstrap confidence intervals (ARCHITECTURE §4.3).

1,000 resamples of the SAVED test predictions — CPU only, zero GPU cost, no re-scoring. This
is what lets the thesis state whether the specialist-vs-generalist gap is significant rather
than noise, so it is on the never-cut list.

Two mandatory implementation details:
  (i)  Resample the benign and attack pools SEPARATELY, each back to its original size, so the
       FPR denominator is fixed.
  (ii) RECOMPUTE the 1%-FPR threshold inside each replicate. The threshold's tail sensitivity
       is the dominant variance source for this metric; holding it fixed understates the CI.

95% CIs on F1 and TPR@1%FPR for every configuration and tier; 3 seeds additionally if compute
allows. Every headline number in the thesis carries its CI.

Implemented by P7.3. Notebook import surface: `bootstrap_cis`.
"""
