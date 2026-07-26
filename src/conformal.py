"""C8 — Conformal threshold (certified FPR bound).

Split-conformal quantile with the finite-sample correction, over the fused scores of BENIGN
examples in the C-split (which is never used for any fitting):

    scores = sorted(S(x) for benign x in C_split)   # n values
    k      = ceil((1 - alpha) * (n + 1))            # NOT (1 - alpha) * n
    tau    = scores[k - 1]                          # k-th smallest

Headline α = 0.01, chosen to align with the TPR@1%FPR headline metric so both describe the
same operating point; also sweep {0.01, 0.05, 0.10}.

What it buys: for exchangeable benign inputs, P(S(x) > τ̂) ≤ α up to O(1/n) — a distribution-
free guarantee rather than "good empirical FPR". The guarantee is marginal, not per-instance;
say so honestly in the thesis. The (n+1) correction makes τ̂ slightly conservative, so observed
FPR typically lands below α and TPR slightly lower — quantify that gap as the price of the
guarantee.

Hard gate: C-split needs ≥100-200 benign examples (bare minimum 1/α = 100 at α=0.01).
`conformal.json` records n, α, τ̂, fuser id and artefact hashes — the threshold is invalid if
the fuser changes.

Implemented by P5.1 (threshold) and P5.2 (bound sanity).
"""
