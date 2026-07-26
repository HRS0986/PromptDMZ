"""C7 — Learned fusion. Contains THE EXTENSION SEAM.

All fusion inputs are assembled by a single function; no other module may hardcode feature
width or index features positionally:

    def build_feature_vector(p_hat, stats_scaled) -> np.ndarray:
        # v1 layout: [p̂1, p̂2, p̂3, s1, s2, s3, s4]  -> shape (7,)
        # future extensions append slots HERE ONLY

This is what lets a future signal (e.g. the spelling-correction channel, out of scope for this
build) be added by appending slots and refitting, without touching anything else.

Three fusers, all trained on the F-split:
    1. Noisy-OR   S = 1 - Π(1 - q_i · p̂_i), learned reliabilities q_i ∈ [0,1] (probs only).
                  At q_i = 1 this reduces EXACTLY to conventional rule-based (probabilistic-OR)
                  fusion — the literature-standard disjunctive baseline (MoJE, WAInjectBench)
                  is the constrained q=1 case of this family, not a separate system and never
                  "the previous architecture". P4.2 has a regression test asserting this.
    2. Logistic regression over all 7 features, standardised — interpretable weights.
    3. MLP, one hidden layer (8-16 units), early stopping — captures interactions.

Selection rule: best TPR@1%FPR under F-split internal CV — the SAME metric that is reported.
Never select on one metric and report another. Tie-break on AUROC. All variants carry into the
final test table as the fusion ablation.

Implemented by P4.1 (seam), P4.2 (fusers), P4.3 (uncalibrated ablation twin).
"""
