"""Non-adapter baselines — configs (g) and (h) of the evaluation matrix (ARCHITECTURE §4.2).

    (g) Perplexity + LightGBM (Alon & Kamfonas): GPT-2 (small) windowed perplexity features
        -> LightGBM. A cheap, citable, non-neural-detector floor. Run standalone on the GPU so
        the GPT-2 load has the T4 budget to itself. Fitted on F-split ONLY.
    (h) TF-IDF + RandomForest / LogisticRegression (Shaheer et al.): word+char TF-IDF, the
        classical floor. CPU only. FIRST thing to cut if time runs short.

Both fit on F-split only — never on any test tier.

Explicitly OUT OF SCOPE, do not attempt: faithful reimplementation of DataSentinel or
Attention Tracker. They are cited and compared qualitatively, with an honest scoping statement
that reproduction was infeasible under T4 constraints. Partial reimplementations are worse
than none.

Implemented by P6.3 and P6.4.
"""
