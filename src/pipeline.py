"""C9 — Decision + attribution: the end-to-end detect(prompt) path.

Applies every persisted artefact READ-ONLY (temperatures, scaler, fuser, τ̂). A pure function
of prompt + artefacts; it fits nothing. Verdict is `INJECTION` if S(x) > τ̂ else `BENIGN`.

    {"verdict": "INJECTION" | "BENIGN",
     "score": S, "threshold": tau,
     "category_scores": {"role_violation": .., "privilege_escalation": .., "obfuscation_evasion": ..}}

Attribution survives the upgrade because fusion consumes, rather than destroys, the
per-adapter calibrated probabilities.

This module is the single definition of "the system". The FastAPI server is a thin wrapper
over it and contains no detection logic of its own, so the demo cannot drift from the
evaluated system — P9.1 asserts `detect()` reproduces the Phase 7 verdicts on 20 known
prompts.

Implemented by P9.1.
"""
