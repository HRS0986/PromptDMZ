"""C6 — Per-adapter temperature calibration.

For each adapter i, fit a scalar T_i > 0 minimising BCE of sigmoid(d_i / T_i) against labels
on the T-split (scipy `minimize_scalar`, bounds [0.05, 20]). At inference: p̂_i = σ(d_i / T_i).

Why it matters: each fine-tuned adapter is miscalibrated by a different amount, so "0.8" means
different things from different adapters and the fusion layer would waste capacity undoing the
distortion. Temperature scaling is monotone, so it provably cannot change any adapter's
ranking or accuracy — that claim goes in the thesis and is checked by the P2.2 monotonicity
test.

Fitted on T-split; ECE measured on F-split (data the temperatures were NOT fitted to).
Artefact: `temperatures.json`. Fitting order is rigid: C6 -> C7 -> C8.

Implemented by P2.1 (fit) and P2.2 (ECE + reliability-diagram data).
"""
