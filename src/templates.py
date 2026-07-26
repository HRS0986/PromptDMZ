"""C2 — Canonical templating. THE ONLY PLACE PROMPTS ARE BUILT.

Three divergent `build_prompt` implementations across the legacy notebooks
(`LoRA-Fine-Tuning.ipynb`, `Evaluation.ipynb`, `Final_Inference_Pipeline.ipynb`) differed in
whitespace/punctuation/newlines and caused silent accuracy loss. No other module — and no
notebook — may construct a prompt string.

Two scoring modes exist because the datasets were built with deliberate template VARIATION
(anti-overfitting, per EDA.ipynb), so there is no single canonical template:

  * Dataset-row scoring (all split evaluations): use the row's own stored `formatted_text`,
    answer-stripped. Never regenerate a template for a stored row.
  * Live inference: one frozen representative variant per adapter.

Non-negotiable: templates end at `<start_of_turn>model\\n` with NO answer text. Never score a
string containing the gold label.

Implemented by P0.2. Deliverables: `get_prompt_without_answer`, `extract_raw_prompt`,
`build_prompt`, and the golden tests that freeze the chosen variants.
"""
