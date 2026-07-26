"""C1 — Model + adapter loading (once per session).

Loads the backbone `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` in 4-bit NF4 — the TRAINING
checkpoint, not plain `google/gemma-3-1b-it` in fp16, which is what the legacy Evaluation
notebook used. That train/eval discrepancy is why legacy metrics are not comparable to new
ones (ARCHITECTURE §0). All new scoring is 4-bit.

Attaches the three private Hub adapters as named adapters on one PeftModel, all
memory-resident for the whole session, never merged:

    role_violation        <- hirushafernando/fyp-gemma3-1b-slm-a-qlora
    privilege_escalation  <- hirushafernando/fyp-gemma3-1b-slm-b-qlora
    obfuscation_evasion   <- hirushafernando/fyp-gemma3-1b-slm-c-qlora

Requires HF_TOKEN (repos are private). Seed the loading code from
`Final_Inference_Pipeline.ipynb`.

Implemented by P1.1. Notebook import surface: `load_model_with_adapters`.
"""
