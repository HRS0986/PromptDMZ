"""SLM-Shield — prompt-injection detection via specialist QLoRA adapters + learned fusion.

Three category-specialised QLoRA adapters share one frozen 4-bit Gemma3-1B-it backbone and
score every prompt simultaneously in a single batched forward pass. Their calibrated
probabilities plus tokenizer-statistics features are combined by a learned fusion layer, and a
split-conformal threshold gives a certified false-positive bound.

Module map (ARCHITECTURE.md §6); components in brackets:

    templates.py        [C2]  the ONLY prompt builder in the codebase
    model_loader.py     [C1]  4-bit backbone + 3 resident adapters
    scoring.py          [C4+C5] batched forward, label-logit extraction, bulk scorer
    tokenizer_stats.py  [C3]  4 surface features, on RAW text
    calibration.py      [C6]  per-adapter temperature scaling
    fusion.py           [C7]  build_feature_vector (the extension seam) + 3 fusers
    conformal.py        [C8]  split-conformal threshold
    pipeline.py         [C9]  end-to-end detect(prompt) over persisted artefacts
    eval/               §4    splits, metrics, bootstrap, baselines, external, run_eval
    serve/              [C10] FastAPI demonstration server

This package imports cleanly under both environments: the uv-managed local/serve env and
Kaggle/Colab's pre-provisioned torch+CUDA. Keep top-level imports here free of heavy
dependencies so `import src` stays cheap and never touches the GPU.
"""

__version__ = "0.1.0"
