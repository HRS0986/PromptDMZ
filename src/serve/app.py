"""C10 — FastAPI demonstration server.

A THIN wrapper over `src.pipeline.detect()` with no detection logic of its own. That is what
keeps the demo faithful to the evaluated system — if the API could compute a verdict its own
way, it could drift from the thesis results.

    POST /detect         single prompt
    POST /detect_batch   list of prompts
    GET  /health         model + artefacts loaded
    GET  /               minimal HTML form for the live demo

Constraints: the model loads exactly once at startup onto `app.state`, never per request;
`torch.inference_mode()` throughout; SINGLE worker — one model in VRAM, never fork GPU
workers. A `--cpu` fallback flag makes the API demonstrable on a laptop without a GPU (slower,
identical outputs).

Not a hardened production service — local/CORS demo use, and documented as such.

    uv run uvicorn src.serve.app:app --reload

Implemented by P9.2. Requires the `serve` extra.
"""
