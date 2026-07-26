"""Shared test fixtures.

The HF token loader lives here rather than in each test module. A per-module copy already
diverged once: `test_splits.py` omitted the `.env` fallback and its Hub-backed acceptance tests
skipped silently while `test_templates.py`'s ran — a skipped gate looks identical to a passing
one in a `-q` summary.
"""

from __future__ import annotations

import os


def load_hf_token() -> str | None:
    """Return the Hugging Face token from the environment, falling back to `.env`."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from dotenv import dotenv_values
    except ImportError:
        return None
    return dotenv_values(".env").get("HF_TOKEN")
