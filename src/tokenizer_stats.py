"""C3 — Tokenizer statistics (CPU side-channel).

Four features, computed on the RAW user prompt — pre-template, so template tokens can never
pollute the statistics (ARCHITECTURE §5 pitfall 6):

    1. fertility            = num_tokens / max(1, num_chars)
    2. byte_fallback_rate   = fraction of byte-fallback tokens (Gemma SentencePiece <0xNN>)
    3. rare_token_fraction  = fraction of tokens outside the top-K frequency table
                              (K default 20_000, fitted on the F-split BENIGN portion only)
    4. token_length_entropy = Shannon entropy of token string-length distribution

Rationale: obfuscated text (encodings, homoglyphs, TokenBreak-style perturbations) distorts
segmentation shape even when semantics are hidden — a signal family orthogonal to the
adapters' semantic evidence, at near-zero CPU cost.

The frequency table and the scaler are fitted on F-split ONLY.

Implemented by P3.1.
"""
