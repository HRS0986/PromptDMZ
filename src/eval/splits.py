"""Data-split protocol (ARCHITECTURE §3). Violations here invalidate every downstream result.

Source material is the three Hub datasets' EXISTING 80/10/10 stratified splits. Never
re-split.

    UNIFIED-VAL  = union of the three validation splits, source-category tagged
    UNIFIED-TEST = union of the three test splits — SEALED until Phase 7, used exactly once

Because benign pools were drawn from shared sources across the three datasets, cross-dataset
collisions are likely: exact-text dedup within each union AND across the VAL/TEST boundary is
mandatory, not optional (§5 pitfall 11). Drop colliders from VAL and log the counts.

UNIFIED-VAL is partitioned (stratified by label × source category, fixed seed) into three
disjoint parts, consumed in this rigid order — C6 -> C7 -> C8:

    T-split  30%   temperature fit                              (C6)
    F-split  40%   fusion training + scaler + rare-token table  (C7, C3)
    C-split  30%   conformal calibration, ≥100-200 benign       (C8)

Hard gates (P0.3): pairwise overlap between T/F/C/UNIFIED-TEST must be 0 exact-hash
collisions, and the C-split must contain ≥100 benign examples or the run fails.

Every output file is labelled with split name + prompt-set hash. That is what makes the
previously-observed failure mode — identical val and test metrics from a duplicated run
(§5 pitfall 7) — structurally impossible rather than merely unlikely.

How the TEST seal survives this module
--------------------------------------
The zero-overlap gate cannot be checked without reading UNIFIED-TEST, and ARCHITECTURE §3
mandates the check. The seal is preserved structurally rather than by discipline: the TEST
manifest written here carries **only `prompt_hash` and `category`** — no labels, no text.
Nothing downstream can compute a metric, fit a parameter, or inspect a distribution from it.
Phase 7 re-pulls the real rows from the Hub and joins on `prompt_hash`. Colliders are dropped
from VAL only; UNIFIED-TEST is never modified.

Why the dedup key is the RAW prompt
-----------------------------------
The same prompt wrapped by two different adapters produces two different `formatted_text`
strings, so hashing `formatted_text` would report zero collisions while the underlying prompts
overlap — precisely the failure §5 pitfall 11 warns about. Keys are therefore SHA-256 over the
raw prompt recovered by `templates.extract_raw_prompt()`: exact text, no normalisation.

Implemented by P0.3. Notebook import surface: `build_unified_splits`, `load_manifests`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.templates import (
    ADAPTERS,
    DATASET_REPOS,
    TEMPLATE_VERSION,
    extract_raw_prompt,
)

log = logging.getLogger(__name__)

# Fixed for the whole build. Changing either invalidates every fitted artefact downstream.
SPLIT_SEED = 42
SPLIT_SHARES: dict[str, float] = {"T": 0.30, "F": 0.40, "C": 0.30}

# ARCHITECTURE §2 C8: at α=0.01 the bare minimum is 1/α = 100 benign. Hard gate.
MIN_C_SPLIT_BENIGN = 100

VAL_SPLIT_NAME = "validation"
TEST_SPLIT_NAME = "test"

DEFAULT_OUT_DIR = Path("artifacts/splits")

# Manifest columns. TEST deliberately carries neither label nor text — see module docstring.
VAL_MANIFEST_COLUMNS = ["prompt_hash", "split", "label", "category", "raw_prompt"]
TEST_MANIFEST_COLUMNS = ["prompt_hash", "category"]


class SplitError(RuntimeError):
    """Raised when a hard gate of the split protocol fails."""


@dataclass
class SplitSummary:
    """Machine-readable record of one `build_unified_splits` run."""

    seed: int
    template_version: str
    shares: dict[str, float]
    rows_per_source: dict[str, dict[str, int]]
    val_rows_before_dedup: int
    val_rows_after_dedup: int
    val_internal_collisions: int
    test_rows_before_dedup: int
    test_rows_after_dedup: int
    test_internal_collisions: int
    val_test_collisions_dropped: int
    split_sizes: dict[str, int]
    split_benign_counts: dict[str, int]
    split_benign_fraction: dict[str, float]
    pairwise_overlap: dict[str, int]
    c_split_benign: int
    manifest_hashes: dict[str, str]


def prompt_hash(raw_prompt: str) -> str:
    """SHA-256 of the exact raw prompt text. No normalisation — "exact text" means exact."""
    return hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()


def prompt_set_hash(hashes) -> str:
    """Order-independent fingerprint of a set of prompt hashes.

    Goes into every manifest so a downstream artefact can prove which prompt set it was fitted
    on. This is the mechanism that makes §5 pitfall 7 (identical val/test metrics from a
    duplicated run) detectable rather than silent.
    """
    return hashlib.sha256("".join(sorted(hashes)).encode("utf-8")).hexdigest()


def _load_split(repo: str, split: str, token: str | None):
    from datasets import load_dataset

    return load_dataset(repo, split=split, token=token)


def build_union(split: str, token: str | None = None, *, with_labels: bool = True) -> pd.DataFrame:
    """Union the three per-category datasets' `split`, tagging each row with its source category.

    `with_labels=False` is used for UNIFIED-TEST: labels are never read, so the sealed split
    cannot leak into anything this module writes.
    """
    frames = []
    for category in ADAPTERS:
        repo = DATASET_REPOS[category]
        ds = _load_split(repo, split, token)
        raw = [extract_raw_prompt(t) for t in ds["formatted_text"]]
        frame = pd.DataFrame(
            {
                "prompt_hash": [prompt_hash(r) for r in raw],
                "category": category,
                "raw_prompt": raw,
            }
        )
        if with_labels:
            frame["label"] = list(ds["label"])
        frames.append(frame)
        log.info("loaded %s[%s]: %d rows", repo, split, len(frame))

    return pd.concat(frames, ignore_index=True)


def dedup_within(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact-text duplicates inside one union, keeping the first occurrence."""
    before = len(df)
    out = df.drop_duplicates(subset="prompt_hash", keep="first").reset_index(drop=True)
    return out, before - len(out)


def drop_val_test_collisions(val: pd.DataFrame, test_hashes: set[str]) -> tuple[pd.DataFrame, int]:
    """Remove VAL rows whose prompt also appears in UNIFIED-TEST.

    Colliders leave VAL, never TEST: the sealed set must stay exactly as the Hub defines it, and
    shrinking it would silently change the Phase 7 denominator.
    """
    before = len(val)
    out = val[~val["prompt_hash"].isin(test_hashes)].reset_index(drop=True)
    return out, before - len(out)


def partition_val(val: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Stratified partition of UNIFIED-VAL into T/F/C, by label × source category.

    Stratifying on the interaction (not on label alone) keeps each split's category mix
    representative, which matters because per-category recall and the cross-category leakage
    matrix are reported per split in §4.3.
    """
    assigned = []
    for (label, category), group in val.groupby(["label", "category"], sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_t = int(round(n * SPLIT_SHARES["T"]))
        n_f = int(round(n * SPLIT_SHARES["F"]))
        # C takes the remainder so the three parts always sum to n exactly.
        shuffled["split"] = ["T"] * n_t + ["F"] * n_f + ["C"] * (n - n_t - n_f)
        assigned.append(shuffled)
        log.info(
            "stratum label=%s category=%s: n=%d -> T=%d F=%d C=%d",
            label,
            category,
            n,
            n_t,
            n_f,
            n - n_t - n_f,
        )

    return pd.concat(assigned, ignore_index=True)


def pairwise_overlaps(sets: dict[str, set[str]]) -> dict[str, int]:
    """Exact-hash intersection size for every unordered pair. All must be 0."""
    names = sorted(sets)
    return {
        f"{a}|{b}": len(sets[a] & sets[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def build_unified_splits(
    out_dir: Path | str = DEFAULT_OUT_DIR,
    seed: int = SPLIT_SEED,
    token: str | None = None,
) -> SplitSummary:
    """Build UNIFIED-VAL/TEST, dedup, partition into T/F/C, and persist manifests.

    Raises `SplitError` if either hard gate fails: any non-zero pairwise overlap, or fewer than
    `MIN_C_SPLIT_BENIGN` benign rows in the C-split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # UNIFIED-TEST first, labels deliberately not read — see module docstring.
    test_raw = build_union(TEST_SPLIT_NAME, token, with_labels=False)
    test, test_dupes = dedup_within(test_raw)
    test_hashes = set(test["prompt_hash"])

    val_raw = build_union(VAL_SPLIT_NAME, token, with_labels=True)
    val_deduped, val_dupes = dedup_within(val_raw)
    val, cross_dropped = drop_val_test_collisions(val_deduped, test_hashes)
    log.info(
        "dedup: VAL %d -> %d (internal %d, val/test %d); TEST %d -> %d (internal %d)",
        len(val_raw),
        len(val),
        val_dupes,
        cross_dropped,
        len(test_raw),
        len(test),
        test_dupes,
    )

    partitioned = partition_val(val, seed=seed)

    # str() throughout: pandas types groupby keys as Scalar, but these are split names.
    by_split = {str(name): part for name, part in partitioned.groupby("split")}
    split_sets = {name: set(part["prompt_hash"]) for name, part in by_split.items()}
    split_sets["TEST"] = test_hashes
    overlaps = pairwise_overlaps(split_sets)

    offenders = {pair: n for pair, n in overlaps.items() if n != 0}
    if offenders:
        raise SplitError(f"non-zero pairwise overlap between splits: {offenders}")

    c_benign = int((partitioned[partitioned["split"] == "C"]["label"] == 0).sum())
    if c_benign < MIN_C_SPLIT_BENIGN:
        raise SplitError(
            f"C-split has {c_benign} benign rows, below the hard minimum "
            f"{MIN_C_SPLIT_BENIGN} required for a conformal bound at α=0.01"
        )

    manifest_hashes = {}
    for name, part in by_split.items():
        path = out_dir / f"unified_val_{name}.parquet"
        part[VAL_MANIFEST_COLUMNS].to_parquet(path, index=False)
        manifest_hashes[name] = prompt_set_hash(part["prompt_hash"])
        log.info("wrote %s (%d rows)", path, len(part))

    test_path = out_dir / "unified_test.parquet"
    test[TEST_MANIFEST_COLUMNS].to_parquet(test_path, index=False)
    manifest_hashes["TEST"] = prompt_set_hash(test["prompt_hash"])
    log.info("wrote %s (%d rows, hashes+category only — sealed)", test_path, len(test))

    summary = SplitSummary(
        seed=seed,
        template_version=TEMPLATE_VERSION,
        shares=SPLIT_SHARES,
        rows_per_source={
            "val": {str(k): int(v) for k, v in val_raw.groupby("category").size().items()},
            "test": {str(k): int(v) for k, v in test_raw.groupby("category").size().items()},
        },
        val_rows_before_dedup=len(val_raw),
        val_rows_after_dedup=len(val),
        val_internal_collisions=val_dupes,
        test_rows_before_dedup=len(test_raw),
        test_rows_after_dedup=len(test),
        test_internal_collisions=test_dupes,
        val_test_collisions_dropped=cross_dropped,
        split_sizes={name: len(part) for name, part in by_split.items()},
        split_benign_counts={
            name: int((part["label"] == 0).sum()) for name, part in by_split.items()
        },
        split_benign_fraction={
            name: round(float((part["label"] == 0).mean()), 4)
            for name, part in by_split.items()
        },
        pairwise_overlap=overlaps,
        c_split_benign=c_benign,
        manifest_hashes=manifest_hashes,
    )

    (out_dir / "split_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("wrote %s", out_dir / "split_summary.json")
    return summary


def load_manifests(out_dir: Path | str = DEFAULT_OUT_DIR) -> dict[str, pd.DataFrame]:
    """Read back the persisted manifests. Read-only; fits nothing.

    Returns keys `T`, `F`, `C`, `TEST`. The `TEST` frame carries hashes and category only.
    """
    out_dir = Path(out_dir)
    frames = {}
    for name in ("T", "F", "C"):
        path = out_dir / f"unified_val_{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing manifest {path}; run build_unified_splits first")
        frames[name] = pd.read_parquet(path)

    test_path = out_dir / "unified_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"missing manifest {test_path}; run build_unified_splits first")
    frames["TEST"] = pd.read_parquet(test_path)
    return frames


def load_summary(out_dir: Path | str = DEFAULT_OUT_DIR) -> SplitSummary:
    """Read back the run summary written alongside the manifests."""
    path = Path(out_dir) / "split_summary.json"
    return SplitSummary(**json.loads(path.read_text(encoding="utf-8")))
