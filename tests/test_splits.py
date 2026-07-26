"""P0.3 — data-split protocol tests (ARCHITECTURE §3).

Two tiers. The synthetic tier exercises every gate deterministically, including the failure
paths that must never fire on real data (and so would otherwise be untested). The Hub tier runs
the real build and asserts the acceptance criteria on the deployed datasets.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.eval.splits import (
    MIN_C_SPLIT_BENIGN,
    SPLIT_SEED,
    SPLIT_SHARES,
    TEST_MANIFEST_COLUMNS,
    VAL_MANIFEST_COLUMNS,
    SplitError,
    build_unified_splits,
    dedup_within,
    drop_val_test_collisions,
    load_manifests,
    load_summary,
    pairwise_overlaps,
    partition_val,
    prompt_hash,
    prompt_set_hash,
)
from src.templates import ADAPTERS


def _frame(prompts, labels=None, category="role_violation"):
    df = pd.DataFrame(
        {
            "prompt_hash": [prompt_hash(p) for p in prompts],
            "category": category,
            "raw_prompt": list(prompts),
        }
    )
    if labels is not None:
        df["label"] = list(labels)
    return df


def _synthetic_val(n_per_stratum=100):
    """A balanced synthetic UNIFIED-VAL covering all six label x category strata."""
    rows = []
    for category in ADAPTERS:
        for label in (0, 1):
            for i in range(n_per_stratum):
                rows.append((f"{category}-{label}-{i}", label, category))
    return pd.DataFrame(
        {
            "prompt_hash": [prompt_hash(p) for p, _, _ in rows],
            "category": [c for _, _, c in rows],
            "raw_prompt": [p for p, _, _ in rows],
            "label": [lab for _, lab, _ in rows],
        }
    )


# --- hashing -----------------------------------------------------------------------------


def test_prompt_hash_is_exact_not_normalised():
    """'Exact text' means exact — whitespace and case must not collide."""
    assert prompt_hash("ignore all rules") != prompt_hash("Ignore all rules")
    assert prompt_hash("ignore all rules") != prompt_hash("ignore all rules ")
    assert prompt_hash("abc") == prompt_hash("abc")


def test_prompt_set_hash_is_order_independent():
    a = ["h3", "h1", "h2"]
    assert prompt_set_hash(a) == prompt_set_hash(sorted(a))
    assert prompt_set_hash(a) != prompt_set_hash(["h1", "h2"])


# --- dedup -------------------------------------------------------------------------------


def test_dedup_within_drops_exact_duplicates_and_counts_them():
    df = _frame(["a", "b", "a", "c", "b"], labels=[0, 1, 0, 1, 1])
    out, dropped = dedup_within(df)
    assert dropped == 2
    assert list(out["raw_prompt"]) == ["a", "b", "c"]


def test_val_test_collisions_are_dropped_from_val_only():
    val = _frame(["keep", "collide"], labels=[0, 1])
    test_hashes = {prompt_hash("collide")}
    out, dropped = drop_val_test_collisions(val, test_hashes)
    assert dropped == 1
    assert list(out["raw_prompt"]) == ["keep"]


def test_dedup_key_is_raw_prompt_not_formatted_text():
    """The same prompt from two categories must collide — that is pitfall 11.

    Hashing `formatted_text` would miss it, since each adapter wraps the prompt differently.
    """
    a = _frame(["shared benign prompt"], labels=[0], category="role_violation")
    b = _frame(["shared benign prompt"], labels=[0], category="privilege_escalation")
    union = pd.concat([a, b], ignore_index=True)
    _, dropped = dedup_within(union)
    assert dropped == 1


# --- partition ---------------------------------------------------------------------------


def test_partition_produces_three_disjoint_splits_covering_every_row():
    val = _synthetic_val()
    out = partition_val(val, seed=SPLIT_SEED)
    assert len(out) == len(val)
    assert set(out["split"]) == {"T", "F", "C"}

    sets = {name: set(part["prompt_hash"]) for name, part in out.groupby("split")}
    assert pairwise_overlaps(sets) == {"C|F": 0, "C|T": 0, "F|T": 0}
    assert set().union(*sets.values()) == set(val["prompt_hash"])


def test_partition_respects_the_configured_shares():
    val = _synthetic_val(n_per_stratum=1000)
    out = partition_val(val, seed=SPLIT_SEED)
    for name, share in SPLIT_SHARES.items():
        actual = (out["split"] == name).mean()
        assert abs(actual - share) < 0.01, f"{name}: {actual:.4f} vs {share}"


def test_partition_is_stratified_by_label_and_category():
    """Every label x category stratum must appear in all three splits at the right share."""
    val = _synthetic_val(n_per_stratum=200)
    out = partition_val(val, seed=SPLIT_SEED)
    for (label, category), group in out.groupby(["label", "category"]):
        for name, share in SPLIT_SHARES.items():
            actual = (group["split"] == name).mean()
            assert abs(actual - share) < 0.02, f"{label}/{category}/{name}: {actual:.4f}"


def test_partition_is_deterministic_under_the_fixed_seed():
    val = _synthetic_val()
    first = partition_val(val, seed=SPLIT_SEED).sort_values("prompt_hash")
    second = partition_val(val, seed=SPLIT_SEED).sort_values("prompt_hash")
    assert list(first["split"]) == list(second["split"])


def test_a_different_seed_produces_a_different_partition():
    val = _synthetic_val()
    a = partition_val(val, seed=SPLIT_SEED).sort_values("prompt_hash")
    b = partition_val(val, seed=SPLIT_SEED + 1).sort_values("prompt_hash")
    assert list(a["split"]) != list(b["split"])


# --- overlap arithmetic ------------------------------------------------------------------


def test_pairwise_overlaps_detects_a_planted_collision():
    sets = {"T": {"a", "b"}, "F": {"c"}, "C": {"d"}, "TEST": {"b"}}
    out = pairwise_overlaps(sets)
    assert out["T|TEST"] == 1
    assert sum(out.values()) == 1


def test_pairwise_overlaps_covers_every_unordered_pair():
    out = pairwise_overlaps({"T": set(), "F": set(), "C": set(), "TEST": set()})
    assert set(out) == {"C|F", "C|T", "C|TEST", "F|T", "F|TEST", "T|TEST"}


# --- hard gates (failure paths) ----------------------------------------------------------


@pytest.fixture
def _tiny_hub(monkeypatch):
    """Patch the Hub loader so the whole build runs offline on controlled rows."""

    def fake(repo, split, token):
        category = {v: k for k, v in __import__(
            "src.templates", fromlist=["DATASET_REPOS"]
        ).DATASET_REPOS.items()}[repo]
        n = 300 if split == "validation" else 100
        return {
            "formatted_text": [
                _formatted(f"{category}-{split}-{i}", category) for i in range(n)
            ],
            "label": [i % 2 for i in range(n)],
        }

    monkeypatch.setattr("src.eval.splits._load_split", fake)
    return fake


def _formatted(prompt, category):
    from src.templates import build_formatted_text

    return build_formatted_text(prompt, category, 0)


def test_c_split_benign_gate_hard_fails(monkeypatch, tmp_path):
    """Below MIN_C_SPLIT_BENIGN the run must raise, not warn."""

    def fake(repo, split, token):
        from src.templates import DATASET_REPOS, build_formatted_text

        category = {v: k for k, v in DATASET_REPOS.items()}[repo]
        n = 60 if split == "validation" else 10
        # All-injection: zero benign anywhere, so the C-split gate must trip.
        return {
            "formatted_text": [
                build_formatted_text(f"{category}-{split}-{i}", category, 1) for i in range(n)
            ],
            "label": [1] * n,
        }

    monkeypatch.setattr("src.eval.splits._load_split", fake)
    with pytest.raises(SplitError, match="below the hard minimum"):
        build_unified_splits(out_dir=tmp_path, token=None)


def test_overlap_gate_hard_fails_when_val_and_test_share_a_prompt(monkeypatch, tmp_path):
    """Bypass the dedup step and confirm the overlap gate still catches the collision."""
    monkeypatch.setattr(
        "src.eval.splits.drop_val_test_collisions", lambda val, hashes: (val, 0)
    )

    def fake(repo, split, token):
        from src.templates import DATASET_REPOS, build_formatted_text

        category = {v: k for k, v in DATASET_REPOS.items()}[repo]
        # Identical prompt pool for validation and test => guaranteed collisions.
        return {
            "formatted_text": [
                build_formatted_text(f"{category}-shared-{i}", category, i % 2)
                for i in range(400)
            ],
            "label": [i % 2 for i in range(400)],
        }

    monkeypatch.setattr("src.eval.splits._load_split", fake)
    with pytest.raises(SplitError, match="non-zero pairwise overlap"):
        build_unified_splits(out_dir=tmp_path, token=None)


# --- offline end-to-end ------------------------------------------------------------------


def test_end_to_end_offline_writes_all_manifests(_tiny_hub, tmp_path):
    summary = build_unified_splits(out_dir=tmp_path, token=None)

    for name in ("T", "F", "C"):
        assert (tmp_path / f"unified_val_{name}.parquet").exists()
    assert (tmp_path / "unified_test.parquet").exists()
    assert (tmp_path / "split_summary.json").exists()

    frames = load_manifests(tmp_path)
    assert list(frames["T"].columns) == VAL_MANIFEST_COLUMNS
    assert list(frames["TEST"].columns) == TEST_MANIFEST_COLUMNS
    assert summary.pairwise_overlap == {k: 0 for k in summary.pairwise_overlap}


def test_test_manifest_carries_no_label_and_no_text(_tiny_hub, tmp_path):
    """The seal is structural: nothing downstream can score or peek at UNIFIED-TEST."""
    build_unified_splits(out_dir=tmp_path, token=None)
    test = load_manifests(tmp_path)["TEST"]
    assert "label" not in test.columns
    assert "raw_prompt" not in test.columns
    assert list(test.columns) == ["prompt_hash", "category"]


def test_summary_round_trips_through_json(_tiny_hub, tmp_path):
    written = build_unified_splits(out_dir=tmp_path, token=None)
    assert load_summary(tmp_path) == written


def test_collision_counts_are_recorded_in_the_summary(_tiny_hub, tmp_path):
    """AC: 'collision counts logged' — they must land in the persisted artefact, not just stderr."""
    build_unified_splits(out_dir=tmp_path, token=None)
    payload = json.loads((tmp_path / "split_summary.json").read_text(encoding="utf-8"))
    for key in (
        "val_internal_collisions",
        "test_internal_collisions",
        "val_test_collisions_dropped",
        "pairwise_overlap",
    ):
        assert key in payload


def test_load_manifests_errors_clearly_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="run build_unified_splits first"):
        load_manifests(tmp_path)


# --- Hub tier (the real acceptance criteria) ---------------------------------------------


from tests.conftest import load_hf_token as _load_token

requires_hub = pytest.mark.skipif(
    _load_token() is None,
    reason="HF_TOKEN not available; the P0.3 acceptance criteria need the private Hub datasets",
)


@pytest.fixture(scope="session")
def real_build(tmp_path_factory):
    """One real build, shared by the acceptance tests below."""
    out = tmp_path_factory.mktemp("splits")
    summary = build_unified_splits(out_dir=out, token=_load_token())
    return out, summary


@requires_hub
def test_ac_pairwise_overlap_is_zero(real_build):
    """AC 1: pairwise overlap between T/F/C/UNIFIED-TEST = 0 exact-hash collisions."""
    _, summary = real_build
    assert summary.pairwise_overlap == {k: 0 for k in summary.pairwise_overlap}
    assert len(summary.pairwise_overlap) == 6


@requires_hub
def test_ac_c_split_has_enough_benign(real_build):
    """AC 2: C-split contains >=100 benign examples (hard fail otherwise)."""
    _, summary = real_build
    assert summary.c_split_benign >= MIN_C_SPLIT_BENIGN


@requires_hub
def test_ac_manifests_persisted(real_build):
    """AC 3: manifests persisted."""
    out, summary = real_build
    for name in ("T", "F", "C"):
        path = out / f"unified_val_{name}.parquet"
        assert path.exists() and path.stat().st_size > 0
        assert len(pd.read_parquet(path)) == summary.split_sizes[name]
    assert (out / "unified_test.parquet").exists()
    assert (out / "split_summary.json").exists()


@requires_hub
def test_ac_collision_counts_logged(real_build):
    """AC 4: collision counts logged."""
    _, summary = real_build
    assert isinstance(summary.val_internal_collisions, int)
    assert isinstance(summary.test_internal_collisions, int)
    assert isinstance(summary.val_test_collisions_dropped, int)


@requires_hub
def test_real_splits_are_disjoint_by_construction(real_build):
    """Belt-and-braces: recompute overlap from the written manifests, not the summary."""
    out, _ = real_build
    frames = load_manifests(out)
    sets = {name: set(frame["prompt_hash"]) for name, frame in frames.items()}
    assert pairwise_overlaps(sets) == {k: 0 for k in pairwise_overlaps(sets)}


@requires_hub
def test_no_val_prompt_survives_in_test(real_build):
    """The VAL/TEST boundary is the one that would silently inflate Phase 7 numbers."""
    out, _ = real_build
    frames = load_manifests(out)
    val_hashes = set().union(*(set(frames[n]["prompt_hash"]) for n in ("T", "F", "C")))
    assert not (val_hashes & set(frames["TEST"]["prompt_hash"]))
