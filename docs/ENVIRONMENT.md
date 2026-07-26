# ENVIRONMENT.md — dependency pins and their provenance

Companion to `ARCHITECTURE.md` §6 and `TASKS.md` P0.1. Every behaviour-critical pin in
`pyproject.toml` is justified here so the thesis environment table has a single source.

## Two environments, on purpose

| | Local / FastAPI serve | Kaggle / Colab (GPU) |
|---|---|---|
| Owner | `uv` | the platform |
| Install | `uv sync` (add `--extra serve` for Phase 9) | `pip install` of the pinned libs **on top of** platform torch |
| torch | resolved from the `pytorch-cu124` index | **platform's pre-provisioned build — never rebuilt** |
| Never do | — | `uv sync` (would rebuild the GPU stack and risk a CUDA mismatch) |

`torch` is deliberately left loose (`>=2.2.1`) so whatever CUDA build Kaggle ships satisfies
it. The behaviour-critical libraries are pinned exactly. `src/` must import cleanly under both.

The **P1.2 batched-vs-sequential gate is what verifies the runtime stack is correct**,
whichever platform it runs on — it is the cross-environment sanity check, not just a
correctness test.

## Pin provenance

Versions were read from the Unsloth banners saved in the reference notebooks' cell outputs —
i.e. what actually ran when the adapters were trained, not what was requested. Note that the
training notebooks used `pip install -U` (unpinned), so the banners are the only record.

| Package | Pin | Source |
|---|---|---|
| `unsloth` | `==2025.7.2` | Banner in **both** training notebooks |
| `transformers` | `==4.53.1` | Banner in `FT_Merged_Adapter.ipynb` — see discrepancy below |
| `torch` | `>=2.2.1` (loose) | Banner records 2.6.0+cu124; left loose for platform GPU builds |
| `peft` | `==0.16.0` | Working local environment |
| `trl` | `==0.19.1` | Working local environment |
| `bitsandbytes` | `==0.50.0` | Working local environment (`Final_Inference_Pipeline.ipynb` asked only for `>=0.46.1`) |
| `accelerate` | `==1.14.0` | Working local environment; named behaviour-critical by TASKS.md P0.1 |

## Known discrepancy — transformers 4.53.1 vs 4.54.1

The two training runs did **not** use the same transformers version:

| Notebook | Adapters produced | Transformers | Unsloth | Torch |
|---|---|---|---|---|
| `LoRA-Fine-Tuning.ipynb` | the 3 **specialists** (`slm-a/b/c-qlora`) | **4.54.1** | 2025.7.2 | 2.6.0+cu124 |
| `FT_Merged_Adapter.ipynb` | the **merged baseline** (`slm-merged-qlora`) | **4.53.1** | 2025.7.2 | 2.6.0+cu124 |

Both ran on an RTX 3060 Laptop (6 GB, Windows), not a T4.

A single pin cannot match both, so `TASKS.md` P0.1's acceptance criterion "pinned versions
match the working-notebook versions" is not literally satisfiable.

**Decision: standardise on `transformers==4.53.1`** — the version already declared in
`CLAUDE.md`, `TASKS.md`, the three Kaggle notebooks, and the working local environment.

Rationale, and why the risk is low:

- LoRA adapters are plain weight matrices; loading them is version-insensitive as long as the
  Gemma3 module names are unchanged, and 4.53 -> 4.54 is a minor bump.
- Stored `formatted_text` is literal text, so templating and the answer-strip path cannot be
  affected by the transformers version at all.
- Consistency matters more than which version: **all** scoring — specialists and baseline —
  runs under one version, which is what keeps the specialist-vs-generalist comparison
  controlled. Mixing versions across the two arms would be the actual threat to validity.
- **P1.2** (batched == sequential within fp16 tolerance) and **P1.3** (≥99% agreement with the
  legacy generate-parse pipeline) are the empirical checks that would surface a real numerical
  problem before any downstream artefact depends on it.

Report this in the thesis environment table rather than omitting it. It sits alongside the
other documented environment caveat — that `Evaluation.ipynb` loaded the backbone in fp16
while training used the 4-bit Unsloth checkpoint, which is why legacy metrics are not
comparable and this build standardises on 4-bit everywhere (`ARCHITECTURE.md` §0, §5.10).

## Kaggle install snippet

The cell used verbatim at the top of all three GPU notebooks:

```python
!pip install -q "transformers==4.53.1" "unsloth==2025.7.2" "peft==0.16.0" \
                "trl==0.19.1" "accelerate==1.14.0" "bitsandbytes==0.50.0"
```

If Kaggle's preinstalled versions clash, restart the kernel after the install and re-run from
the top. `src/` is made importable by `git clone` + `%cd`, not by installing this project, so
`requires-python = ">=3.12"` is never evaluated on Kaggle's Python 3.11.
