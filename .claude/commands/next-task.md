---
description: Start the next task in docs/TASKS.md, respecting phase gates
---

Work on the next task in `docs/TASKS.md`, following the rules in `CLAUDE.md` and the specs in `docs/ARCHITECTURE.md`.

Follow this procedure exactly:

1. **Find the next task.** Read `docs/TASKS.md` and identify the first unchecked `- [ ]` task in the lowest-numbered phase that still has unchecked tasks. State which task ID you are starting (e.g. "Starting P1.2") and quote its one-line description.

2. **Check the phase gate.** If this task is the first task of a new phase, confirm the previous phase is fully complete (all its tasks checked). If it is not, STOP and tell me which earlier tasks remain — do not skip ahead.

3. **Restate the acceptance criteria** for this task verbatim from `docs/TASKS.md`, so we both know the definition of done before you write code.

4. **Confirm scope.** Briefly state what files you will create or edit and what you will NOT touch. If the task depends on an artefact from a previous task that doesn't exist yet, stop and say so.

5. **Wait for my go-ahead** on the plan before implementing, UNLESS the task is trivial and self-contained. For anything touching the scoring path, the data splits, the test set, or model loading, always wait.

6. **Implement**, writing the test that proves each acceptance criterion alongside the code.

7. **Verify.** Run the tests. Report each acceptance criterion as met or not-met with the evidence (test output, printed numbers, artefact paths).

Do NOT check the box in `docs/TASKS.md` yourself — I do that after reviewing. Do NOT start the following task in the same turn.

Reminder of the non-negotiables: `src/templates.py` is the only prompt builder; no label leakage; the test split stays sealed until Phase 7; batched must equal sequential (P1.2) before downstream work is trusted; tokenizer stats on raw text; never retrain the baseline adapter.
