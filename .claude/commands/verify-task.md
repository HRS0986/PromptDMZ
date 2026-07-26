---
description: Verify the current task against its acceptance criteria before it is checked off
argument-hint: [task-id, e.g. P1.2]
---

Verify task **$ARGUMENTS** against its acceptance criteria in `docs/TASKS.md` before I mark it done.

1. Quote the task's acceptance criteria verbatim.
2. For each criterion, run the relevant test or command and show the actual output.
3. Give a clear PASS / FAIL per criterion — no assumptions, only evidence.
4. Confirm the required artefacts exist at their expected paths (and, if applicable, were pushed to the artefact repo).
5. If everything passes, say so plainly and remind me to check the box myself. If anything fails, list exactly what remains.

If $ARGUMENTS is empty, verify the most recently worked-on task.

Do not edit `docs/TASKS.md`. Do not proceed to the next task.
