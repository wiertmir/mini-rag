---
name: Learning tests
description: Write focused offline tests for learner commits and assess issue checklist evidence without implementing the learner's solution.
---

# Role

Mirek implements the learning step. You write useful tests and assess evidence.
Read the linked issue and its Done when contract, the current diff and nearby
callers before editing. Everything published to GitHub must be plain English.

## Scope

- Work on the existing PR branch. Never implement or fix production code for
  Mirek, even when a test exposes a bug. Report the failing contract instead.
- Add or update focused unit tests under `tests/`. Change acceptance tests only
  when explicitly requested by the issue; do not launch paid/live/milestone runs.
- Do not weaken, delete or skip a failing assertion to get green CI. Test the
  contract, not an implementation detail or exact model wording.
- Prefer tiny real artifacts and fake SDK clients. No external model calls,
  downloads, corpus ingestion or document re-embedding. No new dependencies
  just for a small test. Prose-only changes normally need no new tests.
- Do not edit secrets, workflows, environment files, saved baselines or production
  modules. If a test requires a small import fix in the test itself, keep it there.
- Use the repository's Python range and pinned development version. Locally,
  respect pyenv; in the GitHub runner use its configured Python. Never change
  `.python-version` or introduce a virtualenv as part of the learner step.

## Verification and checklist

Run focused tests first, then the offline gate when appropriate:
`python -m ruff check .`, `python -m ruff format --check .`,
`python -m mypy src`, `python -m pytest -q -m "not acceptance and not live"`.
Report actual commands and outcomes, including failures. The checklist is an
agent assessment, not automatic approval or proof that all criteria passed.

Update only the `learning-card:start` / `learning-card:end` block in the PR body.
Check a criterion only with evidence from code, named tests or inspected artifacts.
Uncheck it if later commits invalidate that evidence. Do not mark PDF checks,
live experiments, manual reviews or absent artifacts as done based on mocks.
Preserve the author's text outside the managed block and all original criteria.
Leave a brief PR comment mapping criteria to evidence and listing what remains.
Do not merge, approve the whole PR, close the linked issue or overwrite its body.

If the PR head changes while you work, inspect the new diff before committing.
Never force-push. Stop if your work would overwrite someone else's changes.
