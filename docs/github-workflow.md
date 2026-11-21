# GitHub learning workflow

Mirek owns production code and design decisions. Copilot or Szpaner adds useful
offline tests and checks evidence; Szpaner maintains explanatory docs and ADRs.
No workflow merges a PR or treats a checked box as a human approval.

## Everyday flow

```mermaid
flowchart TD
    Issue["Small issue with a Done when checklist"] --> Branch["Create branch from the issue: number-title"]
    Branch --> Push["Mirek pushes an implementation commit"]
    Push --> PR["Learning PR: create draft or refresh checklist"]
    PR --> CI["CI: Linux and Windows, Python 3.12 and 3.13"]
    CI --> Config{"Agent user token configured?"}
    Config -->|"no"| Manual["Visible setup warning; manual Copilot request remains available"]
    Config -->|"yes, current human commit"| Agent["Copilot: useful offline tests + evidence assessment"]
    Agent --> Changes["Test-only commit and checklist evidence"]
    Changes --> Human["Mirek reviews changes and approves CI when GitHub requires it"]
    Human --> Review["Review and merge by a human"]
```

1. Use **Create branch** in an open task issue. Keep its leading number, e.g.
   `82-w02-d1-1-pin-haystack`. Work on one small task per PR, branching from
   current `main`; do not use the archive branch for ongoing work.
2. Creating the branch at the current base tip does not open an empty PR.
   The first owner push with a commit ahead of `main` creates a **draft PR**.
   The workflow copies the issue's `## Done when` checkboxes and links it with
   `Closes #N`. It does not write a claim that the implementation is complete.
3. Later owner pushes refresh that PR instead of creating another. Matching
   checked criteria and author text outside the managed block are preserved.
   New criteria start unchecked. Renaming a criterion requires reassessment.
4. CI runs offline Python tests on Linux and Windows with Python 3.12 and 3.13;
   lint/types and Node workflow contract tests run on Linux. Because a PR
   created by `GITHUB_TOKEN` does not trigger another PR workflow, creation
   explicitly dispatches CI on the task branch. Later human pushes use normal PR CI.
5. After CI completes (pass or fail), the agent dispatcher checks that the
   branch/PR head is still that run's commit, belongs to this repository and
   came from the owner. It requests at most one Copilot assessment per head SHA.
   Agent/bot runs do not request another agent, preventing a test-commit loop.
6. Copilot reads the issue, diff and `.github/agents/learning-tests.agent.md`.
   It adds focused tests when they help, reports failures without implementing
   the learner's solution, and updates the PR checklist with evidence. A
   documentation-only commit normally needs an explanation, not new tests.
7. Review the evidence, pull the agent's commits before further local work,
   and mark the draft ready when appropriate. GitHub can require **Approve and
   run workflows** after Copilot pushes. This is a GitHub safeguard; the
   dispatcher does not bypass it. Live experiments and milestone checks remain
   explicit, separate actions. Never infer their completion from mocked tests.

A rapid sequence of pushes can queue follow-up requests while Copilot is still
working. The agent must reassess the latest head and never force-push. Its
checklist assessment remains reviewable evidence, not a deterministic proof.

## One-time setup and current limits

These workflow files must be merged into the default branch before the flow
is active: privileged metadata jobs check out only default-branch scripts and
never execute task-branch code with the dispatch credential.

### Automatic draft PRs

Repository Settings → Actions → General → Workflow permissions must allow
**GitHub Actions to create and approve pull requests**. GitHub combines these
in one setting; this workflow creates PRs, never approval reviews. Keep the
default token read-only: individual jobs declare only their required permissions.
Branch names without a leading issue number, closed issues, PR numbers mistaken
for issues, other users and bot pushes are ignored.

### Copilot test dispatch

Copilot is available for this repository, but the dispatcher needs a GitHub
**user token** with Copilot access. `GITHUB_TOKEN` is not a supported identity
for starting cloud-agent tasks. Do not copy a broad local CLI token into Actions.

Create a fine-grained, expiring personal access token restricted to
`wiertmir/mini-rag`, with the permissions required by GitHub's agent-assignment
API: metadata read, and Actions, Contents, Issues and Pull requests read/write.
Store it directly in the repository's Actions secrets as **`LEARNING_AGENT_TOKEN`**:
[repository Actions secrets](https://github.com/wiertmir/mini-rag/settings/secrets/actions).
The workflow uses it only for GitHub API requests, never exposes it to the
learner branch, and never prints its value. Do not put it in an issue or chat.
Token expiry, Copilot policy/quota and failed assignment must be investigated
if the dispatcher stops; a successful comment alone is not proof the agent ran.

Until that secret is configured, automatic draft PRs and CI work independently;
the test-agent workflow emits an explicit setup warning and does not request work.
No claim of automatic test generation should be made before a live task is seen.

For manual use, assign Copilot to the PR and post a comment such as:

> @copilot Follow `.github/agents/learning-tests.agent.md`. Add meaningful offline
> tests for this commit against the linked issue, without implementing production
> code. Update the managed checklist with evidence and report unverified items.

This uses your signed-in GitHub identity and Copilot quota. It does not require
an Actions secret. Review the session log and resulting test diff.

## Archive and diagrams

The completed plain-Python implementation is preserved in branch
[`week-01-own-implementation`](https://github.com/wiertmir/mini-rag/tree/week-01-own-implementation)
and annotated tag
[`week-01-own-implementation-v1`](https://github.com/wiertmir/mini-rag/releases/tag/week-01-own-implementation-v1).
Both include the README diagrams added before creating either ref. The archive
tag should not move. Local ignored PDFs/indexes are not included; manifests,
configuration and saved result fingerprints describe the experiment.

## Official references

- [Starting and assigning Copilot tasks via the API](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/use-cloud-agent-via-the-api)
- [Copilot troubleshooting, PR mentions and approval of workflows](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/troubleshoot-cloud-agent)
- [Events triggered by GITHUB_TOKEN](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-when-your-workflow-runs/triggering-a-workflow#triggering-a-workflow-from-a-workflow)
- [Automatic token permissions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication)
