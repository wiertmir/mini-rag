# mini-rag contribution rules

- GitHub prose, comments and commit messages must be plain English.
- This is a learning project: Mirek writes production implementations. For
  automatic learning-test requests, follow `.github/agents/learning-tests.agent.md`:
  add focused offline tests and report gaps without writing the solution.
- Preserve the `Answer`, citation and refusal contracts in the issue and README.
  Source document IDs and physical page ranges come from chunk metadata.
- Never call live model APIs, re-embed documents, download PDFs or overwrite saved
  evaluation results for a normal unit-test task.
- Use Python 3.12/3.13; the local pyenv pin is in `.python-version`. No new venv.
- Run meaningful focused tests and the existing lint/type/offline checks.
  Do not weaken tests or silently change the issue contract to make code pass.
- No auto-merge. A checked task is an evidence-backed assessment, not approval.
