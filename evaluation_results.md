# Week 1 evaluation results

Reviewed on 2026-11-20, experiment `9c8e117`. [PR #74](https://github.com/wiertmir/mini-rag/pull/74), [week review #21](https://github.com/wiertmir/mini-rag/issues/21).

## Recommendation

For further local work, I would choose **Gemma 4 12B Q8 with an 8K context**.
It scored 25/28 expected-document citation hits with both embedding models.
Larger answer models did not improve that score and took longer to respond.
Qwen 4B is a reasonable, lighter starting point for embeddings.
Keep OpenAI (`text-embedding-3-small` + `gpt-5.4-mini`) as the reference:
the same 25/28 score, averaging 2.66 seconds per question.

**25/28 does not mean 25 correct answers.** The metric checks whether an answer
cites at least one expected document. A good answer from another chapter can
score zero, while an answer based only on bibliography can earn a point.
I would improve evidence evaluation before comparing more models.

## Answer results

I checked seven runs containing 210 records. Each run includes 28 questions
answerable from the corpus and two outside it (Q29 and Q30).
“Citation hits” counts expected-document citations as described above.
“Answerable refusals” counts refusals only for the first 28 questions.
Execution errors remain in the denominator of 28.

| Embeddings | Answer model / records | Citation hits | Answerable refusals | Errors | Mean s | Median s | Max s |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding:4b | [gemma4:12b-q8-8k](eval/results/qwen3-embedding_4b/gemma4_12b-q8-8k/baseline.jsonl) | 25/28 | 2 | 0 | 35.99 | 34.56 | 75.80 |
| qwen3-embedding:4b | [gemma4:31b](eval/results/qwen3-embedding_4b/gemma4_31b/baseline.jsonl) | 25/28 | 2 | 0 | 81.36 | 78.15 | 134.99 |
| qwen3-embedding:4b | [qwen3.6:35b](eval/results/qwen3-embedding_4b/qwen3.6_35b/baseline.jsonl) | 24/28 | 2 | 0 | 41.27 | 41.16 | 65.72 |
| qwen3-embedding:8b | [gemma4:12b-q8-8k](eval/results/qwen3-embedding_8b/gemma4_12b-q8-8k/baseline.jsonl) | 25/28 | 1 | 0 | 34.04 | 31.89 | 72.30 |
| qwen3-embedding:8b | [gemma4:31b](eval/results/qwen3-embedding_8b/gemma4_31b/baseline.jsonl) | 22/28 | 2 | 0 | 90.93 | 89.09 | 158.87 |
| qwen3-embedding:8b | [qwen3.6:35b](eval/results/qwen3-embedding_8b/qwen3.6_35b/baseline.jsonl) | 23/28 | 2 | 1 | 64.61 | 58.67 | 118.42 |
| text-embedding-3-small | [gpt-5.4-mini](eval/results/text-embedding-3-small/gpt-5.4-mini/baseline.jsonl) | 25/28 | 0 | 0 | 2.66 | 2.55 | 4.19 |

All models correctly refused Q29 and Q30 without citations.
There were no `invalid_citation` errors. One execution error occurred:
Q08 with Qwen 8B + Qwen 35B returned `ValidationError`. The loop continued
and saved all 30 records. The saved data cannot tell whether the error arose
while parsing the SDK response or during later validation.

Gemma 31B's mean response time was about 2.26 times that of 12B with 4B
embeddings and 2.67 times with 8B. With scores of 25/28 and 22/28, I see no
reason to choose the larger Gemma here. Qwen 35B scored 24/28 and 23/28.
The 8B embeddings improved document retrieval but not Gemma 12B's answer score.

Times cover the whole question: embedding, scheduling/model loading,
generation and post-processing. They do not measure the LLM alone.
On a 16 GB GPU, models may need reloading or RAM offloading.
During a separate setup check, I observed model swapping with concurrent Qwen
requests; we cannot confirm that it happened in every measured run.
Token costs and energy use were not recorded. Local models still offer
privacy, offline operation and no per-request charges.

## Did retrieval find useful sources?

I also checked three retrieval-only runs, containing 90 records.
`Document hit@5` means an expected document appeared among the five results.

| Embeddings | Document hit@5 | Saved document misses |
|---|---:|---|
| text-embedding-3-small | 27/28 | Q01 |
| qwen3-embedding:4b | 26/28 | Q01, Q21 |
| qwen3-embedding:8b | 28/28 | None |

Finding the document is not enough. Examples:

- Q18 / 4B: four of the five passages are bibliography. The fifth introduces
  aviation but does not provide the full road-versus-air energy-density
  comparison in the draft reference. A refusal need not be a model error.
- Q21 / 8B: the hit `WGIII_CH12_p043_c001` is a table/source-reference fragment,
  without an LCA explanation. Other results also include bibliography.
  All local models refused to answer.
- Q25 / 8B + Gemma 12B: all five passages are bibliography. The model built an
  explanation from titles about finance and de-risking. It earned a citation
  point, although the titles do not support the full explanation. The larger
  models' refusals are not evidence of lower quality here.

See [#76](https://github.com/wiertmir/mini-rag/issues/76).
Test bibliography filtering or reranking as a new experiment, preserving
the current baseline.

## Unfair scores and lost citations

Three OpenAI answers missed the expected-document metric even though
I found support for their content in the PDFs:

| Question | Labeled expectation | Actual cited evidence | Interpretation |
|---|---|---|---|
| Q01 | WGIII_CH01 | WGI_CH01, physical p.9 | Explicitly states WGIII assesses mitigation response options. Valid shorter answer. |
| Q09 | WGIII_CH05 / WGIII_CH06 | WGIII_CH11, physical p.27; also WGIII_CH08 | Efficiency enables clean electrification and reduces supply needs. |
| Q19 | WGIII_CH11 | WGIII_CH06, physical p.65 | Names cement, primary steel and chemical feedstocks as challenging to electrify. |

Expected-document corrections are tracked in [#75](https://github.com/wiertmir/mini-rag/issues/75).
After reviewing the sources, version the question set and rescore the saved
answers. The published 25/28 remains the score for the old labels.

Q22 / 4B + Qwen 35B has a separate problem. The model used markers [2, 5]
and [1, 3, 4], while the parser supports single [n] markers. Only reference 3
survived in the saved citations. Grouped-citation handling needs a defined rule,
with the original answer text preserved: [#78](https://github.com/wiertmir/mini-rag/issues/78).
For Q08, we need the failure stage and an offline reproduction attempt:
[#77](https://github.com/wiertmir/mini-rag/issues/77).

## Manual answer and PDF checks

I compared ten saved OpenAI answers with the draft reference answers.
This is a selected sample, so I do not use it to estimate overall accuracy.

| Question | Assessment |
|---|---|
| Q01 | Correct core purpose, supported by WGI despite the label miss. |
| Q02 | Correct mitigation/adaptation distinction, including an explicit adaptation definition. |
| Q03 | Demographic/economic/activity drivers and energy/land/process sources align with the reference. |
| Q05 | Rapid cuts and mid-century net zero; numerical timing is supported on cited pages. |
| Q09 | Correct complementary efficiency/electrification roles; label false negative. |
| Q10 | Low-carbon electricity and system integration needs align with the reference. |
| Q19 | Appropriate steel/cement/chemicals examples; label false negative. |
| Q25 | De-risking and mobilizing private capital align with the reference and substantive chapter evidence. |
| Q29 | Explicit refusal, no Apple earnings claim or citations. |
| Q30 | Explicit refusal, no invented Tokyo population or citations. |

I also reviewed every local answer scored as a miss, plus Q08's error.
The appendix records document presence, not passage quality.

I checked five selected claims against the PDF text. Page numbers below count
physical PDF pages from 1. For the multi-page citation, I checked both pages
and the range against the retrieved passage.

| Claim | Citation | Short source evidence | Result |
|---|---|---|---|
| Q01: WGIII's role | WGI_CH01 p.9; WGI_CH01_p009_c001 | “Working Group III (WGIII) assesses mitigation response options.” | Supported |
| Q02: adaptation adjusts to climate effects | WGIII_CH14 p.47; WGIII_CH14_p047_c002 | “adjustment to actual or expected climate and its effects” | Supported |
| Q05: immediate cuts / net-zero timing | WGIII_CH03 pp.4–5; WGIII_CH03_p004_c003 | p.4 defines immediate action; p.5 gives CO2 halving in the 2030s and net zero in the 2050s. | Supported across both pages |
| Q09: efficiency complements electrification | WGIII_CH11 p.27; WGIII_CH11_p027_c002 | “Energy efficiency is a critical strategy for net zero transitions and enabling clean electrification.” | Supported |
| Q19: hard-to-electrify industry | WGIII_CH06 p.65; WGIII_CH06_p065_c001 | Cement, primary steel and chemical feedstocks are challenging to electrify. | Supported |

All five claims were supported. Separately, I checked 518 citation metadata
entries and all retained chunk IDs against the current chunks.
Matching metadata does not guarantee a correct answer.

## End-of-week checks

Week 1 passed; follow-ups #75, #76, #77 and #78 remain open.
Week 2 can use the current pipeline, schemas and embeddings as its reference.
This verdict covers week 1, not production readiness.

- The repository was clean at audit start. I used a separate worktree with
  Python 3.13.15. The required Day 1 layout was complete.
- All 20 manifest-listed PDFs were readable; no downloads were needed.
- Two full ingests each produced 6264 chunks, byte-identical to the existing
  chunk file. They took 334.45 s and 333.42 s and were saved to
  `/tmp/pr74-week/run1` and `/tmp/pr74-week/run2`, without overwriting indexes.
- Index row counts, dimensions, hashes and ingest parameters matched:
  OpenAI (6264, 1536), Qwen 4B (6264, 2560), Qwen 8B (6264, 4096).
  Looking up a vector in its own index returned a score of approximately 1.
- Offline tests without an API key and with a closed endpoint: 313 passed,
  2 live tests deselected. Evaluation tests: 38 passed.
  Ruff lint/format, mypy for source and modified tests, and import checks passed.
  Saved-result tests were expanded from one run to all seven; smoke tests
  recompute hits against the current questions.
- [Acceptance run 44738489501](https://github.com/wiertmir/mini-rag/actions/runs/44738489501):
  6 passed, 0 skipped on `9c8e117`, after verifying PDF completeness.
  It covers metadata, determinism, both chunking strategies, overlap, missing
  PDFs and empty pages. Later commits changed only tests and documentation.
- `.env`, indexes, PDFs and logs were not tracked by Git; the example key was
  a placeholder. A history scan returned one false positive in a merge title
  (`ask-function-and-cli-command`), with no key. A patch-only scan found no candidates.
- The audit reused saved answers and embeddings. It made no answer-generation,
  query-embedding or corpus re-embedding requests. Learner implementation and
  saved results were unchanged. Final CI and merge status are in the PR;
  reproduction instructions are in the README.

## Measurement setup and future comparisons

Results and statistics were recomputed from records. All seven runs share:

- 30 questions, `top_k=5`, sequential execution;
- section chunking 600/90, minimum 150 tokens; matching tokenizer,
  caption-skipping and box-handling settings;
- dataset SHA-256: `268f7b1440c1ac549a0c77d663cb0758452208ecfbe3259b584d058c2e679427`;
- chunks SHA-256: `6b2ae2d4d87996d82b83c373c96cba78d3c0a10672b48990c71f35e54d8d1414`;
- prompt SHA-256: `905c77411aa7334bd74922a19d0cc708772e69d24ca5a421bd092e626cb41558`.

The new OpenAI run matches the available question and chunk files.
The earlier mismatch noted in PR #72 does not affect this saved run;
the older result remains in Git history.

Each combination ran only once. We have six local configurations and one
all-OpenAI configuration, so we cannot separate every model's effect.
Qwen rankings differ between retrieval smoke and answer runs: for 4B at Q14
and Q01 with the Qwen generator; for 8B at Q10 and Q01 with Gemma 31B.
The cause is unknown. Generators did not always receive exactly the same context.
Reference answers still have status `draft_not_page_verified`.

The 12B alias identifies an 8K setup. Summaries do not record all effective
context/KV-cache settings, quantization digests, thinking budgets, output-token
counts or warm/cold state. Timings describe these runs; they cannot establish
that architecture alone caused the differences.

Next, improve labels and passage-relevance evaluation (#75, #76), add error
diagnostics (#77), and define citation syntax (#78). For speed comparisons,
cache question vectors, fix the exact contexts, warm one answer model,
record its settings and repeat the measurements.

## Appendix: every scored miss on answerable questions

`present` means an expected document was in the context, not necessarily
that the passage contained an answer. `absent` means it was missing.

| Embeddings | Answer model | Question | Expected document | Observed outcome |
|---|---|---|---|---|
| qwen3-embedding:4b | gemma4:12b-q8-8k | Q01 | absent | answered, no expected-document citation |
| qwen3-embedding:4b | gemma4:12b-q8-8k | Q18 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:4b | gemma4:12b-q8-8k | Q21 | absent | refusal |
| qwen3-embedding:4b | gemma4:31b | Q01 | absent | answered, no expected-document citation |
| qwen3-embedding:4b | gemma4:31b | Q10 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:4b | gemma4:31b | Q21 | absent | refusal |
| qwen3-embedding:4b | qwen3.6:35b | Q01 | absent | answered, no expected-document citation |
| qwen3-embedding:4b | qwen3.6:35b | Q10 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:4b | qwen3.6:35b | Q21 | absent | refusal |
| qwen3-embedding:4b | qwen3.6:35b | Q22 | present (not proof of answer-bearing evidence) | answered, no expected-document citation; grouped-marker loss |
| qwen3-embedding:8b | gemma4:12b-q8-8k | Q01 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:12b-q8-8k | Q10 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:12b-q8-8k | Q21 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | gemma4:31b | Q01 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q05 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q10 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q19 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q21 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | gemma4:31b | Q25 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | qwen3.6:35b | Q01 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | qwen3.6:35b | Q08 | unknown: error | ValidationError |
| qwen3-embedding:8b | qwen3.6:35b | Q10 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | qwen3.6:35b | Q21 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | qwen3.6:35b | Q25 | present (not proof of answer-bearing evidence) | refusal |
| text-embedding-3-small | gpt-5.4-mini | Q01 | absent | answered, no expected-document citation |
| text-embedding-3-small | gpt-5.4-mini | Q09 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| text-embedding-3-small | gpt-5.4-mini | Q19 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
