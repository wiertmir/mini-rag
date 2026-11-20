# Week 1 evaluation results

**Reviewed:** 2026-11-20 · **Experiment commit:** 9c8e117 · **PR:** [#74](https://github.com/wiertmir/mini-rag/pull/74) · **Week review:** [#21](https://github.com/wiertmir/mini-rag/issues/21)

## My conclusion

This is a useful, auditable first RAG baseline. My practical local choice from
these runs is **Gemma 4 12B Q8 with an 8K context**. It matches or exceeds the
expected-document citation hits of the larger local models while taking
substantially less end-to-end time in these samples. Keep **text-embedding-3-small
+ gpt-5.4-mini** as the API-backed reference.

The most important finding is not “8B embeddings win” or “25/28 answers are
correct.” **Document-level metrics penalize valid evidence from other chapters
and reward irrelevant bibliography fragments.** Improve evidence labels and
passage relevance before spending more time comparing model sizes.

## 1. What is actually comparable

I validated **seven runs / 210 answer records**, plus **three retrieval smoke
runs / 90 records**. Counts and latency statistics were recomputed from records.

All seven answer runs use the same:

- 30 questions: 28 answerable, Q29/Q30 unanswerable;
- dataset SHA-256: 268f7b1440c1ac549a0c77d663cb0758452208ecfbe3259b584d058c2e679427;
- chunks SHA-256: 6b2ae2d4d87996d82b83c373c96cba78d3c0a10672b48990c71f35e54d8d1414;
- prompt SHA-256: 905c77411aa7334bd74922a19d0cc708772e69d24ca5a421bd092e626cb41558;
- top_k=5, sequential execution, section chunking 600/90, minimum section 150,
  matching tokenizer/caption/box parameters.

The newly saved OpenAI run matches the available local chunk/dataset bytes.
This resolves the *current artifact* mismatch noted during PR #72, without
retroactively changing that historical measurement. The older OpenAI run
remains in Git history.

Causal-comparison limitations:

- One generation run per combination; no repeated-seed confidence interval.
- Six local combinations plus one all-OpenAI combination, not a complete matrix
  separating cloud/local embedding and answer-model effects.
- Some Qwen rankings differ across generation runs and smoke: 4B Q14, additionally
  Q01 for the Qwen generator; 8B Q10, additionally Q01 for Gemma 31B. These are
  not exact fixed-context generator comparisons. The cause of drift is unproved.
- **cites_expected_document is not answer accuracy, faithfulness or complete
  multi-source coverage.** One expected document suffices. Reference answers
  remain marked draft_not_page_verified.

## 2. Recomputed answer results

“Answerable refusals” excludes the two correct unanswerable refusals. Errors
are separate from refusals and stay in the 28-question denominator.
All runs have **zero invalid_citation errors**; Q29/Q30 refuse without citations
in **all seven runs**. There is one execution error across 210 records.

| Embeddings | Answer model / records | Citation hits | Answerable refusals | Errors | Mean s | Median s | Max s |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding:4b | [gemma4:12b-q8-8k](eval/results/qwen3-embedding:4b/gemma4:12b-q8-8k/baseline.jsonl) | 25/28 | 2 | 0 | 35.99 | 34.56 | 75.80 |
| qwen3-embedding:4b | [gemma4:31b](eval/results/qwen3-embedding:4b/gemma4:31b/baseline.jsonl) | 25/28 | 2 | 0 | 81.36 | 78.15 | 134.99 |
| qwen3-embedding:4b | [qwen3.6:35b](eval/results/qwen3-embedding:4b/qwen3.6:35b/baseline.jsonl) | 24/28 | 2 | 0 | 41.27 | 41.16 | 65.72 |
| qwen3-embedding:8b | [gemma4:12b-q8-8k](eval/results/qwen3-embedding:8b/gemma4:12b-q8-8k/baseline.jsonl) | 25/28 | 1 | 0 | 34.04 | 31.89 | 72.30 |
| qwen3-embedding:8b | [gemma4:31b](eval/results/qwen3-embedding:8b/gemma4:31b/baseline.jsonl) | 22/28 | 2 | 0 | 90.93 | 89.09 | 158.87 |
| qwen3-embedding:8b | [qwen3.6:35b](eval/results/qwen3-embedding:8b/qwen3.6:35b/baseline.jsonl) | 23/28 | 2 | 1 | 64.61 | 58.67 | 118.42 |
| text-embedding-3-small | [gpt-5.4-mini](eval/results/text-embedding-3-small/gpt-5.4-mini/baseline.jsonl) | 25/28 | 0 | 0 | 2.66 | 2.55 | 4.19 |

### What I would choose

- **Local default: Gemma 4 12B Q8 / 8K.** 25/28 with both embedding sizes.
  Within each embedding setup, observed mean latency is about 2.26× (4B) or
  2.67× (8B) lower than Gemma 31B. This is practical evidence on this machine,
  not proof of equal semantic quality or universal model-speed superiority.
- **Gemma 31B is not justified here:** more memory/time, 25/28 or 22/28,
  without demonstrated improvement over 12B.
- **Qwen 3.6 35B is an alternative, not the default winner:** 24/28 or 23/28,
  plus Q08's execution error with 8B embeddings.
- **OpenAI is the low-latency reference:** 25/28, no errors, mean 2.66 seconds.
  Privacy, offline operation and avoiding per-request charges can still favor
  local inference. Token costs/energy were not recorded: no cost ranking is claimed.
- **Embedding choice is not settled.** 8B improves document hit@5 but not 12B's
  answer-hit count, and larger generators score lower with it. 4B is a reasonable
  lighter baseline; assess answer-bearing passages before declaring 8B an upgrade.

### Why 34 seconds can coexist with a fast GPU model

These are end-to-end latencies: query embedding, scheduling/model loading,
generation and post-processing, not isolated tokens/second. On a 16 GB GPU,
local embedding and answer models may trigger repeated loads; larger answer
models can offload to RAM. Concurrent Qwen requests caused swapping during a
separate setup check, but saved summaries do not establish contention during
every experiment.

The 12B alias identifies an 8K setup. Summaries do not freeze all effective
context/KV-cache settings, quantization digests, thinking budgets, output-token
counts or warm/cold state. **Do not attribute differences to architecture alone.**
A later controlled benchmark should cache query vectors, hold exact contexts
fixed, warm one answer model, record these settings and repeat runs.

## 3. Retrieval score versus useful evidence

| Embeddings | Document hit@5 | Saved document misses |
|---|---:|---|
| text-embedding-3-small | 27/28 | Q01 |
| qwen3-embedding:4b | 26/28 | Q01, Q21 |
| qwen3-embedding:8b | 28/28 | None |

**28/28 does not mean 28 contexts contain usable answers.**

- **Q18 / 4B:** all five chunks belong to the expected transport chapter, but
  four are bibliographic material. The remaining aviation introduction contains
  useful information without establishing the full road-vs-air energy-density
  comparison in the draft reference. Document presence alone does not prove
  that a refusal is a generator failure.
- **Q21 / 8B:** the expected-document hit, WGIII_CH12_p043_c001, is a
  table/source-reference fragment rather than an LCA explanation. Several other
  chunks are bibliography. All local models refuse: bigger is not automatically better.
- **Q25 / 8B + Gemma 12B:** all five contexts are bibliography. The answer
  synthesizes explanatory claims from titles about de-risking and finance and
  earns a citation hit. Titles are not equivalent to substantive support from
  the studies or IPCC discussion. The larger models' refusals should not simply
  be called lower factual quality.

Follow-up: [#76 — bibliography and passage relevance](https://github.com/wiertmir/mini-rag/issues/76).
Any filtering/reranking change must be a separate measured experiment, not a
silent overwrite/re-embedding of the reference corpus.

## 4. False negatives and citation-format losses

Three OpenAI-reference “misses” are supported by physical PDF evidence:

| Question | Labeled expectation | Actual cited evidence | Interpretation |
|---|---|---|---|
| Q01 | WGIII_CH01 | WGI_CH01, physical p.9 | Explicitly states WGIII assesses mitigation response options. Valid shorter answer. |
| Q09 | WGIII_CH05 / WGIII_CH06 | WGIII_CH11, physical p.27; also WGIII_CH08 | Efficiency enables clean electrification and reduces supply needs. |
| Q19 | WGIII_CH11 | WGIII_CH06, physical p.65 | Names cement, primary steel and chemical feedstocks as challenging to electrify. |

**Do not edit labels in place merely to improve scores.** Review evidence,
version the dataset and rescore preserved records offline. Published 25/28
remains the score against its frozen labels. See [#75](https://github.com/wiertmir/mini-rag/issues/75).

**Q22 / 4B + Qwen 35B** is different: its answer uses grouped markers
[2, 5] and [1, 3, 4]. The resolver supports single [n] markers, so only ref 3
becomes a saved citation. That is a format/post-processing limitation, not
clean evidence that the model never referred to expected chapters. Decide
acceptance/rejection/normalization explicitly while preserving text as source
authority. See [#78](https://github.com/wiertmir/mini-rag/issues/78).

## 5. A real execution error, with an unknown cause

**Q08 / 8B + Qwen 35B** has error=ValidationError. The loop correctly continued,
wrote 30 records and counted it separately from refusals. Null answer/context
fields cannot distinguish SDK output parsing from downstream validation.
I will not invent a root cause or classify it as a refusal.

Preserve this result and add safe stage-level diagnostics. Prefer an offline
reproduction; do not rerun the entire paid baseline to debug one row.
See [#77](https://github.com/wiertmir/mini-rag/issues/77).

## 6. Ten-record semantic spot-check of the OpenAI reference

Compared saved answers with draft reference answers. This targeted check is
**not a statistically representative accuracy estimate**.

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

Also inspected every local record scored as a miss and Q08's error. The appendix
classifies document presence, not whether the retrieved passage truly answers
the question. Cautious refusal and incomplete labels are not necessarily model errors.

## 7. Five checks against the actual PDFs

Manual semantic checks of extracted text on **1-based physical PDF pages**,
not merely stored chunk metadata. A multi-page citation was included and its
endpoints checked against the retrieved chunk.

| Claim | Citation | Short source evidence | Result |
|---|---|---|---|
| Q01: WGIII's role | WGI_CH01 p.9; WGI_CH01_p009_c001 | “Working Group III (WGIII) assesses mitigation response options.” | Supported |
| Q02: adaptation adjusts to climate effects | WGIII_CH14 p.47; WGIII_CH14_p047_c002 | “adjustment to actual or expected climate and its effects” | Supported |
| Q05: immediate cuts / net-zero timing | WGIII_CH03 pp.4–5; WGIII_CH03_p004_c003 | p.4 defines immediate action; p.5 gives CO2 halving in the 2030s and net zero in the 2050s. | Supported across both pages |
| Q09: efficiency complements electrification | WGIII_CH11 p.27; WGIII_CH11_p027_c002 | “Energy efficiency is a critical strategy for net zero transitions and enabling clean electrification.” | Supported |
| Q19: hard-to-electrify industry | WGIII_CH06 p.65; WGIII_CH06_p065_c001 | Cement, primary steel and chemical feedstocks are challenging to electrify. | Supported |

**5/5 selected citation claims supported.** This does not certify all sentences
or all citations across 210 answers. All **518 citation metadata entries** were
checked against current chunks, as were all retained chunk IDs. Metadata validity
is a separate, weaker check than factual support.

## 8. Week-1 reproducibility and quality audit

AUDIT_PENDING

## 9. Follow-up priorities without moving the baseline

1. Evidence labels [#75](https://github.com/wiertmir/mini-rag/issues/75) and
   passage relevance/bibliography [#76](https://github.com/wiertmir/mini-rag/issues/76).
2. Actionable error diagnostics [#77](https://github.com/wiertmir/mini-rag/issues/77).
3. Explicit grouped-marker grammar [#78](https://github.com/wiertmir/mini-rag/issues/78).
4. Controlled repeated local performance measurements with cached query vectors,
   fixed contexts and recorded runtime settings.

These are bounded follow-ups, not reasons to erase imperfect experiments or
claim production readiness. Week 2 can reuse this measured plain-Python pipeline,
schemas and row-aligned embeddings as its comparison target.

## Appendix: every scored answerable miss

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
