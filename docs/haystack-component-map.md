# Haystack component map and compatibility boundary

## Agreed migration boundary (#85)

Mirek accepted this direction on 2026-11-28: import existing chunks and vectors,
use Haystack for pipeline wiring, document storage and retrieval, and reuse the
application's context, answer schemas, citation validation and refusal behavior.
This is a design decision, not an implemented RAG pipeline. Only the dependency
probe and the three-instance demo/serialization graph are implemented so far.
See the separate [planned indexing and query flows](haystack-pipelines.md).

Here, **parity** means keeping the same source text, vectors and observable
application rules while changing orchestration. It does not mean claiming equal
retrieval rankings or answer quality before measuring them.

## Nine roles: current code and planned treatment

These are nine responsibilities to map, **not nine new classes to implement**.
Application links point to the current weekly source; framework links below
target the pinned Haystack release. All Haystack RAG components in this table
are planned, including built-in components that have not yet been wired here.

| Role | Current module / symbol | Planned treatment | Input → output | State and side effects |
| --- | --- | --- | --- | --- |
| Loader | [storage.load_index_async](../src/mini_rag/storage.py), [ingest.process_document](../src/mini_rag/ingest.py) for original PDF ingest | Custom cached-index adapter delegates validation to `load_index_async`, then pairs chunks with matrix rows. No PDF converter on the import path. | `Path` + optional chunks `Path` → `LoadedIndex` → `list[Document]` | Reads JSONL, NPY and both manifests. Holds source identity; no network or source-file writes. |
| Cleaner | [pdf_parsers.normalize_text](../src/mini_rag/pdf_parsers.py), [chunk.drop_skipped / move_boxes](../src/mini_rag/chunk.py) | Keep existing code for original ingest; bypass on cached import. A core `DocumentCleaner` would be a separate behavior-change experiment. | PDF text / `Block` records → cleaned text / blocks during original ingest | Configured skip/box rules; no execution or text changes during import. |
| Splitter | [chunk.chunk_blocks / split_section](../src/mini_rag/chunk.py), [ingest.process_document](../src/mini_rag/ingest.py) assigns IDs | Keep section-aware chunking for original ingest; bypass on cached import. A core `DocumentSplitter` is not a drop-in parity replacement. | `Block` records + token settings → `TextChunk` → `list[Chunk]` | Tokenizer, overlap, minimum size and physical page ranges determine chunks. No new IDs or boundaries during import. |
| Document embedder | [embed.embed_chunks / run_embed_async / normalize_async](../src/mini_rag/embed.py) | Reuse cached rows as `Document.embedding`. Do not put `OpenAIDocumentEmbedder` in the import graph. | Ordered chunks + float32 matrix `(N, D)` → documents with `list[float]` embeddings | Original embed may call the API and write the source index; planned import does neither. |
| Writer | [storage.save_index](../src/mini_rag/storage.py) writes the original NumPy index | Core `DocumentWriter` into `InMemoryDocumentStore`. Duplicate policy is chosen in #87; persistence lifecycle in #88. | `list[Document]` → `documents_written: int` | Mutates the in-process store; separate persistence writes only the selected Haystack artifact, never the source index. |
| Query embedder | [retrieve.retrieve_from_index_async](../src/mini_rag/retrieve.py) calls [embed.embed_texts_async](../src/mini_rag/embed.py) | Core `OpenAITextEmbedder`, injected/replaced by a fixed-vector fake in offline tests. Select model from the imported manifest. | Question `str` → `embedding: list[float]` of dimension `D` | Runtime model/client settings; one query embedding request per question, not document re-embedding. |
| Retriever | [retrieve.search](../src/mini_rag/retrieve.py), [llm.OpenAIRetriever](../src/mini_rag/llm.py) | Core `InMemoryEmbeddingRetriever`, then an application adapter preserves ordering, metadata and optional score filtering. | Query embedding + store + `top_k` → scored `list[Document]` → `list[SearchResult]` | Reads loaded store. Use cosine mode and `scale_score=False`; normalize ties to score descending, chunk ID ascending. No model calls here. |
| Prompt builder | [prompts.build_context / SYSTEM_PROMPT](../src/mini_rag/prompts.py) | Custom wrapper reuses `build_context`; keep existing question/system formatting in the shared generator. No generic template substitution. | `list[SearchResult]` → `(str, dict[int, Chunk])` | Per-question numbered context and local reference map; no I/O. |
| Generator | [llm.OpenAIAnswerGenerator.generate_async](../src/mini_rag/llm.py) | Custom wrapper calls the shared generator and returns `LlmAnswer`. Core `OpenAIChatGenerator` is an alternative to measure later, not the selected first port. | Question + context tuple → `LlmAnswer` | Runtime client/model; answer-model API call and logs. Structured-output failures remain errors. |

### Data that must survive the adapter

For every source row `i`, preserve `Document.id = chunk.chunk_id`,
`Document.content = chunk.text`, and `Document.embedding = matrix[i]` as a list.
Keep source order during import, including repeated text with different IDs.
Preserve all `Chunk` fields needed to reconstruct the validated application
record: `chunk_id`, `document_id`, `title`, `collection`, `source_path`,
`source_url`, `page`, `page_end`, `text` and `token_count`. Content may carry text,
but conversion back must recover it exactly. Store metadata must serialize paths
in a form the application can reconstruct.

`page` and `page_end` are inclusive, 1-based physical PDF pages. Do not infer them
from an ID. `LoadedIndex` contains `embeddings_matrix`, ordered `chunks`,
`ingest_manifest` and `embedding_manifest`. The existing loader checks chunk hash,
matrix dtype/shape and row count; #88 adds persisted-store source identity,
including a matrix-file hash. Do not claim that extra check already exists.

## What stays in application code

| Responsibility | Existing source to reuse | Required behavior |
| --- | --- | --- |
| Schemas | [models.py](../src/mini_rag/models.py): `Chunk`, `SearchResult`, `LlmAnswer`, `Citation`, `Answer` | Haystack `Document` is a transport record, not a replacement for validated application output. |
| Context and prompt | [prompts.py](../src/mini_rag/prompts.py), [llm.py](../src/mini_rag/llm.py) | Same numbered excerpts, source labels and system/question formatting. |
| Citation resolution | [ask.resolve_citations / ResolvedCitations](../src/mini_rag/ask.py) | Resolve references through local chunks, preserve page ranges, reject unknown references, retain existing mismatch diagnostics and final `Answer` validation. |
| Refusal | [ask.ask_async](../src/mini_rag/ask.py), [llm.OpenAIRetriever](../src/mini_rag/llm.py) | Apply optional `min_score` after retrieval. If none remain, return the local refusal without calling the answer model. Query embedding still occurs. Model-reported insufficient evidence still passes through existing resolution/validation. |
| Evaluation | [evaluation.baseline_evaluation_loop](../src/mini_rag/evaluation.py), [baseline runner](../eval/run_baseline.py), [retrieval smoke](../eval/retrieval_smoke.py) | Keep baseline artifacts and metrics separate from new Haystack results; demonstrate retrieval/answer parity rather than assuming it. |

## Why reuse first

Reuse keeps prompt, structured-output and citation behavior comparable with the
Week 1 archive. The cost is a few custom wrappers and explicit Document/Chunk
conversion. Framework-native replacements may remove glue later, but they must
be separate measured changes: cleaning/splitting can change text and IDs,
embedders can change vectors, and generators can change prompts or response
parsing. In 3.5.0, `OpenAIChatGenerator` uses Chat Completions and returns
`list[ChatMessage]`; the shared generator uses Responses parsing into `LlmAnswer`.
They are not interchangeable just because both call OpenAI.

Detailed choices not settled by #85 remain with their implementation cards:
loader I/O placement (#86), duplicate handling (#87), persistence format and
rebuild policy (#88), and query factory/adapter details (#91–#94).

### Versioned sources for the planned components

The installed `haystack-ai` 3.5.0 source was inspected for the named interfaces.
These upstream links pin those claims to the same release:

- [Document and embedding/metadata fields](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/dataclasses/document.py)
- [InMemoryDocumentStore, including cosine selection](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/document_stores/in_memory/document_store.py)
- [DocumentWriter and duplicate policy](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/components/writers/document_writer.py)
- [OpenAITextEmbedder sockets](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/components/embedders/openai_text_embedder.py)
- [InMemoryEmbeddingRetriever and score scaling](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/components/retrievers/in_memory/embedding_retriever.py)
- [OpenAIChatGenerator, the unselected native alternative](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/components/generators/chat/openai.py)

The document store does not make corpus persistence part of pipeline YAML.
Its 3.5.0 class describes the store as ephemeral; #88 must verify a supported
round trip or define a narrow corpus serializer. An in-memory successful write
does not prove a fresh process can retrieve those documents.

## Selected release

Issue #82 selects `haystack-ai==3.5` (installed distribution version `3.5.0`).
These are equivalent exact-version spellings under Python package version rules;
this is not a `3.5.*` range. Upstream metadata requires Python >=3.11; this
project retains its narrower >=3.12,<3.14 range and pyenv Python 3.13.15.

Verified release sources:

- [PyPI 3.5.0 metadata](https://pypi.org/pypi/haystack-ai/3.5.0/json)
- [Release tag](https://github.com/deepset-ai/haystack/tree/v3.5.0)
- [Version matching rules](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-matching)

The links below target this release's source and API docstrings, rather than
assuming that the rolling documentation describes the pinned version.

## API contracts

| Interface | Contract in 3.5.0 | Versioned source |
| --- | --- | --- |
| `Document` | Carries `content`, `meta`, ID, score and optional embedding. Converting our chunk schema is a later step. | [Document](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/dataclasses/document.py) |
| `@component` | Decorates a component class; `run` parameters define input sockets and `@component.output_types(...)` declares output sockets. `run` returns a dictionary keyed by output name. | [Component](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/component/component.py) |
| `Pipeline.add_component(name, instance)` | Registers an instance under a unique name. | [PipelineBase](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/pipeline/base.py) |
| `Pipeline.connect(sender, receiver)` | Connects socket names such as `first.documents` and `second.documents`; validates component/socket existence and type compatibility. | [PipelineBase](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/pipeline/base.py) |
| `Pipeline.run(data, include_outputs_from=...)` | Nested input dictionary by component and socket; output dictionary by component. Include named intermediate components explicitly to retain consumed outputs. | [Pipeline](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/pipeline/pipeline.py) |
| Async execution | This release exposes `Pipeline.run_async(data, include_outputs_from=None, concurrency_limit=4)` and `run_async_generator(...)`; do not assume older examples requiring a separate async pipeline class. | [Pipeline](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/pipeline/pipeline.py) |
| Serialization | `to_dict` / `from_dict`, `dumps` / `loads`; string serialization defaults to YAML. Custom component initialization state needs compatible serialization methods. | [PipelineBase](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/pipeline/base.py) |
| Deserialization allow-list | `loads(..., allowed_modules=[...], unsafe=False)` and `from_dict` accept additional module patterns. Defaults trust Haystack namespaces and selected standard-library modules. Future project components must be explicitly allowed; do not bypass checks with `unsafe=True`. | [PipelineBase](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/core/pipeline/base.py) |
| `InMemoryDocumentStore` | `write_documents` and `count_documents` establish the offline store boundary. It is not the existing NumPy index or a persistent external vector database. | [Store](https://github.com/deepset-ai/haystack/blob/v3.5.0/haystack/document_stores/in_memory/document_store.py) |

## Verified scope and evidence

`tests/test_haystack_smoke.py` starts a clean interpreter, checks the installed
version against `pyproject.toml`, removes API-key environment variables, disables
telemetry, and rejects socket connections and DNS lookups. It imports all five
existing CLI modules, writes an in-memory document, runs two built-in components
with a typed connection and intermediate outputs, and round-trips pipeline YAML.
This is a dependency compatibility probe, not the learner's RAG implementation.
Async and custom-component serialization contracts above are source-verified;
the smoke test does not claim to implement or validate the future adapter.

CI logs are evidence for installation and the fast gate at their recorded commit.
They do not prove which commands a person previously typed locally. No narrative
or screenshots are required from Mirek for results that tests can reproduce.

## Demo serialization boundary (#84)

`components.build_pipeline()` now constructs the existing graph without running
it: two `TextStats` instances and one `WhitespaceNormalizer`, with the normalized
text connected to the downstream statistics instance. The original demo calls
this factory. No corpus conversion or RAG adapter is introduced.

`roundtrip.build_save_run(text, path)` writes YAML through `Pipeline.dump(file)`
and returns the original run result as sorted JSON. `roundtrip.load_run(text,
path)` uses `Pipeline.load(file, allowed_modules=["mini_rag.haystack_pipeline.components"],
unsafe=False)` and returns the loaded graph's result. These file-based APIs wrap
the same serialization boundary as `dumps`/`loads`; using them is not a contract
violation. The file stores class paths, empty initialization parameters and the
socket connection, not runtime text or results. A missing/broken file fails,
without constructing a replacement graph.

The offline tests cover separate interpreter runs, exact final/intermediate
values, a factory failure sentinel in load mode, rejection without the module
allowance, and parsed YAML contents. They also distinguish a successful JSON diff
from two failed commands producing empty output: each process must exit zero.

The CLI accepts an optional positional text argument, defaulting to the #84
fixture `"One   two\nthree"`. Both documented commands therefore work without
supplying text, while callers can still provide their own input. The factory has
an explicit `Pipeline` return annotation and the save/load help labels match the
actual modes. No custom serialization hooks are needed by these stateless
components. The #84 round trip and loading restrictions are covered by offline
regression tests; this is not a claim about future RAG pipelines or week-level
acceptance.
