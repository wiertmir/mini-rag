# Planned Haystack pipelines — Week 2

These diagrams record the #85 design accepted by Mirek on 2026-11-28.
**The RAG graphs below are not implemented yet.** The working Day 1 demo only
normalizes text, counts it and round-trips its graph through YAML.
The [component map](haystack-component-map.md) names existing code, typed
boundaries and the native alternatives we are not selecting for the first port.

**Legend:** dashed boxes are planned custom application adapters; solid component
boxes are Haystack core; cylinders are files on local disk or an explicitly
labelled RAM store. No document-embedding API participates in indexing.

## 1. Indexing: import existing chunks and vectors (#86–#90)

```mermaid
flowchart LR
    subgraph Disk["Existing local disk — read only"]
        EM[("embedding_manifest.json")]
        Chunks[("Manifest-selected chunk JSONL")]
        Matrix[("embeddings.npy — float32 N by D")]
        IM[("Adjacent ingest manifest")]
    end
    Loader["Custom cached loader<br/>reuse storage.load_index_async"]
    Adapter["Custom Chunk-to-Document adapter<br/>pair chunk i with row i"]
    Writer["Core DocumentWriter"]
    Store[("RAM InMemoryDocumentStore")]
    Persist["Custom persistence lifecycle<br/>validate complete store, stage, publish"]
    Target[("Local Haystack store artifact<br/>separate from graph YAML")]
    Identity[("index/haystack_manifest.json<br/>source identity and import results")]
    EM -->|"Model, D, N, chunks path and hash"| Loader
    Chunks -->|"Ordered Chunk records"| Loader
    Matrix -->|"Matrix rows"| Loader
    IM -->|"Ingest provenance"| Loader
    Loader -->|"LoadedIndex"| Adapter
    Adapter -->|"list of Document with original IDs and vectors"| Writer
    Writer -->|"Write Documents under explicit duplicate policy"| Store
    Writer -->|"documents_written: int"| Persist
    Store -->|"Complete corpus: IDs, text, metadata and vectors"| Persist
    Persist -->|"Validated corpus data"| Target
    Persist -->|"Hashes, model, dimension, counts and parameters"| Identity
    Target -.->|"Existing store loaded before unchanged re-import"| Persist
    Persist -.->|"Restore validated Documents before writer runs"| Store
    classDef custom stroke-dasharray: 6 4;
    class Loader,Adapter,Persist custom;
```

The source directory is normally `index/text-embedding-3-small/`; its manifest
selects the chunks file, normally `index/chunks_600_90.jsonl`. The adjacent ingest
manifest is normally `index/ingest_manifest_600_90.json`. Paths are inputs, not
hardcoded assumptions. A bad hash/shape/dtype fails: it never triggers PDF ingest
or document embedding.

On re-import, load and validate the persisted corpus **before** writing into the
store. Verify its identity against the source cache. A changed source follows
the explicit reject/rebuild policy chosen in #88; rebuilding means importing
cached vectors, not generating new embeddings. Write a new corpus artifact only
after complete validation, keeping the previous valid artifact on failure.
The store path/format remains a #88 decision. The success manifest is specified
in #89, not a file already produced by this design task.

## 2. Query: retrieve, decide, generate and validate (#91–#95)

The query graph and indexing graph must use the same loaded corpus and embedding
space. A new `InMemoryDocumentStore` created by loading graph YAML may be empty.
Load the persisted data and validate the manifest before accepting questions.

```mermaid
flowchart TB
    Corpus[("Local persisted Haystack corpus + import manifest")]
    Load["Custom store loader and identity check"]
    Store[("RAM InMemoryDocumentStore<br/>cosine similarity")]
    Question["Question: str, top_k: int, optional min_score"]
    Embed["Core OpenAITextEmbedder<br/>manifest-selected model"]
    EmbAPI["External API: query embedding model"]
    Retriever["Core InMemoryEmbeddingRetriever<br/>scale_score=False"]
    Adapt["Custom result adapter<br/>stable ordering, Chunk metadata and min_score gate"]
    Trace["Retrieval trace<br/>raw scores, ranks, IDs and page ranges"]
    Gate{"Any results remain?"}
    Refuse["Local Answer<br/>insufficient_evidence=True, no citations"]
    Context["Custom context wrapper<br/>reuse prompts.build_context"]
    Generate["Custom generator wrapper<br/>reuse OpenAIAnswerGenerator"]
    AnswerAPI["External API: structured answer model"]
    Validate["Custom finalization<br/>reuse resolve_citations and Answer validation"]
    Output["Answer to caller"]
    Corpus -->|"Corpus data and source identity"| Load
    Load -->|"Validated Documents"| Store
    Load -->|"Model and dimension"| Embed
    Question -->|"Question text"| Embed
    Embed -->|"Question and model ID"| EmbAPI
    EmbAPI -->|"list of float, dimension D"| Embed
    Embed -->|"embedding: list of float"| Retriever
    Question -->|"top_k"| Retriever
    Store -->|"Stored Documents with embeddings"| Retriever
    Retriever -->|"Scored list of Document"| Adapt
    Question -->|"optional min_score"| Adapt
    Adapt -->|"Pre-filter retrieval trace"| Trace
    Adapt -->|"Retained list of SearchResult"| Gate
    Gate -->|"No"| Refuse
    Refuse -->|"Answer"| Output
    Gate -->|"Yes: retained SearchResult list"| Context
    Context -->|"Numbered context str and local ref map"| Generate
    Question -->|"Question text"| Generate
    Generate -->|"Unchanged system prompt, question and context"| AnswerAPI
    AnswerAPI -->|"Structured response parsed as LlmAnswer"| Generate
    Generate -->|"LlmAnswer"| Validate
    Context -->|"dict of int to Chunk"| Validate
    Gate -->|"Retained chunk IDs in rank order"| Validate
    Validate -->|"Validated Answer or explicit error"| Output
    classDef custom stroke-dasharray: 6 4;
    class Load,Adapt,Context,Generate,Validate custom;
```

The store and retriever must expose raw cosine scores, not a rescaled 0–1 value.
Preserve score-descending/chunk-ID-ascending tie ordering at the application
boundary and verify it on tied fixtures; the framework's name alone does not
prove equivalence. The trace exposes retrieval before filtering so discarded
sources are still diagnosable.

An empty retained result list avoids the **answer-model** call, not the query
embedding call. With evidence, the shared generator returns `LlmAnswer`; local
code resolves citations from the reference map and validates the final `Answer`.
It preserves existing diagnostics for mismatched references, rejects unknown
references, and does not accept invented page metadata from the model.

## 3. Refusal and error paths

This sequence expands the decision that must stay visible in the query flow.
The participants are responsibilities, not extra required component classes.

```mermaid
sequenceDiagram
    actor Caller
    participant Query as Planned query wrapper
    participant Search as Core embedder and retriever
    participant App as Shared application rules
    participant Model as Answer-model API
    Caller->>Query: Question, top_k and optional min_score
    Query->>Search: Query embedding and loaded-store retrieval
    Search-->>Query: Ranked Documents and raw scores
    Query->>App: Adapt, order and filter evidence
    alt No retained evidence
        App-->>Caller: Local refusal Answer, no answer-model request
    else Evidence retained
        App->>App: build_context creates text and local ref map
        App->>Model: Existing prompt, question and context
        Model-->>App: Structured response
        alt Missing or invalid structured output
            App-->>Caller: Explicit error, no successful Answer
        else Parsed LlmAnswer
            App->>App: resolve_citations and validate Answer
            alt Unknown ref or invalid final Answer
                App-->>Caller: Explicit validation error
            else Valid answer or model-declared insufficient evidence
                App-->>Caller: Answer with existing citation and refusal rules
            end
        end
    end
```

## What #85 proves

This document and the nine-role map record the chosen design. Parsing these
diagrams and checking the source/API links verifies documentation consistency,
not that the RAG graphs run. Later task PRs provide implementation tests and
saved comparison evidence. Do not regenerate the frozen Week 1 baseline or
claim measured parity from this documentation-only PR.
