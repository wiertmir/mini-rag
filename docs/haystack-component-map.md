# Haystack compatibility boundary

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
