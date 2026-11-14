## Running the project

  > [!NOTE]
  > The project is currently at the environment and data preparation stage.
  > The `ingest`, `embed`, `retrieve`, and `ask` commands will be implemented
  > during the following development stages.

  ### 1. Set up the environment

  The project supports Python 3.12 and 3.13 (`>=3.12,<3.14`) and uses Python 3.13.15 for
  local development. PyMuPDF has compatibility issues with Python 3.14, so
  Python 3.14 is not currently supported.

  ```bash
  pyenv install -s 3.13.15
  pyenv local 3.13.15

  python -m pip install --upgrade pip
  pip install -e ".[dev]"
  ```

  ### 2. Verify the installation

  ```bash
  python --version
  python -c "import mini_rag; print('The mini_rag package works')"
  ```

  Expected output:

  ```text
  Python 3.13.15
  The mini_rag package works
  ```

  Ruff and mypy target Python 3.12, the minimum supported version. Run the
  lint check with:

  ```bash
  python -m ruff check .
  ```

  ### Run the embedding tests

  ```bash
  python -m pytest tests/test_embed.py
  ```

  Async tests run automatically with `pytest-asyncio`, included in the dev
  dependencies. The embedding tests use a fake client and make no API calls.

  Logging is configured by the `embed` and `ingest` CLI entry points, not
  when importing modules. Both commands write daily logs to `log_folder`
  (`logs/` by default), named after the running script, for example
  `embed_YYYY_MM_DD.log`. Ingest also configures its spawned worker processes.
  Importing modules or collecting tests does not create log files.

  ### Continuous integration

  - **CI** (`.github/workflows/ci.yml`) runs on every push to `main` and every
    pull request: `ruff check`, `ruff format --check`, `mypy src`, and
    `pytest -m "not acceptance"` on Python 3.12 and 3.13. Reproduce locally:

    ```bash
    python -m ruff check . && python -m ruff format --check . \
      && python -m mypy src && python -m pytest -q -m "not acceptance"
    ```

  - **Acceptance** (`.github/workflows/acceptance.yml`) runs the slow black-box
    tests (marker `acceptance`) and is started manually at milestones:

    ```bash
    gh workflow run acceptance.yml              # all acceptance tests
    gh workflow run acceptance.yml -f select=w01_d2
    ```

    It downloads (and caches) the IPCC corpus and fails if any PDF is missing,
    so missing data cannot turn into a green run full of skipped tests.

  ### 3. Prepare the data

  The `documents.csv`, `questions.jsonl`, and `download_pdfs.py` files should
  be located in the `data/` directory.

  Download the documents:

  ```bash
  python data/download_pdfs.py
  ```

  The script should create a `data/pdfs/` directory containing 20 PDF documents.

  Verify the dataset:

  ```bash
  test -f data/documents.csv
  test -f data/questions.jsonl
  test -f data/download_pdfs.py
  test "$(find data/pdfs -maxdepth 1 -name '*.pdf' | wc -l)" -eq 20
  test "$(wc -l < data/questions.jsonl)" -eq 30
  ```

  No output and an exit code of `0` mean that all checks passed.

  ## Planned usage

  The following commands describe the intended project interface. They have not
  been implemented yet.

  ### Process the documents

  ```bash
  python -m mini_rag.ingest \
    --manifest data/documents.csv \
    --pdf-dir data/pdfs \
    --output-dir index \
    --chunk-size 800 \
    --chunk-overlap 120
  ```

  Ingest creates the output directory if needed and writes
  `chunks_<chunk_size>_<chunk_overlap>.jsonl` and
  `ingest_manifest_<chunk_size>_<chunk_overlap>.json` there. The command above
  produces `index/chunks_800_120.jsonl` and `index/ingest_manifest_800_120.json`.
  Different chunking settings can share the same output directory.
  Configure it with `--output-dir`, `OUTPUT_DIR`, or `output_dir` in `[ingest]`.

  Chunking follows the document structure:

  - A new section starts at every heading, unless the previous block is a
    heading too, so `3` / `3.1 Introduction` stay together with their text.
  - Sections run across page boundaries; a chunk records the page it starts on
    (`page`) and ends on (`page_end`). `chunk_id` numbers chunks per start
    page: `<document>_p<page>_c<n>`.
  - A section below `--min-chunk-tokens` (default 200) is merged into the
    next one (the last section of a document merges into the previous one).
  - A section above `--chunk-size` is split into evenly sized windows sharing
    exactly `--chunk-overlap` tokens, so there are no tiny near-duplicate tails.
    Chunks of different sections do not overlap.

  The ingest manifest records the strategy (`"strategy": "sections"`) and
  `min_chunk_tokens` next to the size and overlap.

  The storage readers `mini_rag.storage.read_chunks` and `read_chunks_async`
  raise `ChunkParseError` for malformed JSON and `ChunkValidationError` for
  records that do not satisfy the chunk schema. Both inherit from
  `ChunkReadError`, expose `path` and the 1-based `line_no`, and preserve the
  original Pydantic exception as their cause. Blank lines are invalid JSON;
  filesystem errors propagate unchanged.

  ### Generate embeddings

  Embedding settings use the `[embed]` TOML table. `chunks` is required and has
  no default; provide it with `--chunks`, `CHUNKS`, or the TOML table.
  Defaults are `output_dir = "index"`,
  `embedding_model = "text-embedding-3-small"`, and `batch_size = 256`.

  `mini_rag.embed.save_index` requires a NumPy matrix with dtype `float32`.
  Other dtypes raise `TypeError` before creating the output directory or
  writing files; no implicit conversion is performed.

  ```bash
  python -m mini_rag.embed \
    --chunks index/chunks_800_120.jsonl \
    --output-dir index
  ```

  ### Search the documents

  ```bash
  python -m mini_rag.retrieve \
    "What is climate mitigation?" \
    --top-k 5
  ```

  Retrieval results appear in a Rich table with rank, score, chunk ID,
  document ID, page range, and text preview. API, filesystem, and validation errors
  appear in red on stderr and exit with status 1.

  ### Display a document's chunks

  ```bash
  python -m mini_rag.display --document-id WGIII_CH03
  ```

  Shows every chunk of the document in a Rich table with chunk ID and full,
  multiline text and its page range. Text shared by consecutive chunks (the
  windows of one section) appears in a separate yellow row labelled, for
  example, `c001 ↔ c002`; shared matches shorter than 20 characters are
  ignored. Chunks that share no text, i.e. different sections, are separated
  by lines. Chunks are read from `--chunks`, or from the
  `chunks_path` in `<index_dir>/embedding_manifest.json` when omitted
  (`--index-dir`, default `index`; TOML table `[display]`).

  Limit the output to some pages with `--pages`, using 1-based inclusive
  ranges separated by commas. Open ranges are allowed: `5-` means page 5 to
  the end and `-3` means pages 1–3. Use the `=` form so a leading `-` is not
  read as an option. A chunk is shown when any of its pages is selected. A
  selection without chunks shows an empty table.

  ```bash
  python -m mini_rag.display --document-id WGIII_CH03 --pages="3-7,9,14-"
  ```

  ### Ask a question

  ```bash
  python -m mini_rag.ask \
    "What is climate mitigation?" \
    --top-k 5
  ```

  The `ask` command will return a JSON object containing the generated answer,
  its citations, and an indication of whether the retrieved documents contained
  sufficient evidence.

## Answer contract

Program returns:

- `answer: str` — answer solely based on found context,
- `citations: list[Citation]` — document ids and pages,
- `insufficient_evidence: bool` — info depicting if documents provide enough to return the answer.

If data is not sufficient:

- `answer` contains the message about lack of information,
- `citations` is an empty list,
- `insufficient_evidence` have value `true`.

### Valid answer

```json
{
"answer": "Transport accounts for approximately 15% of global greenhouse-gas emissions.",
"citations": [
    {
    "document_id": "WGIII_CH10",
    "page": 1052
    }
],
"insufficient_evidence": false
}
```

### Not enough information answer

```json
{
"answer": "There is insufficient information in the provided documents.",
"citations": [],
"insufficient_evidence": true
}
```