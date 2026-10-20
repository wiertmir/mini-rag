## Running the project

  > [!NOTE]
  > The project is currently at the environment and data preparation stage.
  > The `ingest`, `embed`, `retrieve`, and `ask` commands will be implemented
  > during the following development stages.

  ### 1. Set up the environment

  The project requires Python 3.13.

  ```bash
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
    --output index/chunks.jsonl \
    --chunk-size 800 \
    --chunk-overlap 120
  ```

  ### Generate embeddings

  ```bash
  python -m mini_rag.embed \
    --chunks index/chunks.jsonl \
    --output-dir index
  ```

  ### Search the documents

  ```bash
  python -m mini_rag.retrieve \
    "What is climate mitigation?" \
    --top-k 5
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