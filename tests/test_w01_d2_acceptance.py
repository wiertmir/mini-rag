"""Black-box acceptance tests for School of Swagger task W01-D2.

The suite deliberately exercises only the documented CLI and its artifacts.
It does not prescribe the internal design of ``mini_rag.ingest``.

Run from the repository root:

    pytest -q tests/test_w01_d2_acceptance.py
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tiktoken
from pypdf import PdfWriter

pytestmark = pytest.mark.acceptance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "documents.csv"
SOURCE_PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
DOCUMENT_IDS = ("WGIII_CH01", "WGIII_CH03", "WGI_CH01")
REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "title",
    "collection",
    "source_url",
    "page",
    "text",
    "token_count",
}


@dataclass(frozen=True)
class IngestRun:
    output: Path
    manifest: Path
    completed: subprocess.CompletedProcess[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"Ingest did not create {path}"
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        assert line.strip(), f"Blank JSONL line at {path}:{line_number}"
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Invalid JSON at {path}:{line_number}: {exc}")
        assert isinstance(value, dict), f"Record {line_number} is not a JSON object"
        records.append(value)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Ingest did not create {path}"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), f"{path} must contain one JSON object"
    return value


def _select_manifest(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        rows = [row for row in reader if row["id"] in DOCUMENT_IDS]

    assert {row["id"] for row in rows} == set(DOCUMENT_IDS)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_ingest(
    *,
    manifest: Path,
    pdf_dir: Path,
    run_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    require_success: bool = True,
) -> IngestRun:
    output = run_dir / f"chunks_{chunk_size}_{chunk_overlap}.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mini_rag.ingest",
            "--manifest",
            str(manifest),
            "--pdf-dir",
            str(pdf_dir),
            "--output-dir",
            str(run_dir),
            "--chunk-size",
            str(chunk_size),
            "--chunk-overlap",
            str(chunk_overlap),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if require_success:
        assert completed.returncode == 0, (
            f"Ingest exited with {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return IngestRun(
        output=output,
        manifest=run_dir / f"ingest_manifest_{chunk_size}_{chunk_overlap}.json",
        completed=completed,
    )


@pytest.fixture(scope="session")
def selected_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    missing = [
        document_id
        for document_id in DOCUMENT_IDS
        if not (SOURCE_PDF_DIR / f"{document_id}.pdf").is_file()
    ]
    if missing:
        pytest.skip(f"Restore the IPCC corpus first; missing PDFs: {missing}")
    path = tmp_path_factory.mktemp("w01-d2-data") / "documents.csv"
    _select_manifest(SOURCE_MANIFEST, path)
    return path


@pytest.fixture(scope="session")
def runs_800(
    tmp_path_factory: pytest.TempPathFactory, selected_manifest: Path
) -> tuple[IngestRun, IngestRun]:
    root = tmp_path_factory.mktemp("w01-d2-800")
    first = _run_ingest(
        manifest=selected_manifest,
        pdf_dir=SOURCE_PDF_DIR,
        run_dir=root / "first",
        chunk_size=800,
        chunk_overlap=120,
    )
    second = _run_ingest(
        manifest=selected_manifest,
        pdf_dir=SOURCE_PDF_DIR,
        run_dir=root / "second",
        chunk_size=800,
        chunk_overlap=120,
    )
    return first, second


@pytest.fixture(scope="session")
def run_600(
    tmp_path_factory: pytest.TempPathFactory, selected_manifest: Path
) -> IngestRun:
    return _run_ingest(
        manifest=selected_manifest,
        pdf_dir=SOURCE_PDF_DIR,
        run_dir=tmp_path_factory.mktemp("w01-d2-600"),
        chunk_size=600,
        chunk_overlap=80,
    )


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def test_cli_writes_valid_chunks_with_traceable_metadata(
    runs_800: tuple[IngestRun, IngestRun],
) -> None:
    records = _read_jsonl(runs_800[0].output)
    assert records, "Ingest produced no chunks"
    assert {record.get("document_id") for record in records} == set(DOCUMENT_IDS)

    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    for position, record in enumerate(records, 1):
        missing = REQUIRED_CHUNK_FIELDS - record.keys()
        assert not missing, f"Chunk {position} is missing fields: {sorted(missing)}"
        assert isinstance(record["chunk_id"], str) and record["chunk_id"].strip()
        assert isinstance(record["page"], int) and record["page"] >= 1
        assert isinstance(record["source_url"], str) and record[
            "source_url"
        ].startswith("https://")
        assert isinstance(record["text"], str) and record["text"].strip()
        assert not re.search(r"[ \t]{2,}", record["text"]), (
            f"Repeated horizontal whitespace in {record['chunk_id']}"
        )
        actual_tokens = len(encoding.encode(record["text"]))
        assert record["token_count"] == actual_tokens
        assert 0 < actual_tokens <= 800

    chunk_ids = [record["chunk_id"] for record in records]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_id values must be unique"


def test_same_configuration_is_byte_for_byte_deterministic(
    runs_800: tuple[IngestRun, IngestRun],
) -> None:
    first, second = runs_800
    first_digest = hashlib.sha256(first.output.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second.output.read_bytes()).hexdigest()
    assert first_digest == second_digest, (
        "Two identical ingest runs produced different chunk files"
    )


def test_both_required_chunking_strategies_are_recorded_and_compared(
    runs_800: tuple[IngestRun, IngestRun], run_600: IngestRun
) -> None:
    chunks_800 = _read_jsonl(runs_800[0].output)
    chunks_600 = _read_jsonl(run_600.output)
    assert len(chunks_600) > len(chunks_800), (
        "The 600/80 strategy should produce more chunks than 800/120 on the same PDFs"
    )

    manifest_800 = _read_json(runs_800[0].manifest)
    manifest_600 = _read_json(run_600.manifest)
    assert manifest_800["parameters"]["chunk_size"] == 800
    assert manifest_800["parameters"]["chunk_overlap"] == 120
    assert manifest_600["parameters"]["chunk_size"] == 600
    assert manifest_600["parameters"]["chunk_overlap"] == 80


def test_configured_overlap_is_present_between_consecutive_chunks(
    run_600: IngestRun,
) -> None:
    records = _read_jsonl(run_600.output)
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    checked = 0
    for previous, current in itertools.pairwise(records):
        same_page = (
            previous["document_id"] == current["document_id"]
            and previous["page"] == current["page"]
        )
        if not same_page or previous["token_count"] < 600:
            continue
        previous_tokens = encoding.encode(previous["text"])
        current_tokens = encoding.encode(current["text"])
        assert previous_tokens[-80:] == current_tokens[:80], (
            f"Missing exact 80-token overlap between {previous['chunk_id']} "
            f"and {current['chunk_id']}"
        )
        checked += 1
    assert checked >= 3, "Could not verify overlap on at least three chunk boundaries"


def test_missing_pdf_is_reported_in_the_ingest_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "documents.csv"
    manifest.write_text(
        "id,title,url,year,collection,verified\n"
        "MISSING_DOC,Missing document,https://example.test/missing.pdf,2026,test,test\n",
        encoding="utf-8",
    )
    run = _run_ingest(
        manifest=manifest,
        pdf_dir=tmp_path / "pdfs",
        run_dir=tmp_path / "output",
        chunk_size=800,
        chunk_overlap=120,
        require_success=False,
    )
    report = _read_json(run.manifest)
    assert "MISSING_DOC" in set(_walk_strings(report)), (
        "The ingest manifest must identify the document whose PDF is missing"
    )


def test_empty_pdf_page_does_not_create_an_empty_chunk(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (pdf_dir / "BLANK.pdf").open("wb") as stream:
        writer.write(stream)

    manifest = tmp_path / "documents.csv"
    manifest.write_text(
        "id,title,url,year,collection,verified\n"
        "BLANK,Blank page,https://example.test/blank.pdf,2026,test,test\n",
        encoding="utf-8",
    )
    run = _run_ingest(
        manifest=manifest,
        pdf_dir=pdf_dir,
        run_dir=tmp_path / "output",
        chunk_size=800,
        chunk_overlap=120,
    )
    assert _read_jsonl(run.output) == []
    _read_json(run.manifest)
