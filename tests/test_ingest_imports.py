import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest


@pytest.mark.parametrize(
    "first,second",
    [
        ("mini_rag.ingest", "mini_rag.pdf_parsers"),
        ("mini_rag.pdf_parsers", "mini_rag.ingest"),
    ],
)
def test_modules_import_in_either_order(first: str, second: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import {first}; import {second}; "
                "from mini_rag.ingest import Manifest as IngestManifest; "
                "from mini_rag.models import Manifest; "
                "assert IngestManifest is Manifest"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ingest_passes_manifest_to_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mini_rag import ingest
    from mini_rag.config import IngestConfig
    from mini_rag.models import Manifest
    from mini_rag.pdf_parsers import PdfDocument, TextExtractionConfig

    manifest_path = tmp_path / "documents.csv"
    manifest_path.write_text(
        "id,title,url,year,collection\n"
        "sample,Sample title,https://example.com/sample,2026,Sample collection\n",
        encoding="utf-8",
    )
    config = IngestConfig.model_construct(manifest=manifest_path, pdf_dir=tmp_path)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"mock PDF")
    received: list[Manifest] = []

    def extract(
        pdf_path: Path, config: TextExtractionConfig, manifest: Manifest
    ) -> PdfDocument:
        assert pdf_path == tmp_path / "sample.pdf"
        assert isinstance(config, TextExtractionConfig)
        received.append(manifest)
        return PdfDocument(
            id=manifest.id,
            source_path=str(pdf_path),
            pages=[],
            title=manifest.title,
            collection=manifest.collection,
        )

    monkeypatch.setattr(ingest, "extract_text", extract)
    manifests = ingest.read_manifest_file(manifest_path)
    chunks, stats = ingest.process_document(manifests[0], config)

    assert chunks == []
    assert stats["sha_256"] == ingest.calculate_file_sha(pdf_path)
    assert stats["status"] == "ok"
    assert received == [
        Manifest(
            id="sample",
            title="Sample title",
            url="https://example.com/sample",
            year="2026",
            collection="Sample collection",
        )
    ]


def test_extract_text_preserves_manifest_metadata(tmp_path: Path) -> None:
    from mini_rag.models import Manifest
    from mini_rag.pdf_parsers import TextExtractionConfig, extract_text

    pdf_path = tmp_path / "sample.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(pdf_path)
    manifest = Manifest(
        id="sample",
        title="Sample title",
        url="https://example.com/sample",
        year="2026",
        collection="Sample collection",
    )

    document = extract_text(pdf_path, TextExtractionConfig(pages=()), manifest)

    assert document.id == manifest.id
    assert document.title == manifest.title
    assert document.collection == manifest.collection
    assert document.source_path == str(pdf_path)
    assert document.pages == []
