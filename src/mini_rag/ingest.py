"""Split PDF documents listed in the manifest into chunks."""

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import TypedDict

import pymupdf
import rich_struct.log

from mini_rag.chunk import CHUNK_MODEL, chunk_text
from mini_rag.config import HelpFormatter, IngestConfig, load_config
from mini_rag.models import Chunk, Manifest
from mini_rag.pdf_parsers import PdfDocument, TextExtractionConfig, extract_text

rich_struct.log.configure(file=True)
log = rich_struct.log.get_logger()


class DocumentStats(TypedDict):
    document_id: str
    source_path: str
    sha_256: str | None
    pages: int
    chunks: int
    status: str
    error: str | None


def read_manifest_file(manifest_path: Path) -> list[Manifest]:
    log.info("Reading manifest file {}", manifest_path)
    with open(manifest_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [Manifest(**row) for row in reader]


def parse_pdf_text(manifest: Manifest, parent_folder: Path) -> PdfDocument:
    pdf_path = parent_folder / (manifest.id + ".pdf")
    log.info("Starting to parse {}", pdf_path)
    pdf_extract_config = TextExtractionConfig()
    return extract_text(pdf_path, pdf_extract_config, manifest)


def process_document(
    manifest: Manifest, config: IngestConfig
) -> tuple[list[Chunk], DocumentStats]:
    doc = parse_pdf_text(manifest, config.pdf_dir)
    chunks: list[Chunk] = []
    log.info("Chunking file {}", doc.source_path)
    for page_no, page in enumerate(doc.pages, 1):
        for chunk_no, (chunk_tokens, chunk) in enumerate(
            chunk_text(page.text, config.chunk_size, config.chunk_overlap), 1
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.id}_p{page_no:03d}_c{chunk_no:03d}",
                    document_id=doc.id,
                    title=doc.title,
                    collection=doc.collection,
                    source_path=doc.source_path,
                    source_url=manifest.url,
                    page=page_no,
                    text=chunk,
                    token_count=chunk_tokens,
                )
            )
    stats: DocumentStats = {
        "document_id": doc.id,
        "source_path": doc.source_path,
        "sha_256": calculate_file_sha(Path(doc.source_path)),
        "pages": len(doc.pages),
        "chunks": len(chunks),
        "status": "ok",
        "error": None,
    }
    return chunks, stats


def main(argv: Sequence[str] | None = None) -> None:
    errors: list[str] = []

    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.ingest",
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    config = load_config(IngestConfig, argv, parser)

    manifests = read_manifest_file(config.manifest)

    chunk_result: list[Chunk] = []

    docs_no = 0
    total_pages = 0
    total_chunks = 0

    doc_stats = []

    with ProcessPoolExecutor(
        max_workers=max(1, min(4, len(manifests))),
        mp_context=get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(process_document, manifest, config)
            for manifest in manifests
        ]
        for manifest, future in zip(manifests, futures, strict=True):
            docs_no += 1
            try:
                chunks, stats = future.result()
            except (OSError, pymupdf.FileNotFoundError, pymupdf.FileDataError) as exc:
                error = f"{manifest.id}: {type(exc).__name__}: {exc}"
                log.error("Failed to process {}", error)
                errors.append(error)
                stats = {
                    "document_id": manifest.id,
                    "source_path": str(config.pdf_dir / f"{manifest.id}.pdf"),
                    "sha_256": None,
                    "pages": 0,
                    "chunks": 0,
                    "status": "error",
                    "error": error,
                }
            else:
                chunk_result.extend(chunks)
                total_pages += stats["pages"]
                total_chunks += stats["chunks"]
            doc_stats.append(stats)

    save_chunk_results(config, chunk_result)
    save_ingest_manifest(config, docs_no, total_pages, total_chunks, doc_stats, errors)
    if errors:
        raise SystemExit(1)


def calculate_file_sha(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def save_ingest_manifest(
    config: IngestConfig,
    docs: int,
    pages: int,
    chunks: int,
    doc_stats: list[DocumentStats],
    errors: list[str],
) -> None:
    manifest_out_path = config.output.parent / "ingest_manifest.json"
    log.info("Writing ingest manifest to {}", manifest_out_path)

    manifest = {
        "parameters": {
            "tokenizer": CHUNK_MODEL,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        "summary": {
            "documents": docs,
            "pages": pages,
            "chunks": chunks,
            "errors": len(errors),
        },
        "documents": doc_stats,
        "errors": errors,
    }

    with open(manifest_out_path, "w", encoding="utf-8") as manifest_out:
        json.dump(manifest, manifest_out, indent=2)


def save_chunk_results(config: IngestConfig, chunk_result: list[Chunk]) -> None:
    log.info("Writing final output to {}", config.output)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output, "w", encoding="utf-8") as chunks_output:
        chunk_iter = (f"{chunk.model_dump_json()}\n" for chunk in chunk_result)
        chunks_output.writelines(chunk_iter)


if __name__ == "__main__":
    main()
