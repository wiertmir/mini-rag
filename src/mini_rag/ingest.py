"""Split PDF documents listed in the manifest into chunks."""

import argparse
import csv
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import pymupdf
import rich_struct.log

from mini_rag.chunk import CHUNK_MODEL, Block, chunk_blocks
from mini_rag.config import HelpFormatter, IngestConfig, load_config
from mini_rag.hashing import calculate_file_sha
from mini_rag.models import (
    Chunk,
    IngestManifest,
    IngestManifestDocument,
    IngestManifestParameters,
    IngestManifestSummary,
    Manifest,
)
from mini_rag.pdf_parsers import PdfDocument, TextExtractionConfig, extract_text

log = rich_struct.log.get_logger()


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
) -> tuple[list[Chunk], IngestManifestDocument]:
    doc = parse_pdf_text(manifest, config.pdf_dir)
    chunks: list[Chunk] = []
    log.info("Chunking file {}", doc.source_path)
    blocks = [
        Block(block.text, page.number, block.heading, block.joined)
        for page in doc.pages
        for block in page.blocks
    ]
    text_chunks = chunk_blocks(
        blocks,
        config.chunk_size,
        config.chunk_overlap,
        config.min_chunk_tokens,
        skip_patterns=config.skip_patterns,
        box_pattern=config.box_pattern,
    )
    # chunk_id numbers chunks per start page, so IDs stay stable and readable.
    per_page: dict[int, int] = {}
    for text_chunk in text_chunks:
        chunk_no = per_page[text_chunk.page_start] = per_page.get(text_chunk.page_start, 0) + 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.id}_p{text_chunk.page_start:03d}_c{chunk_no:03d}",
                document_id=doc.id,
                title=doc.title,
                collection=doc.collection,
                source_path=Path(doc.source_path),
                source_url=manifest.url,
                page=text_chunk.page_start,
                page_end=text_chunk.page_end,
                text=text_chunk.text,
                token_count=text_chunk.token_count,
            )
        )
    stats = IngestManifestDocument(
        document_id=doc.id,
        source_path=doc.source_path,
        sha_256=calculate_file_sha(Path(doc.source_path)),
        pages=len(doc.pages),
        chunks=len(chunks),
        status="ok",
        error=None,
    )
    return chunks, stats


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.ingest",
        description=__doc__,
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )
    config = load_config(IngestConfig, argv, parser)
    rich_struct.log.configure(file=config.log_folder)

    run_ingest(config)


def run_ingest(config: IngestConfig) -> None:
    errors: list[str] = []
    manifests = read_manifest_file(config.manifest)

    chunk_result: list[Chunk] = []

    docs_no = 0
    total_pages = 0

    doc_stats: list[IngestManifestDocument] = []

    with ProcessPoolExecutor(
        max_workers=max(1, min(4, len(manifests))),
        mp_context=get_context("spawn"),
        initializer=partial(rich_struct.log.configure, file=config.log_folder),
    ) as executor:
        futures = [executor.submit(process_document, manifest, config) for manifest in manifests]
        for manifest, future in zip(manifests, futures, strict=True):
            docs_no += 1
            try:
                chunks, stats = future.result()
            except (OSError, pymupdf.FileNotFoundError, pymupdf.FileDataError) as exc:
                error = f"{manifest.id}: {type(exc).__name__}: {exc}"
                log.error("Failed to process {}", error)
                errors.append(error)
                stats = IngestManifestDocument(
                    document_id=manifest.id,
                    source_path=str(config.pdf_dir / f"{manifest.id}.pdf"),
                    sha_256=None,
                    pages=0,
                    chunks=0,
                    status="error",
                    error=error,
                )
            else:
                chunk_result.extend(chunks)
                total_pages += stats.pages
            doc_stats.append(stats)

    save_ingest_manifest(config, docs_no, total_pages, chunk_result, doc_stats, errors)
    if errors:
        raise SystemExit(1)


def load_ingest_manifest(manifest_path: Path) -> IngestManifest:
    log.info("Reading ingest manifest from {}", manifest_path)
    return IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def save_ingest_manifest(
    config: IngestConfig,
    docs: int,
    pages: int,
    chunks: list[Chunk],
    doc_stats: list[IngestManifestDocument],
    errors: list[str],
) -> None:
    save_chunk_results(config, chunks)
    manifest_out_path = (
        config.output_dir / f"ingest_manifest_{config.chunk_size}_{config.chunk_overlap}.json"
    )
    log.info("Writing ingest manifest to {}", manifest_out_path)

    manifest = IngestManifest(
        parameters=IngestManifestParameters(
            tokenizer=CHUNK_MODEL,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            strategy="sections",
            min_chunk_tokens=config.min_chunk_tokens,
            skip_patterns=config.skip_patterns,
            box_pattern=config.box_pattern,
        ),
        summary=IngestManifestSummary(
            documents=docs,
            pages=pages,
            chunks=len(chunks),
            errors=len(errors),
        ),
        documents=doc_stats,
        errors=errors,
    )

    manifest_out_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def save_chunk_results(config: IngestConfig, chunk_result: list[Chunk]) -> None:
    output_path = config.output_dir / f"chunks_{config.chunk_size}_{config.chunk_overlap}.jsonl"
    log.info("Writing final output to {}", output_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as chunks_output:
        chunk_iter = (f"{chunk.model_dump_json()}\n" for chunk in chunk_result)
        chunks_output.writelines(chunk_iter)


if __name__ == "__main__":
    main()
