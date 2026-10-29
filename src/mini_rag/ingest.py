"""Split PDF documents listed in the manifest into chunks."""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import rich_struct.log
from pydantic import BaseModel, Field
from rich_struct import print_obj

from mini_rag.config import HelpFormatter, IngestConfig, load_config
from mini_rag.pdf_parsers import PdfDocument, TextExtractionConfig, extract_text

rich_struct.log.configure(file=True)
log = rich_struct.log.get_logger()


class Manifest(BaseModel):
    id: str = Field(..., description="Id of the whole source document")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Url of the source")
    year: str = Field(..., description="Publish year")
    collection: str = Field(
        ...,
        description="The bigger set document belongs to. Usefull for filtering",
    )


def read_manifest_file(manifest_path: Path) -> list[Manifest]:
    log.info("Reading manifest file {}", manifest_path)
    with open(manifest_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [Manifest(**row) for row in reader]


def parse_pdf_text(doc_id: str, parent_folder: Path) -> PdfDocument:
    pdf_path = parent_folder / (doc_id + ".pdf")
    log.info("Starting to parse {}", pdf_path)
    pdf_extract_config = TextExtractionConfig()
    return extract_text(pdf_path, pdf_extract_config)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.ingest",
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    config = load_config(IngestConfig, argv, parser)

    manifests = read_manifest_file(config.manifest)

    docs = [parse_pdf_text(manifest.id, config.pdf_dir) for manifest in manifests[:2]]

    rich_struct.print_obj(docs[0].pages[0], apply_newlines=True)
    rich_struct.print_obj(docs[0].pages[1], apply_newlines=True)
    rich_struct.print_obj(docs[0].pages[2], apply_newlines=True)
    rich_struct.print_obj(docs[0].pages[3], apply_newlines=True)
    rich_struct.print_obj(docs[0].pages[4], apply_newlines=True)


if __name__ == "__main__":
    main()
