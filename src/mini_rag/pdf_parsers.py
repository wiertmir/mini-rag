import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pymupdf
import rich_struct.log
from pydantic import BaseModel

from mini_rag.models import Manifest

log = rich_struct.log.get_logger()


class PdfPage(BaseModel):
    number: int
    text: str


class PdfDocument(BaseModel):
    id: str
    source_path: str
    pages: list[PdfPage]
    title: str
    collection: str


@dataclass(frozen=True)
class TextExtractionConfig:
    """Page numbers are one-based; skipped section names match exactly.

    A named section ends at the next heading of equal or shallower numbered
    depth. Unnumbered sections end at the next unnumbered heading.
    Regions classified as pictures (including drawing labels), formulas or
    tables are always omitted. skip_elements accepts additional layout labels,
    such as "section-header" or "list-item".
    """

    skip_headers: bool = True
    skip_footers: bool = True
    skip_page_numbers: bool = True
    skip_decorative_markers: bool = True
    skip_contents_pages: bool = True
    skip_captions: bool = False
    skip_footnotes: bool = False
    skip_elements: frozenset[str] = field(default_factory=frozenset)
    skip_sections: tuple[str, ...] = ()
    pages: tuple[int, ...] | None = None
    join_column_and_page_continuations: bool = True


@dataclass
class _TextRegion:
    text: str
    kind: str
    rect: pymupdf.Rect
    bold_start: bool
    decorative: bool


class _TextSpan(TypedDict):
    text: str
    color: int
    size: float
    flags: int


class _TextLine(TypedDict):
    bbox: tuple[float, float, float, float]
    spans: list[_TextSpan]


class _TextBlock(TypedDict):
    type: int
    lines: NotRequired[list[_TextLine]]


class _TextPage(TypedDict):
    blocks: list[_TextBlock]


def _normalize(text: str) -> str:
    return " ".join(text.replace("\x08", "").split())


def _section_depth(text: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+\S", text)
    return len(match[1].split(".")) if match else 1


def normalize_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _read_regions(page: pymupdf.Page) -> list[_TextRegion]:
    page.get_layout()
    if page.layout_information is None:
        raise RuntimeError("PyMuPDF4LLM layout analysis is unavailable.")
    text_page = cast(_TextPage, page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT))
    lines = [
        line
        for block in text_page["blocks"]
        if block["type"] == 0
        for line in block.get("lines", [])
    ]
    if not lines:
        if page.number is None:
            log.warning("PDF page has no native text; OCR is not enabled.")
        else:
            log.warning(
                "PDF page %s has no native text; OCR is not enabled.",
                page.number + 1,
            )
    regions = []
    for x0, y0, x1, y1, kind in page.layout_information:
        if kind in {"picture", "formula", "table"}:
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        selected = sorted(
            (
                line
                for line in lines
                if rect.contains(
                    pymupdf.Point(
                        (line["bbox"][0] + line["bbox"][2]) / 2,
                        (line["bbox"][1] + line["bbox"][3]) / 2,
                    )
                )
            ),
            key=lambda line: (round(line["bbox"][1] / 3), line["bbox"][0]),
        )
        spans = [
            span for line in selected for span in line["spans"] if span["text"].strip()
        ]
        text = _normalize(" ".join(span["text"] for span in spans))
        if not text:
            continue
        decorative = all(span["color"] == 0xFFFFFF for span in spans) and (
            max(span["size"] for span in spans) >= 36
            or (
                text.isdecimal()
                and len(text) <= 2
                and (x1 <= 32 or x0 >= page.rect.width - 32)
            )
        )
        if kind == "text" and re.match(
            r"^(?:FAQ\s+\d+(?:\.\d+)*,\s*)?(?:Figure|Table)\s+\d+(?:\.\d+)*\s*[|:]",
            text,
            flags=re.IGNORECASE,
        ):
            kind = "caption"

        cleared_txt = normalize_text(text)
        regions.append(
            _TextRegion(
                cleared_txt, kind, rect, bool(spans[0]["flags"] & 16), decorative
            )
        )
    return regions


def extract_text(
    pdf_path: Path, config: TextExtractionConfig, manifest: Manifest
) -> PdfDocument:
    """
    Uses native text and layout classification only: no Markdown serialization,
    asset rendering/export, table-cell extraction or explicit drawing traversal.
    Layout classification itself still creates an internal page preview.
    """
    pages: list[PdfPage] = []

    excluded: set[str] = {"picture", "formula", "table"} | set(config.skip_elements)
    excluded.update(
        kind
        for enabled, kind in (
            (config.skip_headers, "page-header"),
            (config.skip_footers, "page-footer"),
            (config.skip_captions, "caption"),
            (config.skip_footnotes, "footnote"),
        )
        if enabled
    )
    skipped_sections: set[str] = {
        _normalize(name).casefold() for name in config.skip_sections
    }

    previous: _TextRegion | None = None
    previous_page = -1
    skipped_depth: int | None = None
    structural = {"title", "section-header", "caption", "footnote", "list-item"}

    with pymupdf.open(pdf_path) as pdf_doc:
        page_numbers: tuple[int, ...] = (
            tuple(range(1, len(pdf_doc) + 1)) if config.pages is None else config.pages
        )
        if any(
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= len(pdf_doc)
            for number in page_numbers
        ):
            raise ValueError(
                f"pages must contain one-based integers from 1 to {len(pdf_doc)}"
            )

        if tuple(sorted(set(page_numbers))) != page_numbers:
            raise ValueError("pages must be unique and in ascending order")

        for page_number in page_numbers:
            allowed_empty_lines = 0

            page = pdf_doc[page_number - 1]
            parts: list[str] = []

            regions: list[_TextRegion] = _read_regions(page)
            log.info("Page: {}, found {} textable regions", page_number, len(regions))

            if config.skip_contents_pages and any(
                region.kind in {"title", "section-header"}
                and region.text.casefold() in {"contents", "table of contents"}
                for region in regions
            ):
                log.debug("Skipping page {}", page_number)
                continue

            for region in regions:
                if region.kind in {"title", "section-header"} and not region.decorative:
                    depth = _section_depth(region.text)
                    if skipped_depth is not None and depth <= skipped_depth:
                        skipped_depth = None
                    if region.text.casefold() in skipped_sections:
                        skipped_depth = depth
                if skipped_depth is not None:
                    log.debug("Skipping region {} because of its depth", region.kind)
                    continue

                if region.kind in excluded:
                    log.debug("Excluding {} region", region.kind)
                    continue

                if config.skip_decorative_markers and region.decorative:
                    log.debug("Skipping decorative {} region", region.kind)
                    continue

                if (
                    config.skip_page_numbers
                    and region.text.isdecimal()
                    and (
                        region.rect.y0 < page.rect.height * 0.06
                        or region.rect.y1 > page.rect.height * 0.94
                    )
                ):
                    log.debug("Skipping page number region")
                    continue

                if previous is not None:
                    boundary = (
                        page_number != previous_page
                        or abs(region.rect.x0 - previous.rect.x0)
                        >= page.rect.width * 0.3
                    )
                    join = (
                        boundary
                        and config.join_column_and_page_continuations
                        and region.kind not in structural
                        and previous.kind not in structural
                        and not region.bold_start
                        and page_number <= previous_page + 1
                    )
                    if not join and allowed_empty_lines > 0:
                        parts.append("")
                        allowed_empty_lines -= 1

                if region.text:
                    parts.append(region.text)
                    allowed_empty_lines = 1
                elif allowed_empty_lines > 0:
                    parts.append(region.text)
                    allowed_empty_lines -= 1
                previous, previous_page = region, page_number

            page = PdfPage(number=page_number, text="\n".join(parts))
            pages.append(page)

    return PdfDocument(
        id=pdf_path.stem,
        source_path=str(pdf_path),
        pages=pages,
        title=manifest.title,
        collection=manifest.collection,
    )
