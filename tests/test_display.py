from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_rag.config import DisplayConfig
from mini_rag.display import (
    ChunkRow,
    build_table,
    load_document_chunks_async,
    main,
    overlap_length,
    select_pages,
    split_chunks,
)
from mini_rag.errors import DocumentNotFoundError
from mini_rag.models import Chunk, EmbeddingManifest
from mini_rag.pages import PageRange, parse_page_ranges
from mini_rag.storage import calculate_file_sha


def make_chunk(text: str, page: int = 1, number: int = 1, document_id: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}_p{page:03d}_c{number:03d}",
        document_id=document_id,
        title="Doc title",
        collection="test",
        source_path=Path("doc.pdf"),
        source_url="https://example.test/doc.pdf",
        page=page,
        page_end=page,
        text=text,
        token_count=len(text.split()),
    )


# --- page ranges --------------------------------------------------------------


def test_parse_page_ranges_closed_single_and_open() -> None:
    assert parse_page_ranges("3-7,9,14-") == (
        PageRange(3, 7),
        PageRange(9, 9),
        PageRange(14, None),
    )


def test_parse_page_ranges_leading_dash_starts_at_page_one() -> None:
    assert parse_page_ranges("-3") == (PageRange(1, 3),)


def test_parse_page_ranges_tolerates_spaces() -> None:
    assert parse_page_ranges(" 2 - 4 , 6 ") == (PageRange(2, 4), PageRange(6, 6))


@pytest.mark.parametrize("spec", ["", "-", "0", "7-3", "a", "3--5", "1,,2", "2-x"])
def test_parse_page_ranges_rejects_invalid_spec(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_page_ranges(spec)


def test_page_range_overlaps_page_span() -> None:
    assert PageRange(3, 5).overlaps(1, 3)
    assert PageRange(3, 5).overlaps(5, 9)
    assert PageRange(3, 5).overlaps(1, 9)
    assert not PageRange(3, 5).overlaps(6, 9)
    assert not PageRange(3, 5).overlaps(1, 2)
    assert PageRange(3, None).overlaps(100, 101)


def test_display_config_parses_pages_once() -> None:
    assert DisplayConfig(document_id="doc", pages="2-3").page_ranges == (PageRange(2, 3),)
    assert DisplayConfig(document_id="doc").page_ranges is None


def test_page_range_includes_bounds() -> None:
    assert PageRange(3, 5).includes(3)
    assert PageRange(3, 5).includes(5)
    assert not PageRange(3, 5).includes(6)
    assert PageRange(3, None).includes(10_000)
    assert not PageRange(3, None).includes(2)


def test_display_config_rejects_invalid_pages_at_load_time() -> None:
    with pytest.raises(ValidationError):
        DisplayConfig(document_id="doc", pages="7-3")


# --- overlap detection --------------------------------------------------------

SHARED_1 = "first shared sentence here."
SHARED_2 = "second shared sentence here."


def test_overlap_length_finds_longest_suffix_prefix() -> None:
    assert overlap_length(f"head {SHARED_1}", f"{SHARED_1} tail") == len(SHARED_1)


def test_overlap_length_zero_without_shared_text() -> None:
    assert overlap_length("abc", "xyz") == 0


def test_overlap_length_ignores_coincidental_short_match() -> None:
    # A chunk ending with "." followed by one starting with "." is not an overlap.
    assert overlap_length("ends with a dot.", ". starts with a dot") == 0


def test_overlap_length_whole_second_text_contained() -> None:
    assert overlap_length(f"alpha {SHARED_1}", SHARED_1) == len(SHARED_1)


def test_split_chunks_separates_unique_and_shared_text() -> None:
    chunks = [
        make_chunk(f"aaa {SHARED_1}", number=1),
        make_chunk(f"{SHARED_1} bbb {SHARED_2}", number=2),
        make_chunk(f"{SHARED_2} ccc", number=3),
    ]

    assert split_chunks(chunks) == [
        [
            ChunkRow("doc_p001_c001\np. 1", "aaa ", is_overlap=False),
            ChunkRow("c001 ↔ c002", SHARED_1, is_overlap=True),
            ChunkRow("doc_p001_c002\np. 1", " bbb ", is_overlap=False),
            ChunkRow("c002 ↔ c003", SHARED_2, is_overlap=True),
            ChunkRow("doc_p001_c003\np. 1", " ccc", is_overlap=False),
        ]
    ]


def test_split_chunks_rebuilds_original_text() -> None:
    chunks = [make_chunk(f"aaa {SHARED_1}", number=1), make_chunk(f"{SHARED_1} bbb", number=2)]
    (rows,) = split_chunks(chunks)

    assert "".join(row.text for row in rows) == f"aaa {SHARED_1} bbb"


def test_split_chunks_starts_new_group_without_shared_text() -> None:
    chunks = [make_chunk("section one", number=1), make_chunk("section two", number=2)]

    assert [len(group) for group in split_chunks(chunks)] == [1, 1]


def test_split_chunks_entirely_overlapped_chunk_is_empty() -> None:
    chunks = [make_chunk(f"aaa {SHARED_1}", number=1), make_chunk(SHARED_1, number=2)]
    (rows,) = split_chunks(chunks)

    assert rows[-1] == ChunkRow("doc_p001_c002\np. 1", "", is_overlap=False)


def test_split_chunks_label_shows_page_range() -> None:
    chunk = make_chunk("spans pages", page=3).model_copy(update={"page_end": 5})

    assert split_chunks([chunk])[0][0].label == "doc_p003_c001\np. 3-5"


def test_split_chunks_single_chunk_has_no_overlap_row() -> None:
    assert split_chunks([make_chunk("only")]) == [
        [ChunkRow("doc_p001_c001\np. 1", "only", is_overlap=False)]
    ]


# --- table --------------------------------------------------------------------


def test_build_table_adds_section_between_groups() -> None:
    chunks = [make_chunk("p1", page=1), make_chunk("p2", page=2), make_chunk("p3", page=3)]
    table = build_table(chunks, "doc: Doc title")

    assert table.row_count == 3
    assert [row.end_section for row in table.rows] == [True, True, False]


def test_build_table_caption_mentions_count_and_selection() -> None:
    table = build_table([make_chunk("x")], "t", pages="1-2")

    assert "1 chunks on pages 1-2" in str(table.caption)


def test_build_table_accepts_empty_selection() -> None:
    assert build_table([], "t").row_count == 0


def test_select_pages_keeps_chunks_overlapping_the_selection() -> None:
    multi = make_chunk("multi", page=2, number=1).model_copy(update={"page_end": 4})
    chunks = [make_chunk("one", page=1), multi, make_chunk("five", page=5)]

    assert select_pages(chunks, parse_page_ranges("3")) == [multi]
    assert select_pages(chunks, parse_page_ranges("5-")) == [chunks[2]]


# --- loading ------------------------------------------------------------------


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")


@pytest.fixture
def chunks_file(tmp_path: Path) -> Path:
    path = tmp_path / "chunks.jsonl"
    write_chunks(
        path,
        [
            make_chunk("first", page=1, document_id="doc"),
            make_chunk("other", page=1, document_id="other"),
            make_chunk("second", page=2, document_id="doc"),
        ],
    )
    return path


@pytest.fixture
def index_dir(tmp_path: Path, chunks_file: Path) -> Path:
    index = tmp_path / "index"
    index.mkdir()
    manifest = EmbeddingManifest(
        model="fake",
        dimension=2,
        records=3,
        chunks_path=chunks_file,
        chunks_sha256=calculate_file_sha(chunks_file),
    )
    (index / "embedding_manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return index


async def test_load_document_chunks_uses_manifest_chunks_path(index_dir: Path) -> None:
    config = DisplayConfig(document_id="doc", index_dir=index_dir)

    chunks = await load_document_chunks_async(config)

    assert [c.text for c in chunks] == ["first", "second"]


async def test_load_document_chunks_prefers_explicit_chunks_file(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.jsonl"
    write_chunks(explicit, [make_chunk("explicit", document_id="doc")])
    config = DisplayConfig(document_id="doc", index_dir=tmp_path / "missing", chunks=explicit)

    chunks = await load_document_chunks_async(config)

    assert [c.text for c in chunks] == ["explicit"]


async def test_load_document_chunks_unknown_document(index_dir: Path) -> None:
    config = DisplayConfig(document_id="nope", index_dir=index_dir)

    with pytest.raises(DocumentNotFoundError):
        await load_document_chunks_async(config)


# --- CLI ----------------------------------------------------------------------


def test_main_filters_pages(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COLUMNS", "200")

    main(["--document-id", "doc", "--index-dir", str(index_dir), "--pages=2"])

    out = capsys.readouterr().out
    assert "doc_p002_c001" in out
    assert "doc_p001_c001" not in out
    assert "doc: Doc title" in out


def test_main_accepts_leading_dash_range_with_equals(
    index_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COLUMNS", "200")

    main(["--document-id", "doc", "--index-dir", str(index_dir), "--pages=-1"])

    out = capsys.readouterr().out
    assert "doc_p001_c001" in out
    assert "doc_p002_c001" not in out


def test_main_reports_unknown_document_without_traceback(
    index_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--document-id", "nope", "--index-dir", str(index_dir)])

    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "DocumentNotFoundError" in err
    assert "Traceback" not in err
