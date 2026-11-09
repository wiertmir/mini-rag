import json
import subprocess
import sys
from pathlib import Path

from mini_rag.config import IngestConfig
from mini_rag.ingest import save_ingest_manifest
from mini_rag.models import Chunk


def test_save_ingest_manifest_writes_both_files_for_each_configuration(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nested" / "index"
    chunk = Chunk(
        chunk_id="sample_p001_c001",
        document_id="sample",
        title="Sample",
        collection="test",
        source_path="sample.pdf",
        source_url="https://example.test/sample.pdf",
        page=1,
        text="Sample text",
        token_count=2,
    )
    expected_files: set[str] = set()
    for chunk_size, chunk_overlap in [(800, 120), (600, 80), (800, 0)]:
        config = IngestConfig.model_construct(
            output_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        save_ingest_manifest(config, 1, 1, [chunk], [], [])

        chunks_name = f"chunks_{chunk_size}_{chunk_overlap}.jsonl"
        manifest_name = f"ingest_manifest_{chunk_size}_{chunk_overlap}.json"
        expected_files.update((chunks_name, manifest_name))
        assert {path.name for path in output_dir.iterdir()} == expected_files
        assert (output_dir / chunks_name).read_text(encoding="utf-8") == (
            chunk.model_dump_json() + "\n"
        )
        manifest = json.loads((output_dir / manifest_name).read_text(encoding="utf-8"))
        assert manifest["parameters"]["chunk_size"] == chunk_size
        assert manifest["parameters"]["chunk_overlap"] == chunk_overlap
        assert manifest["summary"] == {
            "documents": 1,
            "pages": 1,
            "chunks": 1,
            "errors": 0,
        }
        assert manifest["documents"] == []
        assert manifest["errors"] == []


def test_cli_rejects_old_output_option(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mini_rag.ingest",
            "--output",
            str(tmp_path / "chunks.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --output" in result.stderr
    assert not list(tmp_path.iterdir())
