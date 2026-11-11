from pathlib import Path


class ChunkReadError(ValueError):
    def __init__(self, path: Path, line_no: int, reason: str) -> None:
        self.path = path
        self.line_no = line_no
        super().__init__(f"{path}: line {line_no}: {reason}")


class ChunkParseError(ChunkReadError):
    """A JSONL line cannot be parsed as JSON."""


class ChunkValidationError(ChunkReadError):
    """A JSONL record does not satisfy the Chunk schema."""


class ManifestError(Exception):
    """There is something wrong with Manifest"""


class InconsistentManifestError(ManifestError):
    """There is something wrong with Manifest"""


class EmptyInputData(Exception):
    """There is no input data to produce any output"""
