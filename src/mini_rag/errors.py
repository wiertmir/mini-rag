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


class EmptyInputData(ValueError):
    """There is no input data to produce any output"""


class EmptyIndex(ValueError):
    """The index is empty (no chunks)"""


class DimensionMismatchError(ValueError):
    """e.g. a 3-D question against a 2-D index"""


class IndexAndChunksNotSyncError(ValueError):
    """Index got broken and doesn't match size of chunks"""


class StaleIndexError(ValueError):
    """Chunks changed, index file got stale"""


class DocumentNotFoundError(ValueError):
    """The chunks file has no chunks for the requested document"""


class MissingParsedOutputError(Exception):
    """Raised when generation returns no parsed answer, including refusal or incomplete output."""

    def __init__(
        self,
        model: str,
        response_id: str,
        status: str | None,
        refusal: str | None = None,
        incomplete_reason: str | None = None,
    ) -> None:
        self.model = model
        self.response_id = response_id
        self.status = status
        self.refusal = refusal
        self.incomplete_reason = incomplete_reason
        details = f"model={model}, response_id={response_id}, status={status}"
        if incomplete_reason:
            details += f", incomplete_reason={incomplete_reason}"
        if refusal:
            reason = f"Safety refusal: {refusal}"
        elif status == "incomplete" or incomplete_reason:
            reason = "Generation incomplete"
        else:
            reason = "No parsed output returned"
        super().__init__(f"{reason} ({details})")


class NotInitializedError(Exception):
    """Raised when system element is not initialized"""
