import hashlib
from pathlib import Path


def calculate_file_sha(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def calculate_text_sha(text: str) -> str:
    text_bytes = text.encode("utf-8")
    sha256_hash = hashlib.sha256(text_bytes)
    return sha256_hash.hexdigest()
