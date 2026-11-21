"""Portable filesystem names; API model identifiers must stay unchanged."""

import re
from pathlib import Path


def safe_filename(name: str) -> str:
    """Replace Windows-invalid characters and protect reserved device names."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(" .") or "_"
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]", name.split(".")[0], re.IGNORECASE):
        name = "_" + name
    return name


def portable_path(path: Path) -> Path:
    """Normalize components, preserving the native drive/root and relative parents."""
    return Path(
        *(part if part in (path.anchor, "..") else safe_filename(part) for part in path.parts)
    )
