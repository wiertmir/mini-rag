import rich_struct.log
import tiktoken

log = rich_struct.log.get_logger()

CHUNK_MODEL = "text-embedding-3-small"

encoding = tiktoken.encoding_for_model(CHUNK_MODEL)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    log.debug("Encoding text of {} characters length", len(text))
    tokens = encoding.encode(text)
    step = chunk_size - chunk_overlap
    log.debug(
        "Found {} tokens. Chunking into {} size and {} overlap",
        len(tokens),
        chunk_size,
        chunk_overlap,
    )

    result: list[tuple[int, str]] = []

    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk = encoding.decode(chunk_tokens)
        result.append((len(chunk_tokens), chunk))

    return result
