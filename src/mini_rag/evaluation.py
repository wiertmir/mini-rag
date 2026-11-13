from collections.abc import Iterator


def is_hit(expected_documents: frozenset[str], retrieved_document_ids: Iterator[str], answerable: bool) -> bool | None:
    if not answerable:
        return None

    for doc in retrieved_document_ids:
        if doc in expected_documents:
            return True

    return False
