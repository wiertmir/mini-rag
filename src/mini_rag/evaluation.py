from collections.abc import Iterable
from collections.abc import Set as AbstractSet


def is_hit(
    expected_documents: AbstractSet[str], retrieved_document_ids: Iterable[str], answerable: bool
) -> bool | None:
    if not answerable:
        return None

    for doc in retrieved_document_ids:
        if doc in expected_documents:
            return True

    return False
