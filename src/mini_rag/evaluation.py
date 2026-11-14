from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet

from mini_rag.models import EvaluationRecord


def is_hit(
    expected_documents: AbstractSet[str], retrieved_document_ids: Iterable[str], answerable: bool
) -> bool | None:
    if not answerable:
        return None

    for doc in retrieved_document_ids:
        if doc in expected_documents:
            return True

    return False


def summarize_evaluation(records: Sequence[EvaluationRecord]) -> tuple[int, int, list[str]]:
    answerable_count = 0
    hit_count = 0
    missed_ids: list[str] = []

    for record in records:
        if record.hit is None:
            continue

        answerable_count += 1
        if record.hit:
            hit_count += 1
        else:
            missed_ids.append(record.id)

    return answerable_count, hit_count, missed_ids
