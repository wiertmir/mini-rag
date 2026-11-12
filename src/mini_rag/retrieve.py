from collections.abc import Sequence

import numpy as np
import rich_struct.log
from numpy.typing import NDArray

from mini_rag.embed import normalize
from mini_rag.errors import (
    DimensionMismatchError,
    EmptyIndex,
    IndexAndChunksNotSyncError,
)
from mini_rag.models import Chunk, SearchResult

logger = rich_struct.log.get_logger()


def search(
    query_vector: Sequence[float],
    matrix: NDArray[np.float32],
    chunks: Sequence[Chunk],
    top_k: int,
) -> list[SearchResult]:
    d = len(query_vector)
    n = matrix.shape[0]

    if n == 0:
        raise EmptyIndex()

    if d != matrix.shape[1]:
        raise DimensionMismatchError(
            f"Question vector's dimension is {d}, where the index dimesnion is {matrix.shape[1]}"
        )

    if top_k < 1:
        raise ValueError("`top_k` must be at least 1")

    if matrix.shape[0] != len(chunks):
        raise IndexAndChunksNotSyncError(
            f"Index has {matrix.shape[0]} rows, where chunks have {len(chunks)} rows"
        )

    logger.info(
        "Searching for top {} best scoring chunks among {} index rows",
        top_k,
        matrix.shape[0],
    )

    norm_query_vector = normalize(np.array(query_vector, dtype=np.float32))
    scores = matrix @ norm_query_vector
    sort_indices = np.argsort(scores, descending=True, kind="stable")

    return [
        SearchResult(rank=rank, score=float(scores[i]), chunk=chunks[i])
        for (rank, i) in enumerate(sort_indices[:top_k], start=1)
    ]
