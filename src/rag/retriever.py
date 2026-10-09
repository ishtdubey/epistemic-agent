import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class RetrievedChunk:
    text: str
    source: str
    metadata: dict
    score: float


def _tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into words."""
    return re.findall(r"\w+", text.lower())


def keyword_search(query: str, chunks: list[dict], limit: int = 5) -> list[RetrievedChunk]:
    """Rank chunks by how well their words match the query (BM25).
    Each chunk is a dict with keys: text, source, metadata."""
    if not chunks:
        return []

    tokenized_chunks = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:limit]
    return [
        RetrievedChunk(
            text=c["text"],
            source=c["source"],
            metadata=c["metadata"],
            score=float(s),
        )
        for c, s in ranked
        if s > 0
    ]


def blend_results(
    keyword_results: list[RetrievedChunk],
    vector_results: list[RetrievedChunk],
    limit: int = 5,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one using Reciprocal Rank Fusion.
    A chunk earns 1/(k + position) points from each list it appears in."""
    points: dict[tuple, float] = {}
    chunk_by_key: dict[tuple, RetrievedChunk] = {}

    for results in (keyword_results, vector_results):
        for position, chunk in enumerate(results, start=1):
            key = (chunk.source, chunk.text)
            points[key] = points.get(key, 0.0) + 1.0 / (k + position)
            chunk_by_key.setdefault(key, chunk)

    best_keys = sorted(points, key=points.get, reverse=True)[:limit]
    return [
        RetrievedChunk(
            text=chunk_by_key[key].text,
            source=chunk_by_key[key].source,
            metadata=chunk_by_key[key].metadata,
            score=points[key],
        )
        for key in best_keys
    ]


def run_hybrid_retrieval(
    query: str,
    filters: dict | None = None,
    limit: int = 5,
) -> list[RetrievedChunk]:
    # Still a placeholder. Later this will call keyword_search and the
    # Vector DB's meaning search, then blend_results.
    return [
        RetrievedChunk(
            text="This is a placeholder chunk.",
            source="placeholder.txt",
            metadata={"note": "fake data"},
            score=0.0,
        )
    ]