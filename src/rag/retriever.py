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


def run_hybrid_retrieval(
    query: str,
    filters: dict | None = None,
    limit: int = 5,
) -> list[RetrievedChunk]:
    # Still a placeholder. Later this will blend keyword_search
    # with the Vector DB's meaning search.
    return [
        RetrievedChunk(
            text="This is a placeholder chunk.",
            source="placeholder.txt",
            metadata={"note": "fake data"},
            score=0.0,
        )
    ]