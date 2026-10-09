import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.rag.chunking import chunk_file

# Temporary: folder with .txt files. Replaced when the Vector DB is ready.
DATA_DIR = Path("data/sample")


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


def hybrid_pipeline(
    query: str,
    corpus: list[dict],
    vector_results: list[RetrievedChunk],
    limit: int = 5,
    rerank_fn=None,
) -> list[RetrievedChunk]:
    """Run the full search: keyword search + meaning search -> blend -> rerank.
    corpus: all chunks as dicts (text, source, metadata), for keyword search.
    vector_results: chunks already found by the Vector DB's meaning search.
    rerank_fn: only for tests, so they don't need the big model."""
    keyword_results = keyword_search(query, corpus, limit=limit * 3)
    blended = blend_results(keyword_results, vector_results, limit=limit * 3)

    if rerank_fn is None:
        from src.rag.ranker import rerank as rerank_fn  # imported here to avoid a circular import

    return rerank_fn(query, blended, limit=limit)


# ---------------------------------------------------------------------------
# Stand-ins. Replace these two functions when the Vector DB is ready.
# ---------------------------------------------------------------------------

def load_corpus(data_dir=None) -> list[dict]:
    """TEMPORARY: read every .txt file in the data folder and cut it into chunks.
    Later: replace with the Vector DB's 'give me all chunks' function."""
    folder = Path(data_dir or DATA_DIR)
    corpus = []
    for path in sorted(folder.glob("*.txt")):
        corpus.extend(chunk_file(str(path)))
    return corpus


def vector_search(query: str, filters: dict | None = None, limit: int = 5) -> list[RetrievedChunk]:
    """TEMPORARY: returns nothing, so only keyword search is used for now.
    Later: call the Vector DB's meaning search and return RetrievedChunk objects."""
    return []


# ---------------------------------------------------------------------------
# The functions your teammates call
# ---------------------------------------------------------------------------

def _matches_filters(chunk: dict, filters: dict | None) -> bool:
    """True if the chunk's metadata has every key/value in filters."""
    if not filters:
        return True
    return all(chunk["metadata"].get(key) == value for key, value in filters.items())


def _safe_rerank(query: str, chunks: list[RetrievedChunk], limit: int = 5) -> list[RetrievedChunk]:
    """Rerank with the real model. If it can't load (e.g. torch crashes), keep the blended order."""
    try:
        from src.rag.ranker import rerank

        return rerank(query, chunks, limit=limit)
    except (OSError, ImportError):
        print("Warning: reranker unavailable, using the blended order instead.")
        return chunks[:limit]


def run_hybrid_retrieval(
    query: str,
    filters: dict | None = None,
    limit: int = 5,
) -> list[RetrievedChunk]:
    """Full search: keyword + meaning search, blended, then reranked."""
    corpus = [c for c in load_corpus() if _matches_filters(c, filters)]
    vector_results = vector_search(query, filters, limit * 3)
    return hybrid_pipeline(query, corpus, vector_results, limit, rerank_fn=_safe_rerank)


def run_direct_retrieval(
    query: str,
    filters: dict | None = None,
    limit: int = 5,
) -> list[RetrievedChunk]:
    """Lighter version for simple factual questions: one pass, no reranking."""
    corpus = [c for c in load_corpus() if _matches_filters(c, filters)]
    keyword_results = keyword_search(query, corpus, limit=limit * 3)
    vector_results = vector_search(query, filters, limit * 3)
    return blend_results(keyword_results, vector_results, limit=limit)