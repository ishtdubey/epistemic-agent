from functools import lru_cache

from src.rag.retriever import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load the model once, the first time it is needed."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME)


def _default_scorer(query: str, texts: list[str]) -> list[float]:
    """Score how well each text answers the query, using the real model."""
    model = _load_model()
    scores = model.predict([(query, text) for text in texts])
    return [float(s) for s in scores]


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    limit: int = 5,
    scorer=None,
) -> list[RetrievedChunk]:
    """Re-sort chunks so the ones that best answer the query come first.
    `scorer` is only for tests, so they don't need to download the model."""
    if not chunks:
        return []

    scorer = scorer or _default_scorer
    scores = scorer(query, [c.text for c in chunks])

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:limit]
    return [
        RetrievedChunk(
            text=c.text,
            source=c.source,
            metadata=c.metadata,
            score=float(s),
        )
        for c, s in ranked
    ]