from src.rag.retriever import RetrievedChunk, hybrid_pipeline


def no_model_rerank(query, chunks, limit=5):
    """Pretend reranker so the test doesn't need the real model."""
    return chunks[:limit]


def test_pipeline_combines_both_searches_and_keeps_citations():
    corpus = [
        {"text": "cats sleep a lot during the day", "source": "a.txt", "metadata": {"chunk_index": 0}},
        {"text": "stock markets fell sharply today", "source": "c.txt", "metadata": {"chunk_index": 0}},
        {"text": "dogs love to run in the park", "source": "d.txt", "metadata": {"chunk_index": 0}},
        {"text": "the recipe needs two eggs and flour", "source": "e.txt", "metadata": {"chunk_index": 0}},
    ]
    vector_results = [
        RetrievedChunk(
            text="felines rest for many hours",
            source="b.txt",
            metadata={"chunk_index": 0},
            score=0.9,
        )
    ]
    results = hybrid_pipeline(
        "why do cats sleep", corpus, vector_results, limit=5, rerank_fn=no_model_rerank
    )
    sources = [r.source for r in results]
    assert "a.txt" in sources
    assert "b.txt" in sources
    assert all(r.source and r.metadata for r in results)