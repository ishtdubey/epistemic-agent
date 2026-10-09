from src.rag.ranker import rerank
from src.rag.retriever import RetrievedChunk


def make_chunk(text, source):
    return RetrievedChunk(text=text, source=source, metadata={}, score=0.0)


def fake_scorer(query, texts):
    """Pretend model: a text scores 1 point for each query word it contains."""
    words = query.lower().split()
    return [float(sum(w in t.lower() for w in words)) for t in texts]


def test_rerank_puts_best_match_first():
    chunks = [
        make_chunk("stock markets fell", "c.txt"),
        make_chunk("cats sleep a lot", "a.txt"),
    ]
    results = rerank("cats sleep", chunks, scorer=fake_scorer)
    assert results[0].source == "a.txt"


def test_rerank_respects_limit_and_empty_input():
    chunks = [make_chunk("one", "1.txt"), make_chunk("two", "2.txt")]
    assert len(rerank("one", chunks, limit=1, scorer=fake_scorer)) == 1
    assert rerank("one", [], scorer=fake_scorer) == []



