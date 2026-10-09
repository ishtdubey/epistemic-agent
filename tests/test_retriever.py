from src.rag.retriever import RetrievedChunk, blend_results, keyword_search


def make_chunk(text, source):
    return RetrievedChunk(text=text, source=source, metadata={}, score=0.0)


def test_keyword_search_finds_matching_chunk():
    chunks = [
        {"text": "cats sleep a lot during the day", "source": "a.txt", "metadata": {}},
        {"text": "dogs love to run in the park", "source": "b.txt", "metadata": {}},
        {"text": "stock markets fell sharply today", "source": "c.txt", "metadata": {}},
    ]
    results = keyword_search("why do cats sleep", chunks)
    assert len(results) == 1
    assert results[0].source == "a.txt"


def test_blend_puts_chunk_found_by_both_searches_first():
    keyword = [make_chunk("A", "a.txt"), make_chunk("B", "b.txt")]
    vector = [make_chunk("C", "c.txt"), make_chunk("B", "b.txt")]
    results = blend_results(keyword, vector)
    assert results[0].source == "b.txt"


def test_blend_has_no_duplicates_and_respects_limit():
    keyword = [make_chunk("A", "a.txt"), make_chunk("B", "b.txt")]
    vector = [make_chunk("B", "b.txt"), make_chunk("C", "c.txt")]
    results = blend_results(keyword, vector, limit=2)
    assert len(results) == 2
    assert len({(r.source, r.text) for r in results}) == 2