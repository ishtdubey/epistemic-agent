import src.rag.retriever as retriever


def fake_rerank(query, chunks, limit=5):
    """Pretend reranker so the tests don't need the real model."""
    return chunks[:limit]


def make_files(folder):
    (folder / "cats.txt").write_text("Cats sleep a lot during the day to save energy.")
    (folder / "dogs.txt").write_text("Dogs love to run in the park every morning.")
    (folder / "stocks.txt").write_text("Stock markets fell sharply today on rate worries.")
    (folder / "cooking.txt").write_text("The recipe needs two eggs and a pinch of salt.")


def test_run_hybrid_retrieval_returns_cited_chunks(tmp_path, monkeypatch):
    make_files(tmp_path)
    monkeypatch.setattr(retriever, "DATA_DIR", tmp_path)
    monkeypatch.setattr(retriever, "_safe_rerank", fake_rerank)

    results = retriever.run_hybrid_retrieval("why do cats sleep")
    assert results[0].source == "cats.txt"
    assert all(r.source and r.metadata for r in results)


def test_filters_remove_non_matching_chunks(tmp_path, monkeypatch):
    make_files(tmp_path)
    monkeypatch.setattr(retriever, "DATA_DIR", tmp_path)
    monkeypatch.setattr(retriever, "_safe_rerank", fake_rerank)

    results = retriever.run_hybrid_retrieval("why do cats sleep", filters={"chunk_index": 99})
    assert results == []


def test_direct_retrieval_finds_the_right_chunk(tmp_path, monkeypatch):
    make_files(tmp_path)
    monkeypatch.setattr(retriever, "DATA_DIR", tmp_path)

    results = retriever.run_direct_retrieval("why do cats sleep")
    assert results[0].source == "cats.txt"