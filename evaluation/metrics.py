import time


def hit_at_k(sources: list[str], expected_source: str, k: int = 5) -> bool:
    """True if the right file is in the top k results."""
    return expected_source in sources[:k]


def reciprocal_rank(sources: list[str], expected_source: str) -> float:
    """1.0 if the right file is first, 0.5 if second, 0.33 if third... 0 if missing."""
    for position, source in enumerate(sources, start=1):
        if source == expected_source:
            return 1.0 / position
    return 0.0


def evaluate(retrieval_fn, dataset: list[dict], limit: int = 5) -> dict:
    """Run every question through retrieval_fn and score the results.
    retrieval_fn takes (query, limit=...) and returns chunks that have .source."""
    hits = 0
    rr_total = 0.0
    times_ms = []

    for item in dataset:
        start = time.perf_counter()
        results = retrieval_fn(item["question"], limit=limit)
        times_ms.append((time.perf_counter() - start) * 1000)

        sources = [r.source for r in results]
        hits += hit_at_k(sources, item["expected_source"], limit)
        rr_total += reciprocal_rank(sources, item["expected_source"])

    n = len(dataset)
    return {
        "questions": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": rr_total / n if n else 0.0,
        "avg_latency_ms": sum(times_ms) / n if n else 0.0,
    }