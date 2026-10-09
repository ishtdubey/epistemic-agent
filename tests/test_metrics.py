from evaluation.metrics import evaluate, hit_at_k, reciprocal_rank


class FakeChunk:
    def __init__(self, source):
        self.source = source


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], "b", k=2) is True
    assert hit_at_k(["a", "b", "c"], "c", k=2) is False


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], "a") == 1.0
    assert reciprocal_rank(["a", "b"], "b") == 0.5
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_evaluate_with_fake_retriever():
    dataset = [
        {"question": "q1", "expected_source": "a.txt"},
        {"question": "q2", "expected_source": "b.txt"},
    ]

    def fake_retriever(query, limit=5):
        return [FakeChunk("a.txt")]  # always answers a.txt

    scores = evaluate(fake_retriever, dataset)
    assert scores["questions"] == 2
    assert scores["hit_rate"] == 0.5
    assert scores["mrr"] == 0.5