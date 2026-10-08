from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    metadata: dict
    score: float


def run_hybrid_retrieval(
    query: str,
    filters: dict | None = None,
    limit: int = 5,
) -> list[RetrievedChunk]:
    # Temporary fake answer so teammates can already call this function.
    # We replace it with the real search later.
    return [
        RetrievedChunk(
            text="This is a placeholder chunk.",
            source="placeholder.txt",
            metadata={"note": "fake data"},
            score=0.0,
        )
    ]