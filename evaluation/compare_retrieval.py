import json
import sys
from pathlib import Path

import src.rag.retriever as retriever
from evaluation.metrics import evaluate
from src.rag.retriever import hybrid_pipeline, keyword_search, load_corpus, vector_search


def keyword_only(query, limit=5):
    return keyword_search(query, load_corpus(), limit=limit)


def blended_no_rerank(query, limit=5):
    return retriever.run_direct_retrieval(query, limit=limit)


def reranked(query, limit=5):
    # No safety net here on purpose: if the real model can't load,
    # we want to see an error, not silently copy the blended numbers.
    corpus = load_corpus()
    return hybrid_pipeline(query, corpus, vector_search(query, None, limit * 3), limit=limit)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/eval_corpus"
    questions_file = sys.argv[2] if len(sys.argv) > 2 else "evaluation/questions_big.json"

    retriever.DATA_DIR = Path(data_dir)
    dataset = json.loads(Path(questions_file).read_text(encoding="utf-8"))

    variants = {
        "keyword_only": keyword_only,
        "blended_no_rerank": blended_no_rerank,
        "reranked": reranked,
    }

    results = {}
    for name, fn in variants.items():
        try:
            fn("warm up", limit=1)  # first call loads the model, so don't time it
            results[name] = evaluate(fn, dataset)
        except (OSError, ImportError) as error:
            results[name] = {"error": f"could not run: {error}"}

    for name, scores in results.items():
        print(name, scores)

    out_file = Path("evaluation/results/retrieval_comparison.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()