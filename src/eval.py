from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from agent import route
from document_loader import PROJECT_ROOT
from retriever import build_retriever


DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval_set.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "eval_report.json"


def load_eval_set(path: Path = DEFAULT_EVAL_PATH) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def evaluate(
    top_k: int = 3,
    backend: str | None = None,
    rebuild: bool = True,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict:
    retriever = build_retriever(
        rebuild=rebuild,
        backend=backend,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    items = load_eval_set()
    route_hits = 0
    source_hits = 0
    source_total = 0
    reciprocal_rank_sum = 0.0
    retrieval_latencies_ms = []
    failures = []

    for item in items:
        question = item["question"]
        predicted_route = route(question)
        expected_route = item["expected_route"]
        route_ok = predicted_route == expected_route
        route_hits += int(route_ok)

        source_ok = None
        source_rank = None
        if item["expected_source"]:
            source_total += 1
            started = perf_counter()
            results = retriever.search(question, top_k=top_k)
            retrieval_latencies_ms.append((perf_counter() - started) * 1000)
            for rank, result in enumerate(results, start=1):
                if item["expected_source"] in result.chunk.source:
                    source_rank = rank
                    break
            source_ok = source_rank is not None
            source_hits += int(source_ok)
            if source_rank:
                reciprocal_rank_sum += 1.0 / source_rank

        if not route_ok or source_ok is False:
            failures.append(
                {
                    "question": question,
                    "expected_route": expected_route,
                    "predicted_route": predicted_route,
                    "expected_source": item["expected_source"],
                    "source_hit": source_ok,
                    "source_rank": source_rank,
                }
            )

    route_acc = route_hits / len(items) if items else 0.0
    source_hit = source_hits / source_total if source_total else 0.0
    mean_reciprocal_rank = reciprocal_rank_sum / source_total if source_total else 0.0
    mean_latency_ms = sum(retrieval_latencies_ms) / len(retrieval_latencies_ms) if retrieval_latencies_ms else 0.0
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "retriever": retriever.name,
        "eval_count": len(items),
        "top_k": top_k,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "route_accuracy": route_acc,
        "source_hit_at_k": source_hit,
        "mean_reciprocal_rank": mean_reciprocal_rank,
        "mean_retrieval_latency_ms": mean_latency_ms,
        "source_total": source_total,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", default=None, help="auto, bge-faiss, tfidf")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH, help="评测报告输出路径")
    args = parser.parse_args()

    metrics = evaluate(top_k=args.top_k, backend=args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RAG Agent eval")
    print("=" * 24)
    print(f"eval_count: {metrics['eval_count']}")
    print(f"top_k: {metrics['top_k']}")
    print(f"route accuracy: {metrics['route_accuracy']:.2%}")
    print(f"source hit@{args.top_k}: {metrics['source_hit_at_k']:.2%} ({metrics['source_total']} source questions)")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.3f}")
    print(f"mean retrieval latency: {metrics['mean_retrieval_latency_ms']:.2f} ms")
    print(f"failure_count: {len(metrics['failures'])}")
    print(f"report: {args.output}")
    for failure in metrics["failures"][:10]:
        print(json.dumps(failure, ensure_ascii=False))


if __name__ == "__main__":
    main()
