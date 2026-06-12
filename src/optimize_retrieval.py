from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_loader import PROJECT_ROOT
from eval import evaluate


DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "optimization_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="tfidf", help="Use tfidf for fast local optimization.")
    parser.add_argument("--chunk-sizes", default="256,512,700")
    parser.add_argument("--overlaps", default="64,120")
    parser.add_argument("--top-ks", default="3,5")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    results = []
    for chunk_size in [int(x) for x in args.chunk_sizes.split(",")]:
        for overlap in [int(x) for x in args.overlaps.split(",")]:
            if overlap >= chunk_size:
                continue
            for top_k in [int(x) for x in args.top_ks.split(",")]:
                metrics = evaluate(
                    top_k=top_k,
                    backend=args.backend,
                    rebuild=True,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                results.append(metrics)
                print(json.dumps({k: v for k, v in metrics.items() if k != "failures"}, ensure_ascii=False))

    best = max(
        results,
        key=lambda item: (
            item["source_hit_at_k"],
            item["mean_reciprocal_rank"],
            -item["mean_retrieval_latency_ms"],
        ),
    )
    print("\nBest config summary")
    print("=" * 24)
    print(json.dumps({k: v for k, v in best.items() if k != "failures"}, ensure_ascii=False, indent=2))
    report = {
        "backend": args.backend,
        "best": {k: v for k, v in best.items() if k != "failures"},
        "results": [{k: v for k, v in item.items() if k != "failures"} for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
