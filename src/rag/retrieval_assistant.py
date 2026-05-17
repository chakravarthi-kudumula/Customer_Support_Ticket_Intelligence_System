from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.rag.context_builder import ContextConfig, build_context, summarize_context
from src.retrieval.faiss_search import DEFAULT_MANIFEST, search
from src.utils.config import project_path


DEFAULT_REPORT_DIR = project_path("artifacts/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a query using retrieved CFPB complaints.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-snippet-chars", type=int, default=700)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def format_list(values: list[str]) -> str:
    clean = [value for value in values if value]
    if not clean:
        return "not clear from the retrieved complaints"
    return ", ".join(clean)


def build_answer(query: str, context: list[dict], summary: dict) -> str:
    if not context:
        return (
            "I could not find closely related complaints in the current index. "
            "Try a more specific query or lower the similarity threshold."
        )

    lines = [
        f"Query: {query}",
        "",
        f"Retrieval confidence: {summary['confidence']} (top similarity: {summary['top_score']:.4f})",
        "",
        "What the retrieved complaints show:",
        f"- Products: {format_list(summary['products'])}",
        f"- CFPB issue labels: {format_list(summary['issues'])}",
        f"- Companies seen in the retrieved set: {format_list(summary['companies'])}",
        "",
        "Relevant complaint evidence:",
    ]

    for item in context:
        lines.extend(
            [
                (
                    f"- [{item['rank']}] Complaint {item['complaint_id']} "
                    f"({item['product']} | {item['issue']} | similarity {item['similarity']:.4f})"
                ),
                f"  {item['snippet']}",
            ]
        )

    lines.extend(
        [
            "",
            "Answer:",
            (
                "Based on the retrieved CFPB complaints, this query is most closely related to "
                f"{format_list(summary['products'])}. The CFPB issue labels in the retrieved set are "
                f"{format_list(summary['issues'])}. The answer is grounded only in the complaints "
                f"listed above; it should not be treated as legal or financial advice."
            ),
            "",
            f"Citations: {', '.join(summary['complaint_ids'])}",
        ]
    )
    return "\n".join(lines)


def answer_query(
    query: str,
    top_k: int = 5,
    manifest_path: Path = DEFAULT_MANIFEST,
    context_config: ContextConfig = ContextConfig(),
) -> dict:
    results = search(query=query, top_k=top_k, manifest_path=manifest_path)
    context = build_context(results, context_config)
    summary = summarize_context(context)
    answer = build_answer(query, context, summary)
    return {
        "query": query,
        "answer": answer,
        "confidence": summary["confidence"],
        "top_score": summary["top_score"],
        "citations": summary["complaint_ids"],
        "retrieved_count": len(context),
        "results": results,
        "context": context,
    }


def save_result_csv(result: dict, output_path: Path) -> None:
    rows = []
    for item in result["context"]:
        rows.append(
            {
                "query": result["query"],
                "confidence": result["confidence"],
                "top_score": result["top_score"],
                **item,
            }
        )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    result = answer_query(
        query=args.query,
        top_k=args.top_k,
        manifest_path=resolve_path(args.manifest),
        context_config=ContextConfig(
            max_snippet_chars=args.max_snippet_chars,
            min_similarity=args.min_similarity,
        ),
    )
    print(result["answer"])
    if args.output:
        save_result_csv(result, resolve_path(args.output))
        print(f"\nSaved retrieved context: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
