"""Phase 7 pipeline: run retrieval-grounded answer examples."""

from pathlib import Path

import pandas as pd

from src.rag.retrieval_assistant import answer_query
from src.utils.config import project_path


DEFAULT_REPORT_DIR = project_path("artifacts/reports")
EXAMPLE_QUERIES = [
    "My bank charged me twice for the same transaction",
    "A debt collector keeps calling about a debt I do not owe",
    "My mortgage servicer made an escrow mistake",
    "My credit card fraud dispute was denied",
    "My student loan servicer gave me wrong payment information",
]


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Phase 7 RAG-Style Retrieval Report",
        "",
        "FAISS retrieval is used to build answers from similar CFPB complaints, with complaint IDs included as citations.",
        "",
        "No paid LLM or API is used here. The answer is based on retrieved complaint context.",
        "",
        "## Example Queries",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['query']}",
                "",
                f"- Confidence: `{row['confidence']}`",
                f"- Top similarity: `{row['top_score']}`",
                f"- Citations: `{row['citations']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    answers = []

    for query in EXAMPLE_QUERIES:
        result = answer_query(query=query, top_k=5)
        rows.append(
            {
                "query": query,
                "confidence": result["confidence"],
                "top_score": result["top_score"],
                "retrieved_count": result["retrieved_count"],
                "citations": ", ".join(result["citations"]),
            }
        )
        answers.append({"query": query, "answer": result["answer"]})

    examples_path = DEFAULT_REPORT_DIR / "phase7_rag_examples.csv"
    answers_path = DEFAULT_REPORT_DIR / "phase7_rag_answers.md"
    report_path = DEFAULT_REPORT_DIR / "phase7_rag_report.md"

    pd.DataFrame(rows).to_csv(examples_path, index=False)
    answers_path.write_text(
        "\n\n".join(f"## {item['query']}\n\n{item['answer']}" for item in answers),
        encoding="utf-8",
    )
    write_report(report_path, rows)

    print(f"RAG examples: {examples_path}", flush=True)
    print(f"RAG answers: {answers_path}", flush=True)
    print(f"RAG report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
