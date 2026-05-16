"""Phase 6 pipeline: summarize long complaints."""

import argparse
from pathlib import Path

import pandas as pd

from src.summarization.complaint_summarizer import ComplaintSummarizer, SummaryConfig
from src.utils.config import project_path


DEFAULT_INPUT_PATH = project_path("data/processed/cfpb_sample_90k_clean.csv")
DEFAULT_REPORT_DIR = project_path("artifacts/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample complaint summaries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-name", default=SummaryConfig.model_name)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--min-words", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=SummaryConfig.max_input_tokens)
    parser.add_argument("--max-summary-tokens", type=int, default=SummaryConfig.max_summary_tokens)
    parser.add_argument("--min-summary-tokens", type=int, default=SummaryConfig.min_summary_tokens)
    parser.add_argument("--local-files-only", action="store_true", help="Load the model from the local cache without network checks.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def load_long_complaints(input_path: Path, samples: int, min_words: int, seed: int) -> pd.DataFrame:
    columns = [
        "Complaint ID",
        "Product",
        "Issue",
        "Company",
        "State",
        "Date received",
        "text_transformer",
        "word_count",
    ]
    df = pd.read_csv(resolve_path(input_path), usecols=lambda col: col in columns, low_memory=False)
    df = df[df["text_transformer"].notna()].copy()
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce")
    candidates = df[df["word_count"] >= min_words].copy()
    if candidates.empty:
        raise ValueError(f"No complaints found with word_count >= {min_words}")

    per_product = max(1, samples // candidates["Product"].nunique())
    sampled = (
        candidates.groupby("Product", group_keys=False)
        .sample(n=per_product, random_state=seed, replace=False)
        .reset_index(drop=True)
    )
    if len(sampled) < samples:
        remainder = candidates.drop(sampled.index, errors="ignore").sample(
            n=min(samples - len(sampled), len(candidates) - len(sampled)),
            random_state=seed,
        )
        sampled = pd.concat([sampled, remainder], ignore_index=True)
    return sampled.head(samples).reset_index(drop=True)


def write_report(path: Path, results: pd.DataFrame, model_name: str, min_words: int) -> None:
    lines = [
        "# Phase 6 Summarization Report",
        "",
        "Uses a pretrained seq2seq model to shorten long complaint narratives.",
        "",
        "## Setup",
        "",
        f"- Model: `{model_name}`",
        f"- Input filter: complaints with at least `{min_words}` words",
        f"- Samples summarized: `{len(results)}`",
        f"- Mean input words: `{results['input_word_count'].mean():.1f}`",
        f"- Mean summary words: `{results['summary_word_count'].mean():.1f}`",
        f"- Mean compression ratio: `{results['compression_ratio'].mean():.3f}`",
        f"- Truncated inputs: `{int(results['was_truncated'].sum())}`",
        "",
        "## Notes",
        "",
        "- CFPB does not include target summaries, so this step uses the pretrained model as-is.",
        "- Review summaries before using them in any customer-facing workflow.",
        "- Very long complaints may be truncated to fit the model context window.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = SummaryConfig(
        model_name=args.model_name,
        max_input_tokens=args.max_input_tokens,
        max_summary_tokens=args.max_summary_tokens,
        min_summary_tokens=args.min_summary_tokens,
        local_files_only=args.local_files_only,
    )
    complaints = load_long_complaints(resolve_path(args.input), args.samples, args.min_words, args.seed)
    summarizer = ComplaintSummarizer(config)

    rows = []
    for _, row in complaints.iterrows():
        result = summarizer.summarize(row["text_transformer"])
        rows.append(
            {
                "Complaint ID": row.get("Complaint ID"),
                "Product": row.get("Product"),
                "Issue": row.get("Issue"),
                "Company": row.get("Company"),
                "State": row.get("State"),
                "Date received": row.get("Date received"),
                "summary": result["summary"],
                "input_word_count": result["input_word_count"],
                "summary_word_count": result["summary_word_count"],
                "compression_ratio": result["compression_ratio"],
                "was_truncated": result["was_truncated"],
                "source_text_preview": str(row["text_transformer"])[:1000],
            }
        )

    results = pd.DataFrame(rows)
    csv_path = report_dir / "phase6_summary_examples.csv"
    md_path = report_dir / "phase6_summarization_report.md"
    results.to_csv(csv_path, index=False)
    write_report(md_path, results, args.model_name, args.min_words)

    print(f"Summary examples: {csv_path}", flush=True)
    print(f"Summary report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
