import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing.text_preprocessor import TextColumns, add_text_features
from src.utils.config import project_path


DEFAULT_INPUT = project_path("data/processed/cfpb_sample_90k.csv")
DEFAULT_REPORT = project_path("artifacts/reports/phase1_eda_summary.md")
DEFAULT_FIGURES = project_path("artifacts/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 EDA summary and figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def top_keywords_by_product(df: pd.DataFrame, text_col: str, target_col: str, top_n: int = 15) -> dict[str, list[tuple[str, int]]]:
    keywords = {}
    for product, group in df.groupby(target_col):
        vectorizer = CountVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=5)
        matrix = vectorizer.fit_transform(group[text_col].fillna(""))
        counts = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, counts), key=lambda item: item[1], reverse=True)[:top_n]
        keywords[product] = [(term, int(count)) for term, count in ranked]
    return keywords


def markdown_series(series: pd.Series) -> list[str]:
    return ["| value | count |", "| --- | ---: |"] + [
        f"| {str(index).replace('|', '/')} | {int(value):,} |"
        for index, value in series.items()
    ]


def markdown_frame(frame: pd.DataFrame) -> list[str]:
    output = frame.reset_index()
    columns = [str(column) for column in output.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in output.iterrows():
        values = [str(value).replace("|", "/") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown_report(path: Path, df: pd.DataFrame, keywords: dict[str, list[tuple[str, int]]]) -> None:
    columns = TextColumns()
    missing = df[[columns.source_text, columns.target, "Issue"]].isna().sum()
    product_counts = df[columns.target].value_counts()
    issue_counts = df["Issue"].value_counts().head(20)
    length_summary = df[["char_count", "word_count", "approx_token_count"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(2)
    duplicate_count = int(df["is_duplicate_text"].sum())

    lines = [
        "# Phase 1 EDA Summary",
        "",
        f"Rows analyzed: {len(df):,}",
        f"Unique products: {df[columns.target].nunique():,}",
        f"Unique issues: {df['Issue'].nunique():,}",
        f"Duplicate complaint text rows: {duplicate_count:,}",
        "",
        "## Missing Values",
        "",
        *markdown_series(missing),
        "",
        "## Product Distribution",
        "",
        *markdown_series(product_counts),
        "",
        "## Top 20 Issues",
        "",
        *markdown_series(issue_counts),
        "",
        "## Length Distribution Summary",
        "",
        *markdown_frame(length_summary),
        "",
        "## Top Keywords by Product",
        "",
    ]

    for product, terms in keywords.items():
        lines.extend([f"### {product}", ""])
        lines.extend([f"- {term}: {count}" for term, count in terms])
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    columns = TextColumns()
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, y=columns.target, order=df[columns.target].value_counts().index)
    ax.set_title("Product Distribution")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Product")
    plt.tight_layout()
    plt.savefig(figures_dir / "product_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["word_count"], bins=60)
    plt.title("Complaint Word Count Distribution")
    plt.xlabel("Word count")
    plt.ylabel("Complaints")
    plt.tight_layout()
    plt.savefig(figures_dir / "word_count_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x="word_count", y=columns.target, showfliers=False)
    plt.title("Word Count by Product")
    plt.xlabel("Word count")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(figures_dir / "word_count_by_product.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    top_issues = df["Issue"].value_counts().head(15)
    sns.barplot(x=top_issues.values, y=top_issues.index)
    plt.title("Top 15 Issues")
    plt.xlabel("Rows")
    plt.ylabel("Issue")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_issues.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else project_path(args.input)
    report_path = args.report if args.report.is_absolute() else project_path(args.report)
    figures_dir = args.figures_dir if args.figures_dir.is_absolute() else project_path(args.figures_dir)

    df = pd.read_csv(input_path, low_memory=False)
    df = add_text_features(df)
    keywords = top_keywords_by_product(df, "text_ml_clean", "Product")
    write_markdown_report(report_path, df, keywords)
    save_figures(df, figures_dir)

    print(f"Report: {report_path}")
    print(f"Figures: {figures_dir}")
    print(f"Rows analyzed: {len(df)}")


if __name__ == "__main__":
    main()
