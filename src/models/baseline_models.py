import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "artifacts" / "matplotlib"),
)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.config import project_path


@dataclass(frozen=True)
class BaselineConfig:
    input_path: Path = project_path("data/processed/cfpb_sample_90k_clean.csv")
    text_column: str = "text_ml_clean"
    target_column: str = "target"
    test_size: float = 0.2
    random_seed: int = 42
    models_dir: Path = project_path("artifacts/models")
    reports_dir: Path = project_path("artifacts/reports")
    figures_dir: Path = project_path("artifacts/figures")


MODEL_SPECS = {
    "dummy_most_frequent": Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(max_features=1000)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    ),
    "bow_logistic_regression": Pipeline(
        steps=[
            ("vectorizer", CountVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=-1,
                    solver="saga",
                    C=1.0,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    ),
    "tfidf_logistic_regression": Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=80000,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=-1,
                    solver="saga",
                    C=1.0,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    ),
    "tfidf_linear_svm": Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=80000,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(C=1.0, random_state=42)),
        ]
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 baseline NLP classifiers.")
    parser.add_argument("--input", type=Path, default=BaselineConfig.input_path)
    parser.add_argument("--text-column", default=BaselineConfig.text_column)
    parser.add_argument("--target-column", default=BaselineConfig.target_column)
    parser.add_argument("--test-size", type=float, default=BaselineConfig.test_size)
    parser.add_argument("--seed", type=int, default=BaselineConfig.random_seed)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def load_dataset(config: BaselineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.input_path, low_memory=False)
    required_columns = [config.text_column, config.target_column]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[required_columns].dropna().copy()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df = df[df[config.text_column].str.len() > 0].copy()
    return df


def split_dataset(df: pd.DataFrame, config: BaselineConfig):
    return train_test_split(
        df[config.text_column],
        df[config.target_column],
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=df[config.target_column],
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series | list[str]) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    columns = [str(column) for column in display_df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display_df.iterrows():
        values = [str(value).replace("|", "/") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def train_and_evaluate(config: BaselineConfig) -> tuple[pd.DataFrame, str]:
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config)
    x_train, x_test, y_train, y_test = split_dataset(df, config)
    labels = sorted(y_train.unique())

    results = []
    predictions = {}

    for model_name, pipeline in MODEL_SPECS.items():
        print(f"Training {model_name}...", flush=True)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, y_pred)
        results.append({"model_name": model_name, **metrics})
        predictions[model_name] = y_pred
        joblib.dump(pipeline, config.models_dir / f"{model_name}.joblib")

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    best_model_name = str(results_df.iloc[0]["model_name"])
    best_predictions = predictions[best_model_name]

    results_path = config.reports_dir / "baseline_model_results.csv"
    results_df.to_csv(results_path, index=False)

    report = classification_report(y_test, best_predictions, labels=labels, zero_division=0)
    report_path = config.reports_dir / "baseline_classification_report.md"
    report_path.write_text(
        "# Phase 2 Baseline Classification Report\n\n"
        f"Best model: `{best_model_name}`\n\n"
        "## Metrics Summary\n\n"
        + dataframe_to_markdown(results_df)
        + "\n\n## Best Model Classification Report\n\n```text\n"
        + report
        + "\n```\n",
        encoding="utf-8",
    )

    errors = pd.DataFrame(
        {
            "text": x_test.reset_index(drop=True),
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(best_predictions),
        }
    )
    errors = errors[errors["actual"] != errors["predicted"]]
    errors.to_csv(config.reports_dir / "baseline_error_examples.csv", index=False)

    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(config.reports_dir / "baseline_confusion_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Baseline Confusion Matrix: {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.figures_dir / "baseline_confusion_matrix.png", dpi=170)
    plt.close()

    print("Results:")
    print(results_df.round(4).to_string(index=False))
    print(f"Best model: {best_model_name}")
    print(f"Saved metrics: {results_path}")
    return results_df, best_model_name


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        input_path=resolve_path(args.input),
        text_column=args.text_column,
        target_column=args.target_column,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
