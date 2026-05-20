from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

from src.tracking.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    configure_mlflow,
    log_artifact_if_exists,
    log_artifacts_if_exist,
    log_metrics,
    log_params,
)
from src.utils.config import project_path

REPORTS_DIR = project_path("artifacts/reports")
FIGURES_DIR = project_path("artifacts/figures")
MODELS_DIR = project_path("artifacts/models")
TRACKING_REPORT_PATH = REPORTS_DIR / "phase10_mlflow_tracking_report.md"
BASELINE_RESULTS_PATH = REPORTS_DIR / "baseline_model_results.csv"
TRANSFORMER_RESULTS_PATH = REPORTS_DIR / "transformer_model_results.csv"
RETRIEVAL_BENCHMARK_PATH = REPORTS_DIR / "phase5_vector_index_benchmark.csv"
FAISS_MANIFEST_PATH = project_path("artifacts/vector_indexes/latest_faiss_manifest.json")
SUMMARY_EXAMPLES_PATH = REPORTS_DIR / "phase6_summary_examples.csv"
RAG_EXAMPLES_PATH = REPORTS_DIR / "phase7_rag_examples.csv"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def log_baseline_runs() -> list[dict]:
    if not BASELINE_RESULTS_PATH.exists():
        return []

    results = pd.read_csv(BASELINE_RESULTS_PATH)
    logged = []
    best_model = str(results.sort_values("macro_f1", ascending=False).iloc[0]["model_name"])

    for _, row in results.iterrows():
        model_name = str(row["model_name"])
        with mlflow.start_run(run_name=f"baseline_{model_name}", nested=True):
            mlflow.set_tags({"phase": "2", "task": "complaint_classification", "model_family": "classical_ml"})
            log_params({
                "dataset": "cfpb_sample_90k_clean",
                "text_column": "text_ml_clean",
                "target_column": "target",
                "test_size": 0.2,
                "random_seed": 42,
                "model_name": model_name,
                "is_best_model": model_name == best_model,
            })
            log_metrics(row.drop(labels=["model_name"]).to_dict())
            log_artifact_if_exists(MODELS_DIR / f"{model_name}.joblib", "models")
            log_artifacts_if_exist([
                BASELINE_RESULTS_PATH,
                REPORTS_DIR / "baseline_classification_report.md",
                REPORTS_DIR / "baseline_confusion_matrix.csv",
                REPORTS_DIR / "baseline_error_examples.csv",
                FIGURES_DIR / "baseline_confusion_matrix.png",
            ], "reports")
            logged.append({"run": f"baseline_{model_name}", "status": "logged"})
    return logged


def log_transformer_run() -> list[dict]:
    if not TRANSFORMER_RESULTS_PATH.exists():
        return []

    results = pd.read_csv(TRANSFORMER_RESULTS_PATH)
    row = results.iloc[0].to_dict() if not results.empty else {}
    with mlflow.start_run(run_name="transformer_distilbert_classifier", nested=True):
        mlflow.set_tags({"phase": "3", "task": "complaint_classification", "model_family": "transformer"})
        log_params({
            "dataset": "cfpb_sample_90k_clean",
            "model_name": row.get("model_name", "distilbert"),
            "text_column": "text_transformer",
            "target_column": "target",
        })
        log_metrics({key: value for key, value in row.items() if key != "model_name"})
        log_artifacts_if_exist([
            TRANSFORMER_RESULTS_PATH,
            REPORTS_DIR / "transformer_classification_report.md",
            REPORTS_DIR / "transformer_confusion_matrix.csv",
            REPORTS_DIR / "transformer_error_examples.csv",
            FIGURES_DIR / "transformer_confusion_matrix.png",
            REPORTS_DIR / "baseline_vs_transformer.csv",
        ], "reports")
    return [{"run": "transformer_distilbert_classifier", "status": "logged"}]


def log_retrieval_benchmark_run() -> list[dict]:
    if not RETRIEVAL_BENCHMARK_PATH.exists():
        return []

    benchmark = pd.read_csv(RETRIEVAL_BENCHMARK_PATH)
    faiss_manifest = load_json(FAISS_MANIFEST_PATH)
    with mlflow.start_run(run_name="faiss_retrieval_benchmark", nested=True):
        mlflow.set_tags({"phase": "5", "task": "semantic_retrieval", "model_family": "sbert_faiss"})
        log_params({
            "embedding_model": faiss_manifest.get("model_name"),
            "index_type": faiss_manifest.get("index_type"),
            "rows": faiss_manifest.get("rows"),
            "embedding_dim": faiss_manifest.get("embedding_dim"),
            "metric": faiss_manifest.get("metric"),
            "top_k": benchmark["top_k"].iloc[0] if "top_k" in benchmark else None,
            "repeat": benchmark["repeat"].iloc[0] if "repeat" in benchmark else None,
        })
        metric_columns = [col for col in benchmark.columns if col.endswith("_mean") or col.endswith("_p95") or col == "top_k_overlap"]
        metrics = {f"{col}_avg": benchmark[col].mean() for col in metric_columns}
        if "exact_order_match" in benchmark.columns:
            metrics["exact_order_match_rate"] = benchmark["exact_order_match"].astype(float).mean()
        log_metrics(metrics)
        log_artifacts_if_exist([
            RETRIEVAL_BENCHMARK_PATH,
            REPORTS_DIR / "phase5_vector_index_benchmark.md",
            FAISS_MANIFEST_PATH,
        ], "reports")
    return [{"run": "faiss_retrieval_benchmark", "status": "logged"}]


def log_summarization_run() -> list[dict]:
    if not SUMMARY_EXAMPLES_PATH.exists():
        return []

    examples = pd.read_csv(SUMMARY_EXAMPLES_PATH)
    with mlflow.start_run(run_name="summarization_examples", nested=True):
        mlflow.set_tags({"phase": "6", "task": "complaint_summarization", "model_family": "seq2seq"})
        log_params({"dataset": "cfpb_sample_90k_clean", "example_count": len(examples)})
        metrics = {}
        for column in ["input_word_count", "summary_word_count", "compression_ratio"]:
            if column in examples.columns:
                metrics[f"{column}_avg"] = examples[column].mean()
        log_metrics(metrics)
        log_artifacts_if_exist([
            SUMMARY_EXAMPLES_PATH,
            REPORTS_DIR / "phase6_summarization_report.md",
        ], "reports")
    return [{"run": "summarization_examples", "status": "logged"}]


def log_rag_run() -> list[dict]:
    if not RAG_EXAMPLES_PATH.exists():
        return []

    examples = pd.read_csv(RAG_EXAMPLES_PATH)
    with mlflow.start_run(run_name="retrieval_context_examples", nested=True):
        mlflow.set_tags({"phase": "7", "task": "retrieval_context", "model_family": "extractive_rag"})
        log_params({"dataset": "cfpb_sample_90k_clean", "example_count": len(examples)})
        if "top_score" in examples.columns:
            log_metrics({"top_score_avg": examples["top_score"].mean()})
        log_artifacts_if_exist([
            RAG_EXAMPLES_PATH,
            REPORTS_DIR / "phase7_rag_report.md",
            REPORTS_DIR / "phase7_rag_answers.md",
        ], "reports")
    return [{"run": "retrieval_context_examples", "status": "logged"}]


def write_report(tracking_uri: str, experiment_name: str, logged_runs: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 10 MLflow Experiment Tracking",
        "",
        "MLflow is used to track the main completed experiments for this project.",
        "",
        "## Tracking Setup",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Tracking URI: `{tracking_uri}`",
        "- Backend store: `mlflow.db`",
        "- Artifact directory: `mlruns/`",
        "",
        "## Logged Runs",
        "",
    ]
    for item in logged_runs:
        lines.append(f"- `{item['run']}`: {item['status']}")
    lines.extend([
        "",
        "## Open MLflow UI",
        "",
        "```bash",
        "source ticket/bin/activate",
        "mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000",
        "```",
        "",
        "Then open `http://127.0.0.1:5000`.",
    ])
    TRACKING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tracking_uri = configure_mlflow(DEFAULT_EXPERIMENT_NAME)
    logged_runs = []
    with mlflow.start_run(run_name="phase10_experiment_tracking"):
        mlflow.set_tags({"phase": "10", "task": "experiment_tracking"})
        log_params({
            "project": "Customer Support Ticket Intelligence System",
            "dataset": "cfpb_sample_90k_clean",
            "tracking_scope": "baseline_transformer_retrieval_summarization_rag",
        })
        logged_runs.extend(log_baseline_runs())
        logged_runs.extend(log_transformer_run())
        logged_runs.extend(log_retrieval_benchmark_run())
        logged_runs.extend(log_summarization_run())
        logged_runs.extend(log_rag_run())
        write_report(tracking_uri, DEFAULT_EXPERIMENT_NAME, logged_runs)
        log_artifact_if_exists(TRACKING_REPORT_PATH, "reports")

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Logged runs: {len(logged_runs)}")
    print(f"Report: {TRACKING_REPORT_PATH}")


if __name__ == "__main__":
    main()
