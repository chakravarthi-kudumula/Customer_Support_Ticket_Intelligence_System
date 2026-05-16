import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.utils.config import project_path


@dataclass(frozen=True)
class TransformerConfig:
    input_path: Path = project_path("data/processed/cfpb_sample_90k_clean.csv")
    text_column: str = "text_transformer"
    target_column: str = "target"
    model_name: str = "distilbert-base-uncased"
    output_dir: Path = project_path("artifacts/models/distilbert_complaint_classifier")
    reports_dir: Path = project_path("artifacts/reports")
    figures_dir: Path = project_path("artifacts/figures")
    logs_dir: Path = project_path("artifacts/transformer_logs")
    max_length: int = 256
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    epochs: float = 1.0
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_samples_per_class: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT/BERT for complaint classification.")
    parser.add_argument("--input", type=Path, default=TransformerConfig.input_path)
    parser.add_argument("--text-column", default=TransformerConfig.text_column)
    parser.add_argument("--target-column", default=TransformerConfig.target_column)
    parser.add_argument("--model-name", default=TransformerConfig.model_name)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=TransformerConfig.max_length)
    parser.add_argument("--epochs", type=float, default=TransformerConfig.epochs)
    parser.add_argument("--train-batch-size", type=int, default=TransformerConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TransformerConfig.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=TransformerConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TransformerConfig.seed)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional stratified cap per class for CPU-friendly pilot runs.",
    )
    parser.add_argument(
        "--cpu-pilot-samples-per-class",
        type=int,
        default=1000,
        help="Used automatically on CPU when --full-dataset is not supplied.",
    )
    parser.add_argument("--full-dataset", action="store_true", help="Train on all 90k rows even on CPU.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def default_model_output_dir(model_name: str) -> Path:
    if model_name == "distilbert-base-uncased":
        return project_path("artifacts/models/distilbert_complaint_classifier")
    return project_path(f"artifacts/models/{model_slug(model_name)}_complaint_classifier")


def report_path(config: TransformerConfig, filename: str) -> Path:
    stem, suffix = filename.rsplit(".", 1)
    return config.reports_dir / f"{stem}_{model_slug(config.model_name)}.{suffix}"


def figure_path(config: TransformerConfig, filename: str) -> Path:
    stem, suffix = filename.rsplit(".", 1)
    return config.figures_dir / f"{stem}_{model_slug(config.model_name)}.{suffix}"


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


def load_and_prepare_dataframe(config: TransformerConfig) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    df = pd.read_csv(config.input_path, low_memory=False)
    required = [config.text_column, config.target_column]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[required].dropna().copy()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df = df[df[config.text_column].str.len() > 0].copy()

    if config.max_samples_per_class is not None:
        if config.max_samples_per_class < 10:
            raise ValueError(
                "max_samples_per_class must be at least 10 for a stratified "
                "train/validation/test split."
            )
        df = (
            df.groupby(config.target_column, group_keys=False)
            .sample(n=config.max_samples_per_class, random_state=config.seed)
            .reset_index(drop=True)
        )

    labels = sorted(df[config.target_column].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df[config.target_column].map(label2id).astype("int64")
    return df, label2id, id2label


def stratified_split(df: pd.DataFrame, config: TransformerConfig) -> DatasetDict:
    train_df, temp_df = train_test_split(
        df,
        train_size=config.train_size,
        random_state=config.seed,
        stratify=df["label"],
    )
    relative_val_size = config.val_size / (config.val_size + config.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=config.seed,
        stratify=temp_df["label"],
    )

    keep_columns = [config.text_column, config.target_column, "label"]
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[keep_columns].reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df[keep_columns].reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df[keep_columns].reset_index(drop=True)),
        }
    )


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, config: TransformerConfig) -> DatasetDict:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch[config.text_column], truncation=True, max_length=config.max_length)

    tokenized = dataset.map(tokenize_batch, batched=True, desc="Tokenizing")
    tokenized = tokenized.remove_columns([config.text_column, config.target_column])
    return tokenized


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
    }


def build_training_arguments(config: TransformerConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": str(config.output_dir),
        "logging_dir": str(config.logs_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "num_train_epochs": config.epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "save_only_model": True,
        "report_to": "none",
        "seed": config.seed,
        "dataloader_num_workers": 0,
    }
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in accepted})


def save_evaluation_artifacts(
    config: TransformerConfig,
    trainer: Trainer,
    tokenized_test: Dataset,
    original_test: Dataset,
    id2label: dict[int, str],
    label2id: dict[str, int],
    training_seconds: float,
    run_scope: str,
) -> pd.DataFrame:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    test_output = trainer.predict(tokenized_test)
    y_true = test_output.label_ids
    y_pred = np.argmax(test_output.predictions, axis=-1)
    labels = [id2label[idx] for idx in sorted(id2label)]

    metrics = compute_metrics((test_output.predictions, y_true))
    results_df = pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "run_scope": run_scope,
                "max_length": config.max_length,
                "epochs": config.epochs,
                "train_batch_size": config.train_batch_size,
                "eval_batch_size": config.eval_batch_size,
                "learning_rate": config.learning_rate,
                "training_seconds": round(training_seconds, 2),
                **metrics,
            }
        ]
    )
    results_path = report_path(config, "transformer_model_results.csv")
    results_df.to_csv(results_path, index=False)
    results_df.to_csv(config.reports_dir / "transformer_model_results.csv", index=False)

    label_ids = sorted(id2label)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=labels,
        zero_division=0,
    )
    classification_report_path = report_path(config, "transformer_classification_report.md")
    report_body = (
        "# Phase 3 Transformer Classification Report\n\n"
        f"Model: `{config.model_name}`\n\n"
        f"Run scope: `{run_scope}`\n\n"
        "## Metrics Summary\n\n"
        + dataframe_to_markdown(results_df)
        + "\n\n## Test Classification Report\n\n```text\n"
        + report
        + "\n```\n",
    )
    classification_report_path.write_text(report_body, encoding="utf-8")
    (config.reports_dir / "transformer_classification_report.md").write_text(report_body, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(report_path(config, "transformer_confusion_matrix.csv"))
    cm_df.to_csv(config.reports_dir / "transformer_confusion_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Purples", cbar=False)
    plt.title(f"Transformer Confusion Matrix: {config.model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_path(config, "transformer_confusion_matrix.png"), dpi=170)
    plt.savefig(config.figures_dir / "transformer_confusion_matrix.png", dpi=170)
    plt.close()

    test_texts = original_test.to_pandas()
    errors = pd.DataFrame(
        {
            "text": test_texts.get(config.text_column, pd.Series(dtype=str)),
            "actual": [id2label[int(label)] for label in y_true],
            "predicted": [id2label[int(label)] for label in y_pred],
        }
    )
    errors = errors[errors["actual"] != errors["predicted"]]
    errors.to_csv(report_path(config, "transformer_error_examples.csv"), index=False)
    errors.to_csv(config.reports_dir / "transformer_error_examples.csv", index=False)

    (config.output_dir / "label_mapping.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, indent=2),
        encoding="utf-8",
    )

    baseline_path = config.reports_dir / "baseline_model_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path).sort_values("macro_f1", ascending=False).head(1)
        comparison = pd.DataFrame(
            [
                {
                    "stage": "phase2_baseline",
                    "model_name": str(baseline.iloc[0]["model_name"]),
                    "run_scope": "full_dataset",
                    "macro_f1": float(baseline.iloc[0]["macro_f1"]),
                    "accuracy": float(baseline.iloc[0]["accuracy"]),
                },
                {
                    "stage": "phase3_transformer",
                    "model_name": config.model_name,
                    "run_scope": run_scope,
                    "macro_f1": float(results_df.iloc[0]["macro_f1"]),
                    "accuracy": float(results_df.iloc[0]["accuracy"]),
                },
            ]
        )
        comparison.to_csv(report_path(config, "baseline_vs_transformer.csv"), index=False)
        comparison.to_csv(config.reports_dir / "baseline_vs_transformer.csv", index=False)

    return results_df


def train_transformer(config: TransformerConfig) -> pd.DataFrame:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    run_scope = "full_dataset" if config.max_samples_per_class is None else f"pilot_{config.max_samples_per_class}_per_class"
    print(f"Device: {device}", flush=True)
    print(f"Run scope: {run_scope}", flush=True)
    print(f"Model: {config.model_name}", flush=True)

    df, label2id, id2label = load_and_prepare_dataframe(config)
    dataset = stratified_split(df, config)
    print(dataset, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    trainer = Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    training_seconds = time.time() - start

    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    return save_evaluation_artifacts(
        config=config,
        trainer=trainer,
        tokenized_test=tokenized["test"],
        original_test=dataset["test"],
        id2label=id2label,
        label2id=label2id,
        training_seconds=training_seconds,
        run_scope=run_scope,
    )


def main() -> None:
    args = parse_args()
    device = detect_device()
    max_samples_per_class = args.max_samples_per_class
    if max_samples_per_class is None and not args.full_dataset and device == "cpu":
        max_samples_per_class = args.cpu_pilot_samples_per_class
        print(
            "CPU-only environment detected. Running a stratified pilot by default. "
            "Use --full-dataset for the full 90k-row run.",
            flush=True,
        )

    output_dir = resolve_path(args.output_dir) if args.output_dir else default_model_output_dir(args.model_name)

    config = TransformerConfig(
        input_path=resolve_path(args.input),
        text_column=args.text_column,
        target_column=args.target_column,
        model_name=args.model_name,
        output_dir=output_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_samples_per_class=max_samples_per_class,
    )
    results = train_transformer(config)
    print("Results:", flush=True)
    print(results.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
