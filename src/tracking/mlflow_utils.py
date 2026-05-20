from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.utils.config import project_path


DEFAULT_EXPERIMENT_NAME = "customer-support-ticket-intelligence"
DEFAULT_TRACKING_DIR = project_path("mlruns")
DEFAULT_TRACKING_DB = project_path("mlflow.db")


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    tracking_db: Path = DEFAULT_TRACKING_DB,
) -> str:
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_db}"
    artifact_location = tracking_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_params(params: dict[str, Any]) -> None:
    clean_params = {key: value for key, value in params.items() if value is not None}
    if clean_params:
        mlflow.log_params(clean_params)


def log_metrics(metrics: dict[str, Any]) -> None:
    clean_metrics = {}
    for key, value in metrics.items():
        if pd.isna(value):
            continue
        try:
            clean_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    if clean_metrics:
        mlflow.log_metrics(clean_metrics)


def log_artifact_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifacts_if_exist(paths: list[Path], artifact_path: str | None = None) -> None:
    for path in paths:
        log_artifact_if_exists(path, artifact_path)
