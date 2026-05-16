"""Phase 2 pipeline: train and evaluate baseline classical NLP models.

This writes:
    artifacts/models/*.joblib
    artifacts/reports/baseline_model_results.csv
    artifacts/reports/baseline_classification_report.md
    artifacts/reports/baseline_confusion_matrix.csv
    artifacts/reports/baseline_error_examples.csv
    artifacts/figures/baseline_confusion_matrix.png
"""

from src.models.baseline_models import main as train_baseline_models


def main() -> None:
    train_baseline_models()


if __name__ == "__main__":
    main()
