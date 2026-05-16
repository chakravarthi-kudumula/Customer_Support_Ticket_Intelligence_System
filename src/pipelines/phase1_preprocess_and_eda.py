"""Phase 1 pipeline: preprocess text and generate NLP EDA artifacts.

This writes:
    data/processed/cfpb_sample_90k_clean.csv
    artifacts/reports/phase1_eda_summary.md
    artifacts/figures/*.png
"""

from src.data.phase1_eda_report import main as generate_eda_report
from src.preprocessing.build_preprocessed_dataset import main as build_preprocessed_dataset


def main() -> None:
    build_preprocessed_dataset()
    generate_eda_report()


if __name__ == "__main__":
    main()
