"""Phase 0 pipeline: create the balanced CFPB sample dataset.

This orchestrates the raw-to-balanced-sample step and writes:
    data/processed/cfpb_sample_90k.csv
"""

from src.data.create_balanced_sample import main as create_balanced_sample


def main() -> None:
    create_balanced_sample()


if __name__ == "__main__":
    main()
