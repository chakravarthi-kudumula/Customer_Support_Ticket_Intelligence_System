"""Phase 3 pipeline: fine-tune DistilBERT/BERT for complaint classification.

CPU-only environments run a stratified pilot by default. Use --full-dataset for
full 90k-row training when you have enough compute time.
"""

from src.models.transformer_classifier import main as fine_tune_transformer


def main() -> None:
    fine_tune_transformer()


if __name__ == "__main__":
    main()
