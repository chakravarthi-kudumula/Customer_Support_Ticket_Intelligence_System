import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.text_preprocessor import TextColumns, add_text_features
from src.utils.config import project_path


DEFAULT_INPUT = project_path("data/processed/cfpb_sample_90k.csv")
DEFAULT_OUTPUT = project_path("data/processed/cfpb_sample_90k_clean.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 cleaned CFPB sample dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else project_path(args.input)
    output_path = args.output if args.output.is_absolute() else project_path(args.output)

    df = pd.read_csv(input_path, low_memory=False)
    columns = TextColumns()

    before_rows = len(df)
    df = df[df[columns.source_text].notna()].copy()
    df = df[df[columns.target].notna()].copy()
    df = add_text_features(df, columns)
    df = df[df[columns.transformer_text].str.len() > 0].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows before filtering: {before_rows}")
    print(f"Rows after filtering: {len(df)}")
    print("Product counts:")
    print(df[columns.label].value_counts().to_string())
    print("Length summary:")
    print(df[["char_count", "word_count", "approx_token_count"]].describe().round(2).to_string())
    print(f"Duplicate complaint texts: {int(df['is_duplicate_text'].sum())}")


if __name__ == "__main__":
    main()
