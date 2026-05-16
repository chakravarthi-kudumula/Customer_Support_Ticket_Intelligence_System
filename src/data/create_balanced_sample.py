import argparse
import csv
import random
from collections import Counter
from pathlib import Path


TARGET_PRODUCTS = [
    "Debt collection",
    "Checking or savings account",
    "Mortgage",
    "Credit card",
    "Money transfer, virtual currency, or money service",
    "Student loan",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "complaints.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cfpb_sample_90k.csv"
TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Product"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced random sample from the CFPB complaints CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to raw complaints CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the sampled output CSV.",
    )
    parser.add_argument(
        "--samples-per-product",
        type=int,
        default=15_000,
        help="Number of random rows to keep per target product.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def reservoir_sample_by_product(
    input_path: Path,
    target_products: list[str],
    text_column: str,
    target_column: str,
    samples_per_product: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], Counter]:
    rng = random.Random(seed)
    reservoirs = {product: [] for product in target_products}
    seen_counts: Counter = Counter()
    target_set = set(target_products)

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = {text_column, target_column} - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        for row in reader:
            product = row.get(target_column)
            narrative = row.get(text_column)

            if product not in target_set or not narrative or not narrative.strip():
                continue

            seen_counts[product] += 1
            seen = seen_counts[product]
            reservoir = reservoirs[product]

            if len(reservoir) < samples_per_product:
                reservoir.append(row)
                continue

            replacement_index = rng.randrange(seen)
            if replacement_index < samples_per_product:
                reservoir[replacement_index] = row

    return reservoirs, seen_counts


def validate_sample(reservoirs: dict[str, list[dict[str, str]]], samples_per_product: int) -> None:
    short_products = {
        product: len(rows)
        for product, rows in reservoirs.items()
        if len(rows) < samples_per_product
    }

    if short_products:
        raise ValueError(
            "Not enough eligible rows for every product. "
            f"Products below target: {short_products}"
        )


def write_sample(
    reservoirs: dict[str, list[dict[str, str]]],
    output_path: Path,
    seed: int,
) -> Counter:
    rows = [row for product_rows in reservoirs.values() for row in product_rows]
    rng = random.Random(seed)
    rng.shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return Counter(row["Product"] for row in rows)


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    reservoirs, seen_counts = reservoir_sample_by_product(
        input_path=input_path,
        target_products=TARGET_PRODUCTS,
        text_column=TEXT_COLUMN,
        target_column=TARGET_COLUMN,
        samples_per_product=args.samples_per_product,
        seed=args.seed,
    )
    validate_sample(reservoirs, args.samples_per_product)
    final_counts = write_sample(reservoirs, output_path, args.seed)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Eligible rows scanned by product: {dict(seen_counts)}")
    print(f"Final sampled rows: {sum(final_counts.values())}")
    print("Final product counts:")
    for product in TARGET_PRODUCTS:
        print(f"  {product}: {final_counts[product]}")


if __name__ == "__main__":
    main()

