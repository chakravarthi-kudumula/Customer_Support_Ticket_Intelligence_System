import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config import project_path


@dataclass(frozen=True)
class EmbeddingConfig:
    input_path: Path = project_path("data/processed/cfpb_sample_90k_clean.csv")
    output_dir: Path = project_path("artifacts/embeddings")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_column: str = "text_transformer"
    max_rows: int | None = 10_000
    batch_size: int = 64
    seed: int = 42
    normalize_embeddings: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sentence embeddings for semantic search.")
    parser.add_argument("--input", type=Path, default=EmbeddingConfig.input_path)
    parser.add_argument("--output-dir", type=Path, default=EmbeddingConfig.output_dir)
    parser.add_argument("--model-name", default=EmbeddingConfig.model_name)
    parser.add_argument("--text-column", default=EmbeddingConfig.text_column)
    parser.add_argument("--max-rows", type=int, default=EmbeddingConfig.max_rows)
    parser.add_argument("--full-dataset", action="store_true", help="Embed all rows instead of the default sample.")
    parser.add_argument("--batch-size", type=int, default=EmbeddingConfig.batch_size)
    parser.add_argument("--seed", type=int, default=EmbeddingConfig.seed)
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
    return model_name.replace("sentence-transformers/", "").replace("/", "_").replace("-", "_").lower()


def load_metadata(config: EmbeddingConfig) -> pd.DataFrame:
    columns = [
        "Complaint ID",
        "Product",
        "Issue",
        "Company",
        "State",
        "Date received",
        "text_transformer",
        "text_raw",
        "target",
    ]
    df = pd.read_csv(config.input_path, usecols=lambda col: col in columns, low_memory=False)
    if config.text_column not in df.columns:
        raise ValueError(f"Missing text column: {config.text_column}")

    df = df[df[config.text_column].notna()].copy()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df = df[df[config.text_column].str.len() > 0].copy()

    if config.max_rows is not None and len(df) > config.max_rows:
        stratify_col = "Product" if "Product" in df.columns else None
        if stratify_col:
            per_class = max(1, config.max_rows // df[stratify_col].nunique())
            sampled = (
                df.groupby(stratify_col, group_keys=False)
                .sample(n=per_class, random_state=config.seed)
                .reset_index(drop=True)
            )
            if len(sampled) < config.max_rows:
                remainder = df.drop(sampled.index, errors="ignore").sample(
                    n=min(config.max_rows - len(sampled), len(df) - len(sampled)),
                    random_state=config.seed,
                )
                sampled = pd.concat([sampled, remainder], ignore_index=True)
            df = sampled.head(config.max_rows)
        else:
            df = df.sample(n=config.max_rows, random_state=config.seed)

    df = df.reset_index(drop=True)
    df.insert(0, "embedding_row_id", range(len(df)))
    return df


def build_embeddings(config: EmbeddingConfig) -> tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(config)
    device = detect_device()
    print(f"Embedding rows: {len(metadata):,}", flush=True)
    print(f"Model: {config.model_name}", flush=True)
    print(f"Device: {device}", flush=True)

    model = SentenceTransformer(config.model_name, device=device)
    model_cache_path = config.output_dir / f"model_{model_slug(config.model_name)}"
    model.save(str(model_cache_path))
    embeddings = model.encode(
        metadata[config.text_column].tolist(),
        batch_size=config.batch_size,
        show_progress_bar=True,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    ).astype("float32")

    slug = model_slug(config.model_name)
    scope = "full" if config.max_rows is None else f"sample_{len(metadata)}"
    embedding_path = config.output_dir / f"complaint_embeddings_{slug}_{scope}.npy"
    metadata_path = config.output_dir / f"complaint_metadata_{slug}_{scope}.csv"
    manifest_path = config.output_dir / f"embedding_manifest_{slug}_{scope}.json"

    np.save(embedding_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    manifest = {
        **asdict(config),
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "model_cache_path": str(model_cache_path),
        "rows": int(len(metadata)),
        "embedding_dim": int(embeddings.shape[1]),
        "device": device,
        "scope": scope,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    latest_manifest = config.output_dir / "latest_embedding_manifest.json"
    latest_manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Embeddings: {embedding_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return embedding_path, metadata_path


def main() -> None:
    args = parse_args()
    max_rows = None if args.full_dataset else args.max_rows
    config = EmbeddingConfig(
        input_path=resolve_path(args.input),
        output_dir=resolve_path(args.output_dir),
        model_name=args.model_name,
        text_column=args.text_column,
        max_rows=max_rows,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    build_embeddings(config)


if __name__ == "__main__":
    main()
