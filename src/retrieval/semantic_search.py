import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config import project_path


DEFAULT_MANIFEST = project_path("artifacts/embeddings/latest_embedding_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic search over complaint embeddings.")
    parser.add_argument("--query", required=True, help="Natural-language search query.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for results.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def search(query: str, top_k: int, manifest_path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    manifest_path = resolve_path(manifest_path)
    manifest = load_manifest(manifest_path)
    embeddings = np.load(manifest["embedding_path"])
    metadata = pd.read_csv(manifest["metadata_path"], low_memory=False)

    model_source = manifest.get("model_cache_path", manifest["model_name"])
    local_only = Path(str(model_source)).exists()
    model = SentenceTransformer(model_source, device=detect_device(), local_files_only=local_only)
    query_embedding = model.encode(
        [query],
        normalize_embeddings=bool(manifest.get("normalize_embeddings", True)),
        convert_to_numpy=True,
    ).astype("float32")[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = metadata.iloc[top_indices].copy()
    results.insert(0, "similarity", scores[top_indices])
    results.insert(1, "query", query)

    display_columns = [
        "similarity",
        "Product",
        "Issue",
        "Company",
        "State",
        "Complaint ID",
        "text_transformer",
    ]
    available = [col for col in display_columns if col in results.columns]
    return results[available]


def print_results(results: pd.DataFrame) -> None:
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        text = str(row.get("text_transformer", ""))
        snippet = text[:500].replace("\n", " ")
        print(f"\nResult {rank}")
        print(f"Similarity: {row.get('similarity', 0):.4f}")
        print(f"Product: {row.get('Product', '')}")
        print(f"Issue: {row.get('Issue', '')}")
        print(f"Company: {row.get('Company', '')}")
        print(f"Complaint ID: {row.get('Complaint ID', '')}")
        print(f"Snippet: {snippet}")


def main() -> None:
    args = parse_args()
    results = search(args.query, args.top_k, args.manifest)
    print_results(results)
    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
