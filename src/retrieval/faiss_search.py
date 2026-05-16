import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config import project_path


DEFAULT_MANIFEST = project_path("artifacts/vector_indexes/latest_faiss_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic search over a FAISS complaint vector index.")
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
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sentence_model(manifest: dict) -> SentenceTransformer:
    model_source = manifest.get("model_cache_path") or manifest["model_name"]
    local_only = Path(str(model_source)).exists()
    return SentenceTransformer(model_source, device=detect_device(), local_files_only=local_only)


def search(query: str, top_k: int, manifest_path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    metadata = pd.read_csv(manifest["metadata_path"], low_memory=False)

    model = load_sentence_model(manifest)
    query_embedding = model.encode(
        [query],
        normalize_embeddings=bool(manifest.get("normalize_embeddings", True)),
        convert_to_numpy=True,
    ).astype("float32")

    # On Apple Silicon, importing FAISS before SentenceTransformer model loading can
    # crash the process in some local environments. Keep FAISS lazy and late.
    import faiss

    index = faiss.read_index(manifest["index_path"])
    start = time.perf_counter()
    scores, indices = index.search(query_embedding, top_k)
    search_ms = (time.perf_counter() - start) * 1000

    top_indices = indices[0]
    results = metadata.iloc[top_indices].copy()
    results.insert(0, "similarity", scores[0])
    results.insert(1, "query", query)
    results.insert(2, "rank", range(1, len(results) + 1))
    results.insert(3, "search_ms", round(search_ms, 4))

    display_columns = [
        "rank",
        "similarity",
        "search_ms",
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
    if results.empty:
        print("No results found.")
        return

    search_ms = results["search_ms"].iloc[0] if "search_ms" in results.columns else None
    if search_ms is not None:
        print(f"FAISS search latency: {search_ms:.4f} ms")

    for _, row in results.iterrows():
        text = str(row.get("text_transformer", ""))
        snippet = text[:500].replace("\n", " ")
        print(f"\nResult {int(row.get('rank', 0))}")
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
