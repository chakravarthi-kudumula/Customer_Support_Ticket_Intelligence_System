import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.utils.config import portable_project_path, project_path


DEFAULT_EMBEDDING_MANIFEST = project_path("artifacts/embeddings/latest_embedding_manifest.json")
DEFAULT_INDEX_DIR = project_path("artifacts/vector_indexes")


@dataclass(frozen=True)
class FaissIndexConfig:
    embedding_manifest_path: Path = DEFAULT_EMBEDDING_MANIFEST
    output_dir: Path = DEFAULT_INDEX_DIR
    index_type: str = "flat_ip"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS vector index from complaint embeddings.")
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--index-type",
        choices=["flat_ip", "hnsw_ip"],
        default="flat_ip",
        help="flat_ip is exact cosine search for normalized embeddings; hnsw_ip is approximate.",
    )
    parser.add_argument("--hnsw-m", type=int, default=FaissIndexConfig.hnsw_m)
    parser.add_argument("--hnsw-ef-construction", type=int, default=FaissIndexConfig.hnsw_ef_construction)
    parser.add_argument("--hnsw-ef-search", type=int, default=FaissIndexConfig.hnsw_ef_search)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def index_slug(index_type: str, embedding_manifest: dict) -> str:
    model_name = embedding_manifest["model_name"]
    model_slug = model_name.replace("sentence-transformers/", "").replace("/", "_").replace("-", "_").lower()
    return f"faiss_{index_type}_{model_slug}_{embedding_manifest.get('scope', 'unknown')}"


def create_index(dim: int, config: FaissIndexConfig):
    import faiss
    if config.index_type == "flat_ip":
        return faiss.IndexFlatIP(dim)

    if config.index_type == "hnsw_ip":
        index = faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.hnsw_ef_construction
        index.hnsw.efSearch = config.hnsw_ef_search
        return index

    raise ValueError(f"Unsupported FAISS index type: {config.index_type}")


def build_faiss_index(config: FaissIndexConfig) -> tuple[Path, Path]:
    embedding_manifest_path = resolve_path(config.embedding_manifest_path)
    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_manifest = load_json(embedding_manifest_path)
    embedding_path = portable_project_path(embedding_manifest["embedding_path"])
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    embeddings = np.load(embedding_path).astype("float32", copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    embeddings = np.ascontiguousarray(embeddings)

    rows, dim = embeddings.shape
    index = create_index(dim, config)
    import faiss

    print(f"Building FAISS index: {config.index_type}", flush=True)
    print(f"Rows: {rows:,}", flush=True)
    print(f"Embedding dim: {dim}", flush=True)
    start = time.perf_counter()
    index.add(embeddings)
    build_seconds = time.perf_counter() - start

    slug = index_slug(config.index_type, embedding_manifest)
    index_path = output_dir / f"{slug}.index"
    manifest_path = output_dir / f"{slug}_manifest.json"

    faiss.write_index(index, str(index_path))
    manifest = {
        **asdict(config),
        "embedding_manifest_path": str(embedding_manifest_path),
        "output_dir": str(output_dir),
        "index_path": str(index_path),
        "metadata_path": embedding_manifest["metadata_path"],
        "model_name": embedding_manifest["model_name"],
        "model_cache_path": embedding_manifest.get("model_cache_path"),
        "text_column": embedding_manifest.get("text_column", "text_transformer"),
        "normalize_embeddings": bool(embedding_manifest.get("normalize_embeddings", True)),
        "metric": "inner_product",
        "similarity_interpretation": "cosine_similarity_when_embeddings_are_normalized",
        "rows": int(rows),
        "embedding_dim": int(dim),
        "faiss_version": faiss.__version__,
        "build_seconds": round(build_seconds, 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    latest_manifest = output_dir / "latest_faiss_manifest.json"
    latest_manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Index: {index_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Build seconds: {build_seconds:.4f}", flush=True)
    return index_path, manifest_path


def main() -> None:
    args = parse_args()
    config = FaissIndexConfig(
        embedding_manifest_path=resolve_path(args.embedding_manifest),
        output_dir=resolve_path(args.output_dir),
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )
    build_faiss_index(config)


if __name__ == "__main__":
    main()
