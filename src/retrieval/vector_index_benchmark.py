import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache"))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config import project_path


DEFAULT_EMBEDDING_MANIFEST = project_path("artifacts/embeddings/latest_embedding_manifest.json")
DEFAULT_FAISS_MANIFEST = project_path("artifacts/vector_indexes/latest_faiss_manifest.json")
DEFAULT_REPORT_DIR = project_path("artifacts/reports")
DEFAULT_QUERIES = [
    "My account was charged twice",
    "Debt collector keeps calling me about a debt I do not owe",
    "Mortgage company will not fix escrow mistake",
    "Credit card company denied my fraud dispute",
    "Student loan servicer gave me wrong payment information",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NumPy cosine search with FAISS vector search.")
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST)
    parser.add_argument("--faiss-manifest", type=Path, default=DEFAULT_FAISS_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else project_path(path)


def load_json(path: Path) -> dict:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sentence_model(manifest: dict) -> SentenceTransformer:
    model_source = manifest.get("model_cache_path") or manifest["model_name"]
    local_only = Path(str(model_source)).exists()
    return SentenceTransformer(model_source, device=detect_device(), local_files_only=local_only)


def timed_ms(fn) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1000, value


def benchmark_query(
    query: str,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    index,
    top_k: int,
    repeat: int,
) -> dict:
    numpy_times = []
    faiss_times = []
    numpy_indices = None
    faiss_indices = None

    for _ in range(repeat):
        numpy_ms, numpy_result = timed_ms(lambda: embeddings @ query_embedding)
        scores = numpy_result
        top_indices = np.argsort(scores)[::-1][:top_k]
        numpy_times.append(numpy_ms)
        numpy_indices = top_indices

        faiss_ms, faiss_result = timed_ms(lambda: index.search(query_embedding.reshape(1, -1), top_k))
        _, indices = faiss_result
        faiss_times.append(faiss_ms)
        faiss_indices = indices[0]

    overlap = len(set(numpy_indices.tolist()) & set(faiss_indices.tolist())) / top_k
    exact_order_match = bool(np.array_equal(numpy_indices, faiss_indices))
    return {
        "query": query,
        "top_k": top_k,
        "repeat": repeat,
        "numpy_ms_mean": round(float(np.mean(numpy_times)), 4),
        "numpy_ms_p95": round(float(np.percentile(numpy_times, 95)), 4),
        "faiss_ms_mean": round(float(np.mean(faiss_times)), 4),
        "faiss_ms_p95": round(float(np.percentile(faiss_times, 95)), 4),
        "top_k_overlap": round(float(overlap), 4),
        "exact_order_match": exact_order_match,
    }


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(report_path: Path, rows: list[dict], faiss_manifest: dict) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Phase 5 Vector Database Benchmark",
        "",
        "FAISS is used here as the vector search index for complaint embeddings.",
        "",
        "## Index",
        "",
        f"- Index type: `{faiss_manifest['index_type']}`",
        f"- Rows: `{faiss_manifest['rows']:,}`",
        f"- Embedding dimension: `{faiss_manifest['embedding_dim']}`",
        f"- Metric: `{faiss_manifest['metric']}`",
        f"- Similarity: `{faiss_manifest['similarity_interpretation']}`",
        f"- Build time: `{faiss_manifest['build_seconds']}` seconds",
        "",
        "## Latency And Recall Check",
        "",
        markdown_table(df),
        "",
        "For normalized SBERT embeddings, FAISS inner product scores match cosine similarity.",
        "`flat_ip` is exact search, so it should return the same top-k results as NumPy while keeping search logic inside FAISS.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    embedding_manifest_path: Path = DEFAULT_EMBEDDING_MANIFEST,
    faiss_manifest_path: Path = DEFAULT_FAISS_MANIFEST,
    top_k: int = 5,
    repeat: int = 5,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> tuple[Path, Path]:
    embedding_manifest = load_json(resolve_path(embedding_manifest_path))
    faiss_manifest = load_json(resolve_path(faiss_manifest_path))

    embeddings = np.load(embedding_manifest["embedding_path"]).astype("float32", copy=False)
    embeddings = np.ascontiguousarray(embeddings)
    model = load_sentence_model(faiss_manifest)

    query_embeddings = model.encode(
        DEFAULT_QUERIES,
        normalize_embeddings=bool(faiss_manifest.get("normalize_embeddings", True)),
        convert_to_numpy=True,
    ).astype("float32")

    # Import FAISS after SentenceTransformer loads. On this Mac setup,
    # loading FAISS first caused native crashes in a few test runs.
    import faiss

    index = faiss.read_index(faiss_manifest["index_path"])
    rows = [
        benchmark_query(query, query_embedding, embeddings, index, top_k, repeat)
        for query, query_embedding in zip(DEFAULT_QUERIES, query_embeddings)
    ]

    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "phase5_vector_index_benchmark.csv"
    md_path = report_dir / "phase5_vector_index_benchmark.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_markdown_report(md_path, rows, faiss_manifest)

    print(f"Benchmark CSV: {csv_path}", flush=True)
    print(f"Benchmark report: {md_path}", flush=True)
    return csv_path, md_path


def main() -> None:
    args = parse_args()
    run_benchmark(
        embedding_manifest_path=resolve_path(args.embedding_manifest),
        faiss_manifest_path=resolve_path(args.faiss_manifest),
        top_k=args.top_k,
        repeat=args.repeat,
        report_dir=resolve_path(args.report_dir),
    )


if __name__ == "__main__":
    main()
