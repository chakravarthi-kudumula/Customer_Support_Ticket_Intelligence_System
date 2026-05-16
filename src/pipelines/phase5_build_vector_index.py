"""Phase 5 pipeline: build and check the FAISS index."""

from src.retrieval.faiss_vector_store import FaissIndexConfig, build_faiss_index
from src.retrieval.vector_index_benchmark import run_benchmark


def main() -> None:
    _, manifest_path = build_faiss_index(FaissIndexConfig())
    run_benchmark(faiss_manifest_path=manifest_path)


if __name__ == "__main__":
    main()
