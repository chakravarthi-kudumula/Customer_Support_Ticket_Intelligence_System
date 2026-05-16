"""Phase 4 pipeline: build complaint embeddings for semantic search."""

from src.embeddings.build_embeddings import main as build_embeddings


def main() -> None:
    build_embeddings()


if __name__ == "__main__":
    main()
