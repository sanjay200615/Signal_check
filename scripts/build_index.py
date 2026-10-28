"""Build the persisted FAISS index for the local evidence corpus."""

from pathlib import Path

from app.services.corpus import load_documents, split_into_passages
from app.services.retriever import EvidenceIndex, SentenceTransformerEmbedder


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data/evidence/documents.json"
INDEX_PATH = ROOT / "data/index/evidence.faiss"
METADATA_PATH = ROOT / "data/index/metadata.json"


def main() -> None:
    documents = load_documents(CORPUS_PATH)
    passages = split_into_passages(documents)
    embedder = SentenceTransformerEmbedder()
    index = EvidenceIndex.build(passages, embedder)
    index.save(INDEX_PATH, METADATA_PATH)
    print(f"Indexed {len(passages)} passages at {INDEX_PATH}")


if __name__ == "__main__":
    main()
