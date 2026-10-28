from pathlib import Path

import numpy as np

from app.services.corpus import load_documents, split_into_passages
from app.services.retriever import EvidenceIndex


class KeywordEmbedder:
    """Small deterministic embedder for retrieval unit tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [
                [
                    sum(
                        1
                        for word in text.lower().split()
                        if sum(ord(character) for character in word) % 16 == bucket
                    )
                    for bucket in range(16)
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_build_save_load_and_search_preserves_metadata(tmp_path: Path) -> None:
    documents = load_documents(Path("data/evidence/documents.json"))
    passages = split_into_passages(documents, max_words=80, overlap_words=15)
    embedder = KeywordEmbedder()
    evidence_index = EvidenceIndex.build(passages, embedder)

    index_path = tmp_path / "evidence.faiss"
    metadata_path = tmp_path / "metadata.json"
    evidence_index.save(index_path, metadata_path)
    loaded_index = EvidenceIndex.load(index_path, metadata_path)

    results = loaded_index.search("soap and hands", embedder, top_k=1)

    assert len(results) == 1
    passage, similarity = results[0]
    assert passage.passage_id == "cdc-handwashing_p1"
    assert passage.source_name == "Centers for Disease Control and Prevention"
    assert similarity > 0


def test_empty_query_is_rejected() -> None:
    documents = load_documents(Path("data/evidence/documents.json"))
    evidence_index = EvidenceIndex.build(
        split_into_passages(documents),
        KeywordEmbedder(),
    )

    try:
        evidence_index.search("  ", KeywordEmbedder())
    except ValueError as error:
        assert str(error) == "query must not be empty."
    else:
        raise AssertionError("Expected empty query to be rejected.")


def test_duplicate_passages_are_not_indexed_twice() -> None:
    documents = load_documents(Path("data/evidence/documents.json"))
    passages = split_into_passages(documents)
    duplicate = passages + [passages[0]]

    index = EvidenceIndex.build(duplicate, KeywordEmbedder())

    assert index._index.ntotal == len(passages)
