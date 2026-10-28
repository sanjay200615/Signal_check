"""Semantic evidence indexing and retrieval."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from app.services.corpus import EvidencePassage


class EmbeddingModel(Protocol):
    """Minimal interface required by the FAISS index."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding vector per text."""


class SentenceTransformerEmbedder:
    """Load the configured Sentence Transformer model on demand."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class EvidenceIndex:
    """A persisted FAISS inner-product index with aligned passage metadata."""

    def __init__(self, index: Any, passages: list[EvidencePassage]) -> None:
        if index.ntotal != len(passages):
            raise ValueError("FAISS vector count must match passage metadata count.")
        self._index = index
        self._passages = passages

    @classmethod
    def build(
        cls,
        passages: list[EvidencePassage],
        embedder: EmbeddingModel,
    ) -> "EvidenceIndex":
        if not passages:
            raise ValueError("Cannot build an evidence index without passages.")
        unique_passages = _deduplicate_passages(passages)
        vectors = _normalise_embeddings(
            embedder.encode([passage.text for passage in unique_passages])
        )
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, unique_passages)

    def save(self, index_path: Path, metadata_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(
            json.dumps([asdict(passage) for passage in self._passages], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "EvidenceIndex":
        try:
            import faiss

            index = faiss.read_index(str(index_path))
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to load evidence index: {error}") from error
        if not isinstance(raw_metadata, list):
            raise ValueError("Evidence metadata must contain a JSON list.")
        passages = [EvidencePassage(**item) for item in raw_metadata]
        return cls(index, passages)

    def search(
        self,
        query: str,
        embedder: EmbeddingModel,
        *,
        top_k: int = 3,
    ) -> list[tuple[EvidencePassage, float]]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        query_vector = _normalise_embeddings(embedder.encode([query]))
        if query_vector.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension does not match the FAISS index.")
        scores, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index != -1:
                results.append((self._passages[int(index)], float(score)))
        return results


def _normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embedding model must return a non-empty 2D array.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector.")
    return vectors / norms


def _deduplicate_passages(
    passages: list[EvidencePassage],
) -> list[EvidencePassage]:
    seen: set[tuple[str, str]] = set()
    unique: list[EvidencePassage] = []
    for passage in passages:
        key = (passage.source_url or passage.document_id, passage.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(passage)
    return unique
