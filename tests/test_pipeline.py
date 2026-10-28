from pathlib import Path

import numpy as np

from app.services.corpus import load_documents, split_into_passages
from app.services.evidence_provider import LocalEvidenceProvider
from app.services.evidence_provider import RetrievedEvidence
from app.services.pipeline import TextAnalysisPipeline, _search_queries_for_claim
from app.services.retriever import EvidenceIndex


class Embedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class NLI:
    def __call__(self, premise: str, hypothesis: str) -> dict[str, dict[str, float]]:
        return {"labels": {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}}


class SelectiveEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0] if "zzquery" not in text else [0, 1, 0, 0] for text in texts],
            dtype=np.float32,
        )


def test_pipeline_returns_claims_evidence_and_status() -> None:
    documents = load_documents(Path("data/evidence/documents.json"))
    pipeline = TextAnalysisPipeline(
        LocalEvidenceProvider(
            EvidenceIndex.build(split_into_passages(documents), Embedder()),
            Embedder(),
        ),
        NLI(),
        top_k=1,
    )

    claims, overall = pipeline.analyze(
        "Washing hands with soap and water helps protect people from getting sick."
    )

    assert overall == "supported"
    assert claims[0].status == "supported"
    assert claims[0].evidence[0].nli_label == "entailment"
    assert claims[0].evidence[0].source_quality == "official"


def test_low_similarity_evidence_does_not_create_a_verdict() -> None:
    documents = load_documents(Path("data/evidence/documents.json"))
    pipeline = TextAnalysisPipeline(
        LocalEvidenceProvider(
            EvidenceIndex.build(split_into_passages(documents), SelectiveEmbedder()),
            SelectiveEmbedder(),
        ),
        NLI(),
        top_k=1,
        minimum_similarity=0.9,
    )

    claims, overall = pipeline.analyze("This claim is zzquery and unrelated to the corpus.")

    assert claims[0].status == "insufficient_evidence"
    assert overall == "insufficient_evidence"


def test_life_status_claim_uses_claim_preserving_query_variants() -> None:
    assert _search_queries_for_claim("Trump is dead") == [
        "Trump is dead",
        "Trump death",
        "Trump alive",
    ]
    assert _search_queries_for_claim("NASA published a climate report.") == [
        "NASA published a climate report."
    ]


def test_pipeline_keeps_unrelated_life_status_results_out_of_nli() -> None:
    class Provider:
        def search(self, query: str, *, top_k: int):
            return [
                RetrievedEvidence(
                    "other-person",
                    "Trump ally Lindsey Graham dies",
                    "Example",
                    "https://example.test/other",
                    None,
                    "Lindsey Graham died.",
                    0.9,
                    "live_news_result",
                    "reputable",
                    "2026-10-27",
                    "live",
                    0.95,
                )
            ]

    class NLI:
        def __init__(self):
            self.called = False

        def __call__(self, premise, hypothesis):
            self.called = True
            return {"labels": {"entailment": 1.0}}

    nli = NLI()
    claims, status = TextAnalysisPipeline(Provider(), nli).analyze("Trump is dead")

    assert status == "insufficient_evidence"
    assert claims[0].evidence == []
    assert nli.called is False


def test_unrelated_evidence_is_rejected_even_when_retrieval_returns_it() -> None:
    documents = load_documents(Path("data/evidence/documents.json"))
    pipeline = TextAnalysisPipeline(
        LocalEvidenceProvider(
            EvidenceIndex.build(split_into_passages(documents), Embedder()),
            Embedder(),
        ),
        NLI(),  # would "entail" anything, so only the relevance gate can stop it
        top_k=1,
    )

    claims, overall = pipeline.analyze("NASA published a climate report.")

    assert overall == "insufficient_evidence"
    assert claims[0].evidence == []
