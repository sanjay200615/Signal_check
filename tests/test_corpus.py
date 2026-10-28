import json
from pathlib import Path

import pytest

from app.services.corpus import load_documents, split_into_passages


CORPUS_PATH = Path("data/evidence/documents.json")


def test_loads_curated_documents_with_provenance() -> None:
    documents = load_documents(CORPUS_PATH)

    assert len(documents) == 8
    assert documents[0].document_id == "cdc-handwashing"
    assert documents[0].source_name == "Centers for Disease Control and Prevention"
    assert documents[0].source_url
    assert {document.source_type for document in documents} >= {
        "official_consumer_guidance",
        "official_cybersecurity_guidance",
        "official_public_safety_guidance",
    }


def test_passages_preserve_source_metadata_and_have_stable_ids() -> None:
    documents = load_documents(CORPUS_PATH)

    passages = split_into_passages(documents, max_words=12, overlap_words=2)

    assert passages
    assert passages[0].passage_id == "cdc-handwashing_p1"
    assert passages[0].document_id == "cdc-handwashing"
    assert passages[0].source_url == documents[0].source_url
    assert all(passage.text for passage in passages)


def test_invalid_corpus_document_is_rejected(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps([{"document_id": "missing-fields"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        load_documents(invalid_path)
