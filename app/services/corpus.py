"""Load and prepare the local evidence corpus for later retrieval."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceDocument:
    document_id: str
    title: str
    source_name: str
    source_url: str | None
    published_date: str | None
    retrieved_date: str
    source_type: str
    trust_notes: str
    text: str
    last_updated_date: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    document_id: str
    title: str
    source_name: str
    source_url: str | None
    published_date: str | None
    retrieved_date: str
    source_type: str
    trust_notes: str
    text: str
    last_updated_date: str | None = None
    domain: str | None = None


REQUIRED_FIELDS = {
    "document_id",
    "title",
    "source_name",
    "source_url",
    "published_date",
    "retrieved_date",
    "source_type",
    "trust_notes",
    "text",
}


def load_documents(path: Path) -> list[EvidenceDocument]:
    """Load and validate evidence documents from a JSON file."""
    try:
        raw_documents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load evidence corpus at {path}: {error}") from error

    if not isinstance(raw_documents, list):
        raise ValueError("Evidence corpus must contain a JSON list.")

    documents = [_parse_document(item, index) for index, item in enumerate(raw_documents)]
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Evidence document_id values must be unique.")
    return documents


def split_into_passages(
    documents: list[EvidenceDocument],
    *,
    max_words: int = 80,
    overlap_words: int = 15,
) -> list[EvidencePassage]:
    """Split documents into stable, metadata-preserving word passages."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between zero and max_words - 1.")

    passages: list[EvidencePassage] = []
    for document in documents:
        words = document.text.split()
        step = max_words - overlap_words
        for passage_number, start in enumerate(range(0, len(words), step), start=1):
            passage_words = words[start : start + max_words]
            if not passage_words:
                continue
            passages.append(
                EvidencePassage(
                    passage_id=f"{document.document_id}_p{passage_number}",
                    document_id=document.document_id,
                    title=document.title,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    published_date=document.published_date,
                    retrieved_date=document.retrieved_date,
                    source_type=document.source_type,
                    trust_notes=document.trust_notes,
                    text=" ".join(passage_words),
                    last_updated_date=document.last_updated_date,
                    domain=document.domain,
                )
            )
    return passages


def _parse_document(item: Any, index: int) -> EvidenceDocument:
    if not isinstance(item, dict):
        raise ValueError(f"Evidence document at index {index} must be an object.")
    missing_fields = REQUIRED_FIELDS - item.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Evidence document at index {index} is missing: {missing}.")

    string_fields = (
        "document_id",
        "title",
        "source_name",
        "retrieved_date",
        "source_type",
        "trust_notes",
        "text",
    )
    for field in string_fields:
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"Evidence field '{field}' must be a non-empty string.")

    optional_string_fields = ("source_url", "published_date")
    for field in optional_string_fields:
        if item[field] is not None and not isinstance(item[field], str):
            raise ValueError(f"Evidence field '{field}' must be a string or null.")
        if isinstance(item[field], str) and not item[field].strip():
            raise ValueError(f"Evidence field '{field}' must not be blank.")
    for field in ("last_updated_date", "domain"):
        if field in item and item[field] is not None and not isinstance(item[field], str):
            raise ValueError(f"Evidence field '{field}' must be a string or null.")

    values = {field: item[field] for field in REQUIRED_FIELDS}
    values["last_updated_date"] = item.get("last_updated_date")
    values["domain"] = item.get("domain")
    return EvidenceDocument(**values)
