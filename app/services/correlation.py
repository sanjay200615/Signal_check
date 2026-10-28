"""Cross-input relationship signals for a verification case."""

from dataclasses import dataclass
import re
from difflib import SequenceMatcher


@dataclass(frozen=True)
class CorrelationResult:
    score: float
    shared_urls: tuple[str, ...]
    shared_entities: tuple[str, ...]
    similar_pairs: tuple[tuple[str, str], ...]


def correlate_texts(items: list[tuple[str, str]]) -> CorrelationResult:
    urls: dict[str, set[str]] = {}
    entities: dict[str, set[str]] = {}
    similar_pairs: list[tuple[str, str]] = []
    for input_id, text in items:
        for url in re.findall(r"https?://[^\s]+", text.lower()):
            urls.setdefault(url.rstrip(".,!?"), set()).add(input_id)
        for entity in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
            entities.setdefault(entity.casefold(), set()).add(input_id)
    shared_urls = tuple(url for url, ids in urls.items() if len(ids) > 1)
    shared_entities = tuple(entity for entity, ids in entities.items() if len(ids) > 1)
    for index, (left_id, left_text) in enumerate(items):
        for right_id, right_text in items[index + 1:]:
            if SequenceMatcher(None, left_text.casefold(), right_text.casefold()).ratio() >= 0.7:
                similar_pairs.append((left_id, right_id))
    score = min(
        1.0,
        0.45 * bool(shared_urls)
        + 0.3 * bool(shared_entities)
        + 0.25 * bool(similar_pairs),
    )
    return CorrelationResult(score, shared_urls, shared_entities, tuple(similar_pairs))
