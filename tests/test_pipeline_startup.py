from pathlib import Path

import app.services.evidence_provider as evidence_provider
import app.services.verifier as verifier
from app.services.pipeline import build_default_pipeline


class StubNLI:
    def __call__(self, premise: str, hypothesis: str) -> dict[str, dict[str, float]]:
        return {"labels": {"neutral": 1.0}}


def test_live_pipeline_does_not_load_local_embedding_index(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "BartNLIModel", lambda *args, **kwargs: StubNLI())

    class StubLiveProvider:
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(
        evidence_provider,
        "LiveNewsEvidenceProvider",
        StubLiveProvider,
    )

    pipeline = build_default_pipeline(
        Path("missing-root"),
        evidence_provider="live",
    )

    assert pipeline is not None
