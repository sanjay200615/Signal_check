from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.common import (
    CaseAnalysisContract,
    CaseInputContract,
    ClaimContract,
    InputType,
    ModalitySignal,
    ProcessingStatus,
    Verdict,
)


def test_common_contract_represents_all_planned_modalities() -> None:
    inputs = [
        CaseInputContract(
            input_id=f"input-{input_type.value}",
            input_type=input_type,
            created_at=datetime.now(timezone.utc),
            content_hash="a" * 64,
        )
        for input_type in InputType
    ]

    analysis = CaseAnalysisContract(
        case_id="case-1",
        status=ProcessingStatus.COMPLETED,
        inputs=inputs,
        claims=[
            ClaimContract(
                claim_id="claim-1",
                claim_text="A claim",
                source_input_ids=["input-text"],
            )
        ],
        signals=[
            ModalitySignal(
                signal_id="signal-1",
                input_id="input-url",
                modality=InputType.URL,
                signal_type="phishing_risk",
                status=ProcessingStatus.COMPLETED,
                score=0.8,
            )
        ],
        verdict=Verdict.DECEPTIVE,
        risk_score=0.8,
    )

    assert {item.input_type for item in analysis.inputs} == set(InputType)
    assert analysis.verdict == Verdict.DECEPTIVE


def test_common_contract_rejects_invalid_scores_and_hashes() -> None:
    with pytest.raises(ValidationError):
        CaseInputContract(
            input_id="input-1",
            input_type=InputType.TEXT,
            created_at=datetime.now(timezone.utc),
            content_hash="short",
        )

    with pytest.raises(ValidationError):
        ModalitySignal(
            signal_id="signal-1",
            input_id="input-1",
            modality=InputType.IMAGE,
            signal_type="manipulation",
            status=ProcessingStatus.COMPLETED,
            score=1.5,
        )
