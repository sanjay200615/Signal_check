import json
from pathlib import Path


def test_demo_cases_cover_required_outcomes() -> None:
    cases = json.loads(Path("data/demo_cases.json").read_text(encoding="utf-8"))
    statuses = {case["expected_status"] for case in cases}

    assert statuses == {"supported", "refuted", "insufficient_evidence"}
    assert all(case["text"].strip() for case in cases)
