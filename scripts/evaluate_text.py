"""Evaluate final status on the manually reviewed local text set."""

import json
from collections import Counter
from pathlib import Path

from app.services.pipeline import build_default_pipeline


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = json.loads(
        (ROOT / "data/evaluation/text_cases.json").read_text(encoding="utf-8")
    )
    pipeline = build_default_pipeline(ROOT)
    actual: list[str] = []
    expected: list[str] = []
    for case in cases:
        _, status = pipeline.analyze(case["text"])
        expected.append(case["expected_status"])
        actual.append(status)
        print(f"{case['case_id']}: expected={case['expected_status']} actual={status}")

    correct = sum(left == right for left, right in zip(expected, actual))
    print(f"cases={len(cases)} correct={correct} accuracy={correct / len(cases):.3f}")
    print(f"confusion_counts={dict(Counter(zip(expected, actual)))}")


if __name__ == "__main__":
    main()
