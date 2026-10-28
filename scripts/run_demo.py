"""Run labelled examples through the real local verification pipeline."""

import json
from pathlib import Path

from app.services.pipeline import build_default_pipeline


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pipeline = build_default_pipeline(ROOT)
    cases = json.loads((ROOT / "data/demo_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        _, status = pipeline.analyze(case["text"])
        print(f"{case['case_id']}: expected={case['expected_status']} actual={status}")


if __name__ == "__main__":
    main()
