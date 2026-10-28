import pytest

from evaluation.datasets import load_claims, load_scam_cases
from evaluation.metrics import classification_report, coverage_error_curve


def test_report_matches_hand_computed_values():
    gold = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    report = classification_report(gold, predicted, ["a", "b"])
    assert report["accuracy"] == 0.75
    # a: P=1, R=0.5, F1=2/3.  b: P=2/3, R=1, F1=0.8.  macro = 0.7333
    assert report["per_label"]["a"]["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert report["per_label"]["b"]["f1"] == pytest.approx(0.8, abs=1e-4)
    assert report["macro_f1"] == pytest.approx(0.7333, abs=1e-4)
    assert report["confusion"] == {"a": {"a": 1, "b": 1}, "b": {"b": 2}}


def test_abstention_is_reported_separately():
    gold = ["deceptive", "genuine", "genuine", "deceptive"]
    predicted = ["deceptive", "needs_review", "genuine", "needs_review"]
    report = classification_report(
        gold, predicted, ["deceptive", "genuine"], abstain_label="needs_review"
    )
    assert report["abstain_rate"] == 0.5
    assert report["accuracy_on_answered"] == 1.0
    assert report["accuracy"] == 0.5


def test_all_abstained_has_no_answered_accuracy():
    report = classification_report(
        ["a"], ["abstain"], ["a"], abstain_label="abstain"
    )
    assert report["abstain_rate"] == 1.0
    assert report["accuracy_on_answered"] is None


def test_report_rejects_bad_input():
    with pytest.raises(ValueError):
        classification_report(["a"], [], ["a"])
    with pytest.raises(ValueError):
        classification_report([], [], ["a"])


def test_coverage_error_curve_is_monotone_for_a_calibrated_ranking():
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    correct = [True, True, True, False, True, False]
    curve = coverage_error_curve(confidences, correct, points=3)
    assert [point["coverage"] for point in curve] == [0.3333, 0.6667, 1.0]
    assert curve[0]["error"] == 0.0
    assert curve[-1]["error"] == pytest.approx(0.3333, abs=1e-4)


def test_claims_dataset_is_well_formed_and_balanced():
    claims = load_claims()
    assert len({item["id"] for item in claims}) == len(claims)
    labels = {item["label"] for item in claims}
    assert labels == {"supported", "refuted", "insufficient_evidence"}
    assert all(item["text"].strip() for item in claims)


def test_scam_dataset_is_well_formed():
    cases = load_scam_cases()
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["label"] for case in cases} == {"deceptive", "genuine", "needs_review"}
    for case in cases:
        assert case["message"].strip()
        if case["url"]:
            assert case["url"].startswith(("http://", "https://"))
