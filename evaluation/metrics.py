"""Dependency-free classification metrics with explicit abstention handling."""

from collections import Counter


def classification_report(
    gold: list[str],
    predicted: list[str],
    labels: list[str],
    *,
    abstain_label: str | None = None,
) -> dict[str, object]:
    """Score predictions against gold labels.

    Abstentions are counted as predictions of ``abstain_label``. Macro-F1 is
    taken over ``labels``, so an abstain label that is also a gold label is
    scored like any other class.
    """
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length.")
    if not gold:
        raise ValueError("at least one example is required.")

    total = len(gold)
    correct = sum(1 for g, p in zip(gold, predicted) if g == p)
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in gold if g == label),
        }

    report: dict[str, object] = {
        "n": total,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(v["f1"] for v in per_label.values()) / len(labels), 4),
        "per_label": per_label,
        "confusion": _confusion(gold, predicted),
    }
    if abstain_label is not None:
        answered = [(g, p) for g, p in zip(gold, predicted) if p != abstain_label]
        report["abstain_rate"] = round((total - len(answered)) / total, 4)
        report["accuracy_on_answered"] = (
            round(sum(1 for g, p in answered if g == p) / len(answered), 4)
            if answered
            else None
        )
    return report


def coverage_error_curve(
    confidences: list[float],
    correct: list[bool],
    *,
    points: int = 5,
) -> list[dict[str, float]]:
    """Error rate when only the most confident fraction of predictions is kept."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length.")
    if not confidences:
        raise ValueError("at least one example is required.")
    ranked = sorted(zip(confidences, correct), key=lambda item: -item[0])
    curve = []
    for step in range(1, points + 1):
        keep = max(1, round(len(ranked) * step / points))
        kept = ranked[:keep]
        curve.append(
            {
                "coverage": round(keep / len(ranked), 4),
                "error": round(1 - sum(1 for _, ok in kept if ok) / keep, 4),
            }
        )
    return curve


def _confusion(gold: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = {}
    for g, p in zip(gold, predicted):
        table.setdefault(g, Counter())[p] += 1
    return {g: dict(counts) for g, counts in table.items()}
