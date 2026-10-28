"""Evidence-grounded natural-language explanations for verification results."""

from app.services.pipeline import PipelineClaim


def explain_claim(result: PipelineClaim) -> str:
    """Create a concise explanation from the NLI verdict and retrieved evidence."""
    claim = result.claim.claim_text.rstrip(".")
    evidence_count = len(result.evidence)

    if not evidence_count:
        return (
            f'I could not find direct, sufficiently relevant evidence to verify "{claim}". '
            "The claim is unsubstantiated by the retrieved sources; this is not proof "
            "that it is true or false."
        )

    sources = _source_list(result)
    if result.status == "refuted":
        return (
            f'The claim "{claim}" is likely false based on the retrieved evidence. '
            f"The language model found contradiction signals in {evidence_count} "
            f"retrieved source{'s' if evidence_count != 1 else ''}, including {sources}. "
            "This is a model-assisted assessment and should be checked against the linked sources."
        )
    if result.status == "supported":
        return (
            f'The claim "{claim}" is likely supported by the retrieved evidence. '
            f"The language model found supporting signals across {evidence_count} "
            f"source{'s' if evidence_count != 1 else ''}, including {sources}. "
            "This is evidence-based support, not an absolute guarantee of truth."
        )
    if result.status == "conflicting_evidence":
        return (
            f'The evidence for "{claim}" is conflicting. '
            f"The retrieved sources do not agree, so this claim should not be "
            f"treated as settled. Relevant sources include {sources}."
        )
    return (
        f'The claim "{claim}" remains uncertain. '
        f"The retrieved evidence was not strong or consistent enough to produce "
        f"a reliable verdict. Relevant sources include {sources}."
    )


def _source_list(result: PipelineClaim) -> str:
    names = []
    for item in result.evidence:
        if item.source_name not in names:
            names.append(item.source_name)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"
