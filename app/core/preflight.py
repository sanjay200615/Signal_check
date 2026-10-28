"""Repeatable environment checks for running the research prototype."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import os

from app.core.config import Settings


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_preflight(root: Path, *, load_models: bool = False) -> list[Check]:
    """Check configuration and required artifacts without changing the environment."""
    checks: list[Check] = []
    try:
        settings = Settings.from_environment()
        checks.append(Check("configuration", True, f"provider={settings.evidence_provider}"))
    except ValueError as error:
        return [Check("configuration", False, str(error))]

    for package in (
        "fastapi",
        "httpx",
        "numpy",
        "transformers",
        "PIL",
        "rapidocr_onnxruntime",
        "faster_whisper",
    ):
        checks.append(_package_check(package))

    if settings.evidence_provider == "live":
        checks.append(_package_check("ddgs"))
    elif settings.evidence_provider == "local":
        checks.extend(_local_index_checks(root))
        checks.append(_package_check("faiss"))
        checks.append(_package_check("sentence_transformers"))
    elif settings.evidence_provider == "tavily":
        checks.append(
            Check(
                "tavily_api_key",
                bool(os.getenv("TAVILY_API_KEY")),
                "TAVILY_API_KEY is configured"
                if os.getenv("TAVILY_API_KEY")
                else "TAVILY_API_KEY is not configured",
            )
        )

    if load_models:
        checks.extend(_model_load_checks(settings.evidence_provider, root))
    else:
        checks.append(
            Check(
                "model_loading",
                True,
                "skipped; pass --load-models to perform network/model loading",
            )
        )
    return checks


def _package_check(package: str) -> Check:
    available = importlib.util.find_spec(package) is not None
    return Check(
        f"python:{package}",
        available,
        "import available" if available else "package is not installed",
    )


def _local_index_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    for name in ("data/index/evidence.faiss", "data/index/metadata.json"):
        path = root / name
        checks.append(
            Check(
                name,
                path.is_file(),
                str(path) if path.is_file() else f"missing: {path}",
            )
        )
    return checks


def _model_load_checks(provider: str, root: Path) -> list[Check]:
    checks: list[Check] = []
    try:
        from app.services.verifier import BartNLIModel

        BartNLIModel()
        checks.append(Check("nli_model", True, "facebook/bart-large-mnli loaded"))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        checks.append(Check("nli_model", False, str(error)))

    if provider == "local":
        try:
            from app.services.retriever import EvidenceIndex, SentenceTransformerEmbedder

            SentenceTransformerEmbedder()
            EvidenceIndex.load(
                root / "data/index/evidence.faiss",
                root / "data/index/metadata.json",
            )
            checks.append(Check("local_retrieval", True, "embedding model and index loaded"))
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            checks.append(Check("local_retrieval", False, str(error)))
    return checks
