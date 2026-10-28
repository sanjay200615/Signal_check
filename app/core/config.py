"""Runtime configuration for the text verification API."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    max_text_length: int = 10_000
    top_k: int = 3
    minimum_similarity: float = 0.3
    model_load_timeout_seconds: float = 120.0
    evidence_provider: str = "live"

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            max_text_length=_integer_env("MAX_TEXT_LENGTH", cls.max_text_length),
            top_k=_integer_env("TOP_K", cls.top_k),
            minimum_similarity=_float_env(
                "MINIMUM_SIMILARITY", cls.minimum_similarity
            ),
            model_load_timeout_seconds=_float_env(
                "MODEL_LOAD_TIMEOUT_SECONDS", cls.model_load_timeout_seconds
            ),
            evidence_provider=os.getenv("EVIDENCE_PROVIDER", cls.evidence_provider).lower(),
        )
        if settings.max_text_length <= 0 or settings.top_k <= 0:
            raise ValueError("MAX_TEXT_LENGTH and TOP_K must be greater than zero.")
        if not 0 <= settings.minimum_similarity <= 1:
            raise ValueError("MINIMUM_SIMILARITY must be between zero and one.")
        if settings.model_load_timeout_seconds <= 0:
            raise ValueError("MODEL_LOAD_TIMEOUT_SECONDS must be greater than zero.")
        if settings.evidence_provider not in {"local", "live", "tavily"}:
            raise ValueError("EVIDENCE_PROVIDER must be local, live, or tavily.")
        return settings


def _integer_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error
