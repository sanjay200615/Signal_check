"""Speech-to-text for voice notes and calls, using faster-whisper locally (CPU, no key)."""

from dataclasses import dataclass
from io import BytesIO
import logging
import os
import threading
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "small"
MAX_AUDIO_SECONDS = 600.0  # only the first ten minutes are transcribed, bounding CPU time
MIN_LANGUAGE_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Transcript:
    """What was said. ``text`` is the English text used for analysis."""

    text: str
    original_text: str
    language: str
    language_probability: float
    duration_seconds: float
    translated: bool
    truncated: bool = False

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())

    def metadata(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "duration_seconds": round(self.duration_seconds, 1),
            "translated": self.translated,
            "truncated": self.truncated,
            "original_text": self.original_text if self.translated else None,
            "source": "whisper",
        }


class AudioTranscriber(Protocol):
    def transcribe(self, content: bytes) -> Transcript | str:
        """Return a transcript or raise an explicit processing error."""


@dataclass(frozen=True)
class TranscriptAudioTranscriber:
    """Adapter for a transcript the user supplied themselves."""

    transcript: str

    def transcribe(self, content: bytes) -> str:
        if not content:
            raise ValueError("audio must not be empty.")
        if not self.transcript.strip():
            raise RuntimeError("audio transcription is not configured.")
        return self.transcript.strip()


class WhisperTranscriber:
    """Local faster-whisper transcription with language detection and translation to English."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("WHISPER_MODEL", DEFAULT_MODEL)
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
                except (ImportError, OSError) as error:
                    raise RuntimeError("The speech recognition model could not be loaded.") from error
            return self._model

    def transcribe(self, content: bytes) -> Transcript:
        if not content:
            raise ValueError("audio must not be empty.")
        model = self._load()
        try:
            segments, info = model.transcribe(BytesIO(content), beam_size=1, vad_filter=True)
            original, truncated = _collect(segments)
        except (ValueError, EOFError) as error:  # PyAV's InvalidDataError is a ValueError
            raise ValueError("The uploaded file is not valid audio.") from error
        except Exception as error:  # other PyAV error types for undecodable data
            if type(error).__module__.startswith("av"):
                raise ValueError("The uploaded file is not valid audio.") from error
            raise

        text, translated = original, False
        if original.strip() and info.language != "en" and info.language_probability >= MIN_LANGUAGE_CONFIDENCE:
            segments, _ = model.transcribe(
                BytesIO(content), task="translate", language=info.language, beam_size=1, vad_filter=True
            )
            text, _ = _collect(segments)
            translated = True
        return Transcript(
            text=text.strip(),
            original_text=original.strip(),
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(info.duration),
            translated=translated,
            truncated=truncated or info.duration > MAX_AUDIO_SECONDS,
        )


def _collect(segments) -> tuple[str, bool]:
    """Join segment text, stopping once the duration cap is passed."""
    parts, truncated = [], False
    for segment in segments:
        if segment.start > MAX_AUDIO_SECONDS:
            truncated = True
            break
        parts.append(segment.text.strip())
    return " ".join(part for part in parts if part), truncated


_shared: WhisperTranscriber | None = None


def get_transcriber() -> WhisperTranscriber:
    """The shared transcriber (the model loads once, on first use)."""
    global _shared
    if _shared is None:
        _shared = WhisperTranscriber()
    return _shared
