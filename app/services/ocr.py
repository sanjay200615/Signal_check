"""Full-page OCR for screenshots and photos, using RapidOCR (ONNX, CPU, no key)."""

from dataclasses import dataclass
from io import BytesIO
import re
import threading

from PIL import Image

_MAX_SIDE = 1800
_OCR_SCALE = 1.1
_MIN_CONFIDENCE = 0.5
_engine = None
_engine_lock = threading.Lock()


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    top: float
    left: float


@dataclass(frozen=True)
class OcrResult:
    lines: tuple[OcrLine, ...]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def read_image(image_bytes: bytes) -> OcrResult:
    """Return the recognised lines in reading order. Raises ValueError for non-images."""
    if not image_bytes:
        raise ValueError("image must not be empty.")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as error:
        raise ValueError("The uploaded file is not a valid image.") from error

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    else:
        image = image.convert("RGB")
    scale = min(_OCR_SCALE, _MAX_SIDE / max(image.size))
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))

    try:
        import numpy as np

        raw, _ = _get_engine()(np.asarray(image)[:, :, ::-1])  # RapidOCR expects BGR
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("Image OCR is unavailable. The RapidOCR engine could not be loaded.") from error

    lines = []
    for box, text, confidence in raw or []:
        text = " ".join(str(text).split())
        if text and float(confidence) >= _MIN_CONFIDENCE:
            lines.append(
                OcrLine(
                    text=text,
                    confidence=float(confidence),
                    top=min(point[1] for point in box),
                    left=min(point[0] for point in box),
                )
            )
    return OcrResult(tuple(_reading_order(lines)))


def _reading_order(lines: list[OcrLine]) -> list[OcrLine]:
    """Group lines whose tops are within half a line height, then sort left to right."""
    ordered = sorted(lines, key=lambda line: (line.top, line.left))
    rows: list[list[OcrLine]] = []
    for line in ordered:
        if rows and abs(rows[-1][0].top - line.top) < 14:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda item: item.left)]


_DOMAIN_IN_TEXT = re.compile(
    r"(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})(?:/[^\s]*)?", re.IGNORECASE
)


def domains_in_text(text: str) -> list[str]:
    """Domain-like tokens in OCR text, lower-cased, order preserved, e-mail hosts excluded."""
    found: list[str] = []
    for match in _DOMAIN_IN_TEXT.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "@":
            continue
        domain = match.group(1).lower()
        if domain not in found:
            found.append(domain)
    return found


def claim_text_from_ocr(text: str) -> str:
    """Drop short fragments that are screenshot chrome ("Messages", "Today 10:42", "Sign in").

    A line is kept if it has at least three words or ends like a sentence. Remaining lines are
    joined with spaces, because OCR line breaks usually fall mid-sentence.
    """
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if len(line.split()) >= 3 or line.endswith((".", "!", "?")):
            kept.append(line)
    return " ".join(kept)


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine
