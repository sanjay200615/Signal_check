"""Lightweight image signals used before advanced visual models are added."""

from dataclasses import dataclass
from io import BytesIO

from PIL import Image


_EDITING_SOFTWARE = ("photoshop", "gimp", "canva", "picsart", "snapseed", "lightroom", "pixlr", "affinity", "paint.net")


@dataclass(frozen=True)
class ImageAssessment:
    width: int
    height: int
    format: str
    has_metadata: bool
    perceptual_hash: str
    software: str | None = None
    camera: str | None = None
    taken_at: str | None = None
    has_gps: bool = False
    edited_with_software: bool = False


def assess_image(content: bytes) -> ImageAssessment:
    if not content:
        raise ValueError("image must not be empty.")
    try:
        with Image.open(BytesIO(content)) as image:
            pixels = image.convert("L").resize((8, 8))
            values = list(pixels.getdata())
            average = sum(values) / len(values)
            image_hash = "".join("1" if value >= average else "0" for value in values)
            exif = image.getexif()
            software = _text(exif.get(305))
            camera = " ".join(part for part in (_text(exif.get(271)), _text(exif.get(272))) if part) or None
            exif_ifd = exif.get_ifd(0x8769) if exif else {}
            return ImageAssessment(
                width=image.width,
                height=image.height,
                format=image.format or "unknown",
                has_metadata=bool(exif),
                perceptual_hash=f"{int(image_hash, 2):016x}",
                software=software,
                camera=camera,
                taken_at=_text(exif_ifd.get(36867)) or _text(exif.get(306)),
                has_gps=bool(exif.get_ifd(0x8825)) if exif else False,
                edited_with_software=bool(software) and any(name in software.lower() for name in _EDITING_SOFTWARE),
            )
    except (OSError, ValueError) as error:
        raise ValueError("uploaded content is not a valid image.") from error


def _text(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().strip("\x00") or None if value is not None else None
