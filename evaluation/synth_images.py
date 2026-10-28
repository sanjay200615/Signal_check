"""Render screenshots with known ground truth for evaluating the image pipeline.

These are synthetic: they test the mechanics (OCR, brand and domain checks) with
exact labels, not real-world generalisation. Brand pages use the brand *name* as
text; no real logos are reproduced.
"""

from io import BytesIO
import textwrap

from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_sms(text: str, sender: str = "Messages", width: int = 720) -> bytes:
    """A phone chat screenshot with one incoming message bubble."""
    body = _font(30)
    lines = textwrap.wrap(text, width=34)
    height = 220 + 42 * len(lines)
    image = Image.new("RGB", (width, height), (242, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 90], fill=(250, 250, 252))
    draw.text((width // 2 - 60, 30), sender, fill=(20, 20, 20), font=_font(32))
    bubble_bottom = 140 + 42 * len(lines) + 30
    draw.rounded_rectangle([30, 120, width - 90, bubble_bottom], radius=28, fill=(226, 226, 232))
    for i, line in enumerate(lines):
        draw.text((55, 140 + 42 * i), line, fill=(15, 15, 15), font=body)
    draw.text((40, bubble_bottom + 20), "Today 10:42", fill=(130, 130, 135), font=_font(22))
    return _png(image)


def render_login_page(
    brand: str,
    address: str,
    *,
    accent: tuple[int, int, int] = (0, 85, 170),
    notice: str | None = None,
    width: int = 900,
    height: int = 620,
) -> bytes:
    """A browser screenshot of a sign-in page showing ``brand`` with ``address`` in the address bar."""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 56], fill=(228, 230, 235))
    draw.rounded_rectangle([90, 10, width - 30, 46], radius=18, fill=(255, 255, 255))
    draw.text((112, 17), address, fill=(40, 40, 40), font=_font(22))
    draw.rectangle([0, 56, width, 150], fill=accent)
    draw.text((40, 78), brand, fill=(255, 255, 255), font=_font(48))
    draw.text((40, 190), f"Sign in to your {brand} account", fill=(30, 30, 30), font=_font(34))
    for i, label in enumerate(("Email or phone", "Password")):
        top = 260 + i * 90
        draw.text((40, top - 28), label, fill=(90, 90, 90), font=_font(22))
        draw.rectangle([40, top, width - 40, top + 50], outline=(160, 160, 160), width=2)
    draw.rounded_rectangle([40, 450, 280, 510], radius=10, fill=accent)
    draw.text((100, 462), "Sign in", fill=(255, 255, 255), font=_font(30))
    if notice:
        draw.text((40, 540), notice, fill=(180, 30, 30), font=_font(22))
    return _png(image)


def render_plain(color: tuple[int, int, int] = (90, 140, 200), size: tuple[int, int] = (640, 480)) -> bytes:
    """A text-free gradient image."""
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    for y in range(size[1]):
        shade = int(255 * y / size[1])
        draw.line([(0, y), (size[0], y)], fill=(color[0], max(0, color[1] - shade // 4), min(255, color[2] + shade // 8)))
    return _png(image)
