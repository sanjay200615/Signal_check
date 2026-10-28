"""URL extraction and pretrained image OCR for multimodal claim analysis."""

from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


class _ArticleTextParser(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.meta_description = ""
        self.text_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attributes = dict(attrs)
            if attributes.get("name", "").lower() == "description":
                self.meta_description = attributes.get("content", "").strip()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if not clean or self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(clean)
        else:
            self.text_parts.append(clean)


def fetch_url_text(url: str, *, timeout: float = 20.0, max_chars: int = 50_000) -> str:
    """Fetch a public web page and return its title plus readable text."""
    normalized_url = url.strip()
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http or https URL.")
    try:
        response = httpx.get(
            normalized_url,
            follow_redirects=True,
            headers={"User-Agent": "SignalCheck/0.1"},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError("The URL could not be fetched.") from error
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type:
        raise ValueError("The URL does not contain an HTML article.")
    parser = _ArticleTextParser()
    parser.feed(response.text)
    title = " ".join(parser.title_parts)
    body = " ".join(parser.text_parts)
    description = " ".join(parser.meta_description.split())
    extracted = " ".join(part for part in (title, description, body) if part).strip()
    if len(extracted) < 30:
        raise ValueError("The URL did not contain enough readable text.")
    return extracted[:max_chars]


def extract_image_text(image_bytes: bytes) -> str:
    """Extract all readable text from an image or screenshot with RapidOCR."""
    from app.services.ocr import read_image

    text = read_image(image_bytes).text
    if len(text.strip()) < 3:
        raise ValueError("No readable claim text was found in the image.")
    return text
