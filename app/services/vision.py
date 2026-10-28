"""CLIP-based image/text similarity and zero-shot labelling (local, CPU, no key)."""

from io import BytesIO
import os
import threading

import numpy as np
from PIL import Image

MODEL_NAME = "openai/clip-vit-base-patch32"
_lock = threading.Lock()
_model = None
_processor = None


def vision_enabled() -> bool:
    return os.getenv("VISION_MODELS", "on").strip().lower() not in {"off", "0", "false", "no"}


def _load():
    global _model, _processor
    if not vision_enabled():
        raise RuntimeError("Vision models are disabled (VISION_MODELS=off).")
    with _lock:
        if _model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor

                _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                _model = CLIPModel.from_pretrained(MODEL_NAME).eval()
            except (ImportError, OSError) as error:
                raise RuntimeError("The CLIP vision model could not be loaded.") from error
        return _model, _processor


def _open(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise ValueError("The uploaded file is not a valid image.") from error


def embed_images(images: list[bytes]) -> np.ndarray:
    """L2-normalised CLIP image embeddings, shape (n, d)."""
    import torch

    model, processor = _load()
    pixels = processor(images=[_open(data) for data in images], return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**pixels)
    return _normalise(features.numpy())


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalised CLIP text embeddings (text beyond 77 tokens is truncated)."""
    import torch

    model, processor = _load()
    tokens = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = model.get_text_features(**tokens)
    return _normalise(features.numpy())


def image_text_similarity(image: bytes, texts: list[str]) -> list[float]:
    """Cosine similarity between one image and each text."""
    return (embed_texts(texts) @ embed_images([image])[0]).tolist()


def zero_shot(image: bytes, labels: list[str], template: str = "a photo of {}") -> list[tuple[str, float]]:
    """Softmax probability of each label for the image, using CLIP's own logit scale."""
    model, _ = _load()
    scale = float(model.logit_scale.exp())
    logits = scale * (embed_texts([template.format(label) for label in labels]) @ embed_images([image])[0])
    logits -= logits.max()
    probabilities = np.exp(logits) / np.exp(logits).sum()
    return list(zip(labels, probabilities.tolist()))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)
