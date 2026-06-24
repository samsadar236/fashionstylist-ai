"""
FashionCLIP wrapper. Loaded via HuggingFace transformers.

Embeddings are L2-normalised, so cosine similarity == dot product.

Robust to transformers-version differences: get_image_features /
get_text_features may return a plain tensor OR a wrapped output object. We pull
the already-projected embedding straight out (image_embeds / text_embeds, or
pooler_output as a fallback) and never re-project it.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class FashionEncoder:
    def __init__(self, model_id: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @staticmethod
    def _pick(out, names: tuple[str, ...]) -> torch.Tensor:
        """Return the embedding tensor from a tensor or a ModelOutput wrapper."""
        if torch.is_tensor(out):
            return out
        for n in names:
            v = getattr(out, n, None)
            if torch.is_tensor(v):
                return v
        raise TypeError(f"Cannot extract embedding from {type(out).__name__}")

    @staticmethod
    def _normalize(feats: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        feats = self._pick(out, ("image_embeds", "pooler_output"))
        return self._normalize(feats).cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        feats = self._pick(out, ("text_embeds", "pooler_output"))
        return self._normalize(feats).cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]