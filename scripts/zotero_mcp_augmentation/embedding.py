from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalEmbeddingRuntime:
    model_name: str
    actual_device: str
    dimension: int
    batch_size: int
    normalize: bool

    def encode(self, texts: list[str]):
        raise NotImplementedError


class SentenceTransformerRuntime(LocalEmbeddingRuntime):
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
        normalize: bool,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        actual_device = str(getattr(self.model, "device", device))
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = int(self.model.get_embedding_dimension())
        else:
            dimension = int(self.model.get_sentence_embedding_dimension())
        super().__init__(
            model_name=model_name,
            actual_device=actual_device,
            dimension=dimension,
            batch_size=max(1, int(batch_size)),
            normalize=normalize,
        )

    def encode(self, texts: list[str]):
        import numpy as np

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingRuntimeFactory:
    @staticmethod
    def create(
        *,
        model_name: str,
        device: str,
        batch_size: int,
        normalize: bool,
    ) -> SentenceTransformerRuntime:
        return SentenceTransformerRuntime(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
        )
