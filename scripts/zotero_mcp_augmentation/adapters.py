from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ChunkSourceAdapter(ABC):
    @abstractmethod
    def resolve_document_id(self, *, source_path: Path, requested_document_id: str | None) -> str:
        raise NotImplementedError

    @abstractmethod
    def resolve_title(self, *, source_path: Path, detected_title: str, requested_title: str | None) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_chunk_metadata(self, *, source_path: Path) -> dict[str, object]:
        raise NotImplementedError


class LocalFileAdapter(ChunkSourceAdapter):
    def resolve_document_id(self, *, source_path: Path, requested_document_id: str | None) -> str:
        return (requested_document_id or "").strip() or source_path.stem

    def resolve_title(self, *, source_path: Path, detected_title: str, requested_title: str | None) -> str:
        return (requested_title or "").strip() or detected_title or source_path.stem

    def build_chunk_metadata(self, *, source_path: Path) -> dict[str, object]:
        return {"source_path": str(source_path)}
