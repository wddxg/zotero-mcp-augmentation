from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ChunkingConfig:
    max_chars: int = 1800
    target_chars: int = 1200
    overlap_chars: int = 150


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: int
    document_id: str
    title: str
    section: str
    part_label: str
    heading_path: str
    heading_title: str
    heading_level: int
    location_label: str
    char_count: int
    text: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_path: str = ""
    retrieval_weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class HeadingInfo:
    heading_level: int
    heading_title: str
    heading_display: str
    remainder: str


@dataclass(slots=True)
class HeadingState:
    section: str = "body"
    heading_path: str = ""
    heading_title: str = ""
    heading_level: int = 0
    heading_stack: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class ChunkDocument:
    source_path: Path
    document_id: str
    title: str
    cleaned_text: str
    chunks: list[ChunkRecord]


@dataclass(slots=True)
class StoreBuildConfig:
    chunks_jsonl: Path
    out_dir: Path
    model_name: str
    device: str
    batch_size: int
    write_chunk_size: int
    dtype: str
    min_char_count: int
    normalize: bool
    allow_resume: bool


@dataclass(slots=True)
class QueryConfig:
    store_dir: Path
    query: str
    top_k_documents: int
    top_k_chunks: int
    device: str
    score_chunk_size: int
    document_id: str = ""
    disable_row_weights: bool = False
