from __future__ import annotations

import re
from pathlib import Path

from .adapters import ChunkSourceAdapter, LocalFileAdapter
from .cleaning import MarkdownCleaner
from .models import ChunkDocument, ChunkingConfig, ChunkRecord, HeadingState


class HeadingAwareChunkBuilder:
    SECTION_DISPLAY_LABELS = {
        "abstract_zh": "中文摘要",
        "abstract_en": "英文摘要",
        "keywords_zh": "中文关键词",
        "keywords_en": "英文关键词",
        "introduction": "引言/绪论",
        "conclusion": "结论",
        "body": "正文",
    }

    def __init__(self, *, cleaner: MarkdownCleaner, config: ChunkingConfig) -> None:
        self.cleaner = cleaner
        self.config = config

    @staticmethod
    def split_paragraphs(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]

    def make_location_label(self, section: str, heading_path: str) -> str:
        part = self.SECTION_DISPLAY_LABELS.get(section, section or "正文")
        return f"{part} > {heading_path}" if heading_path.strip() else part

    def chunk_paragraphs(self, paragraphs: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush(keep_overlap: bool) -> None:
            nonlocal current, current_len
            if not current:
                return
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if keep_overlap and self.config.overlap_chars > 0:
                overlap: list[str] = []
                overlap_len = 0
                for paragraph in reversed(current):
                    overlap.insert(0, paragraph)
                    overlap_len += len(paragraph)
                    if overlap_len >= self.config.overlap_chars:
                        break
                current = overlap
                current_len = sum(len(item) for item in current)
            else:
                current = []
                current_len = 0

        for paragraph in paragraphs:
            if len(paragraph) > self.config.max_chars:
                parts = [part.strip() for part in re.split(r"(?<=[。！？!?；;:])", paragraph) if part.strip()]
                for part in parts:
                    if current and current_len + len(part) > self.config.max_chars:
                        flush(True)
                    current.append(part)
                    current_len += len(part)
                    if current_len >= self.config.target_chars:
                        flush(True)
                continue
            if current and current_len + len(paragraph) > self.config.max_chars:
                flush(True)
            current.append(paragraph)
            current_len += len(paragraph)
            if current_len >= self.config.target_chars:
                flush(True)

        if current:
            flush(False)
        return [chunk for chunk in chunks if chunk.strip()]

    def build_records(
        self,
        cleaned_text: str,
        *,
        document_id: str,
        title: str,
        base_metadata: dict[str, object] | None = None,
    ) -> list[ChunkRecord]:
        paragraphs = self.split_paragraphs(cleaned_text)
        records: list[ChunkRecord] = []
        state = HeadingState()
        current_paragraphs: list[str] = []
        chunk_id = 0
        base_metadata = dict(base_metadata or {})

        def flush_block() -> None:
            nonlocal chunk_id, current_paragraphs
            if not current_paragraphs:
                return
            for text_chunk in self.chunk_paragraphs(current_paragraphs):
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        title=title,
                        section=state.section,
                        part_label=self.SECTION_DISPLAY_LABELS.get(state.section, state.section or "正文"),
                        heading_path=state.heading_path,
                        heading_title=state.heading_title,
                        heading_level=state.heading_level,
                        location_label=self.make_location_label(state.section, state.heading_path),
                        char_count=len(text_chunk),
                        text=text_chunk,
                        keywords=list(base_metadata.get("keywords", []) or []),
                        tags=list(base_metadata.get("tags", []) or []),
                        source_path=str(base_metadata.get("source_path", "") or ""),
                        retrieval_weight=float(base_metadata.get("retrieval_weight", 1.0) or 1.0),
                    )
                )
                chunk_id += 1
            current_paragraphs = []

        for paragraph in paragraphs:
            first_line = paragraph.splitlines()[0].strip()
            section_heading = self.cleaner.looks_like_section_heading(first_line)
            if section_heading:
                flush_block()
                state = HeadingState(section=section_heading)
                remainder = "\n".join(paragraph.splitlines()[1:]).strip()
                if remainder:
                    current_paragraphs.append(remainder)
                continue

            heading = self.cleaner.parse_body_heading(paragraph)
            if heading and state.section not in {"abstract_zh", "abstract_en", "keywords_zh", "keywords_en"}:
                flush_block()
                self._apply_heading(state, heading.heading_level, heading.heading_display, heading.heading_title)
                if heading.remainder:
                    current_paragraphs.append(heading.remainder)
                continue

            current_paragraphs.append(paragraph)

        flush_block()
        return records

    @staticmethod
    def _apply_heading(state: HeadingState, level: int, display: str, title: str) -> None:
        state.heading_stack = [entry for entry in state.heading_stack if int(entry["heading_level"]) < level]
        state.heading_stack.append({"heading_level": level, "heading_display": display})
        state.heading_level = level
        state.heading_title = title
        state.heading_path = " > ".join(str(entry["heading_display"]) for entry in state.heading_stack)


class ChunkingWorkflow:
    def __init__(
        self,
        *,
        cleaner: MarkdownCleaner,
        builder: HeadingAwareChunkBuilder,
        adapter: ChunkSourceAdapter | None = None,
    ) -> None:
        self.cleaner = cleaner
        self.builder = builder
        self.adapter = adapter or LocalFileAdapter()

    def run(
        self,
        *,
        source_path: Path,
        raw_text: str,
        requested_document_id: str | None = None,
        requested_title: str | None = None,
    ) -> ChunkDocument:
        cleaned_text, detected_title = self.cleaner.clean(raw_text)
        document_id = self.adapter.resolve_document_id(source_path=source_path, requested_document_id=requested_document_id)
        title = self.adapter.resolve_title(
            source_path=source_path,
            detected_title=detected_title,
            requested_title=requested_title,
        )
        metadata = self.adapter.build_chunk_metadata(source_path=source_path)
        chunks = self.builder.build_records(cleaned_text, document_id=document_id, title=title, base_metadata=metadata)
        return ChunkDocument(
            source_path=source_path,
            document_id=document_id,
            title=title,
            cleaned_text=cleaned_text,
            chunks=chunks,
        )
