from __future__ import annotations

from datetime import datetime

from .embedding import EmbeddingRuntimeFactory
from .io_utils import JsonIO, JsonlIO
from .models import StoreBuildConfig


class ChunkTextFormatter:
    LOW_VALUE_PATTERNS = (
        "acknowledg",
        "funding",
        "conflict of interest",
        "declaration of competing interest",
        "availability of data",
        "致谢",
        "基金项目",
    )

    @classmethod
    def is_low_value_text(cls, text: str) -> bool:
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return True
        return any(normalized.startswith(pattern) for pattern in cls.LOW_VALUE_PATTERNS)

    @staticmethod
    def normalize_token(token: str) -> str:
        value = " ".join((token or "").strip().split())
        return value.strip(" -–—:：;；,，、()（）[]【】<>《》\"'")

    @classmethod
    def split_tokens(cls, value: object) -> list[str]:
        import re

        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = [part for part in re.split(r"[;；,，、/\n|]+", str(value)) if part.strip()]
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            token = cls.normalize_token(str(item))
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(token)
        return result

    @classmethod
    def build_embedding_text(cls, row: dict[str, object]) -> str:
        parts = [f"Title: {str(row.get('title', '') or '').strip()}"]
        document_id = str(row.get("document_id", "") or "").strip()
        if document_id:
            parts.append(f"Document ID: {document_id}")
        section = str(row.get("section", "") or "").strip()
        if section:
            parts.append(f"Section: {section}")
        location = str(row.get("location_label", "") or "").strip()
        if location:
            parts.append(f"Location: {location}")
        keywords = cls.split_tokens(row.get("keywords"))
        if keywords:
            parts.append(f"Keywords: {'; '.join(keywords)}")
        tags = cls.split_tokens(row.get("tags"))
        if tags:
            parts.append(f"Tags: {'; '.join(tags)}")
        parts.append("Chunk:")
        parts.append(str(row.get("text", "") or "").strip())
        return "\n".join(part for part in parts if part)


class LocalChunkStoreBuilder:
    def __init__(self, *, config: StoreBuildConfig, formatter: ChunkTextFormatter | None = None) -> None:
        self.config = config
        self.formatter = formatter or ChunkTextFormatter()

    def run(self) -> tuple[int, int, str]:
        rows = JsonlIO.read_rows(self.config.chunks_jsonl)
        selected_rows = self._select_rows(rows)
        if not selected_rows:
            raise SystemExit("No eligible chunk rows matched the current filters.")

        total_rows, dimension, actual_device = self._build_embeddings(selected_rows)
        JsonlIO.write_rows(self.config.out_dir / "rows.jsonl", selected_rows)
        self._write_manifest(total_rows, dimension, actual_device)
        self._write_completed_progress(total_rows, dimension, actual_device)
        return total_rows, dimension, actual_device

    def _select_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        for row in rows:
            text = str(row.get("text", "") or "")
            if len(text.strip()) < self.config.min_char_count:
                continue
            if self.formatter.is_low_value_text(text):
                continue
            enriched = dict(row)
            enriched["char_count"] = int(row.get("char_count", len(text)) or len(text))
            enriched["embedding_text"] = self.formatter.build_embedding_text(enriched)
            selected.append(enriched)
        return selected

    def _build_embeddings(self, rows: list[dict[str, object]]) -> tuple[int, int, str]:
        from numpy.lib.format import open_memmap
        import numpy as np

        progress_path = self.config.out_dir / "build_progress.json"
        embeddings_path = self.config.out_dir / "embeddings.npy"
        runtime = EmbeddingRuntimeFactory.create(
            model_name=self.config.model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
            normalize=self.config.normalize,
        )
        total_rows = len(rows)
        dimension = int(runtime.dimension)
        np_dtype = np.float16 if self.config.dtype == "float16" else np.float32
        completed_rows = 0

        self._prepare_output_dir()

        if self.config.allow_resume and progress_path.exists() and embeddings_path.exists():
            progress = JsonIO.read_json(progress_path)
            self._validate_resume_progress(progress, total_rows, dimension)
            completed_rows = min(total_rows, int(progress.get("completed_rows", 0) or 0))
            memmap = np.load(embeddings_path, mmap_mode="r+")
        else:
            memmap = open_memmap(embeddings_path, mode="w+", dtype=np_dtype, shape=(total_rows, dimension))
            JsonIO.write_json(
                progress_path,
                {
                    "status": "running",
                    "model": self.config.model_name,
                    "device": runtime.actual_device,
                    "dimension": dimension,
                    "row_count": total_rows,
                    "completed_rows": 0,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

        for start in range(completed_rows, total_rows, self.config.write_chunk_size):
            end = min(total_rows, start + self.config.write_chunk_size)
            vectors = runtime.encode([str(row.get("embedding_text", "") or "") for row in rows[start:end]])
            memmap[start:end] = vectors.astype(np_dtype, copy=False)
            memmap.flush()
            JsonIO.write_json(
                progress_path,
                {
                    "status": "running",
                    "model": self.config.model_name,
                    "device": runtime.actual_device,
                    "dimension": dimension,
                    "row_count": total_rows,
                    "completed_rows": end,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(f"rows {start + 1}-{end}/{total_rows}")

        return total_rows, dimension, runtime.actual_device

    def _prepare_output_dir(self) -> None:
        if self.config.out_dir.exists() and not self.config.allow_resume:
            existing = sorted(path.name for path in self.config.out_dir.iterdir())
            if existing:
                raise SystemExit(
                    f"Output directory is not empty: {self.config.out_dir}. Use --resume or choose a new --out-dir."
                )
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

    def _validate_resume_progress(self, progress: dict[str, object], total_rows: int, dimension: int) -> None:
        if int(progress.get("row_count", -1) or -1) != total_rows:
            raise SystemExit("Resume row_count mismatch.")
        if str(progress.get("model", "") or "") != self.config.model_name:
            raise SystemExit("Resume model mismatch.")
        if int(progress.get("dimension", -1) or -1) != dimension:
            raise SystemExit("Resume dimension mismatch.")

    def _write_manifest(self, total_rows: int, dimension: int, actual_device: str) -> None:
        JsonIO.write_json(
            self.config.out_dir / "manifest.json",
            {
                "store_kind": "local_chunk_store",
                "model": self.config.model_name,
                "device": actual_device,
                "row_count": total_rows,
                "dimension": dimension,
                "normalize": self.config.normalize,
                "dtype": self.config.dtype,
                "text_field": "embedding_text",
                "source_chunks": str(self.config.chunks_jsonl),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _write_completed_progress(self, total_rows: int, dimension: int, actual_device: str) -> None:
        JsonIO.write_json(
            self.config.out_dir / "build_progress.json",
            {
                "status": "completed",
                "model": self.config.model_name,
                "device": actual_device,
                "row_count": total_rows,
                "completed_rows": total_rows,
                "dimension": dimension,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
