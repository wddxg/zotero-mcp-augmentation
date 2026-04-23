from __future__ import annotations

from collections import defaultdict

from .embedding import EmbeddingRuntimeFactory
from .io_utils import JsonIO, JsonlIO
from .models import QueryConfig


class LocalChunkStoreQueryEngine:
    def __init__(self, *, config: QueryConfig) -> None:
        import numpy as np

        self.config = config
        self.manifest = JsonIO.read_json(self.config.store_dir / "manifest.json")
        self.rows = JsonlIO.read_rows(self.config.store_dir / "rows.jsonl")
        self.embeddings = np.load(self.config.store_dir / "embeddings.npy", mmap_mode="r")
        self._apply_document_filter()

    def _apply_document_filter(self) -> None:
        import numpy as np

        document_filter = self.config.document_id.strip()
        if not document_filter:
            return
        keep = [i for i, row in enumerate(self.rows) if str(row.get("document_id", "") or "") == document_filter]
        if not keep:
            raise SystemExit(f"No rows found for document_id={document_filter}")
        self.rows = [self.rows[i] for i in keep]
        self.embeddings = self.embeddings[np.array(keep, dtype=np.int64)]

    def run(self) -> dict[str, object]:
        query_vec, actual_device = self._encode_query()
        row_weights = np.array(
            [self._resolve_row_weight(row) for row in self.rows],
            dtype=np.float32,
        )
        raw_scores, scores = self._compute_scores(query_vec, row_weights)
        return {
            "query": self.config.query,
            "device": actual_device,
            "store_dir": str(self.config.store_dir),
            "row_count": len(self.rows),
            "row_weights_enabled": not self.config.disable_row_weights,
            "document_filter": self.config.document_id.strip(),
            "top_documents": self._build_top_documents(scores, raw_scores),
            "top_chunks": self._build_top_chunks(scores, raw_scores, row_weights),
        }

    def _encode_query(self) -> tuple[np.ndarray, str]:
        runtime = EmbeddingRuntimeFactory.create(
            model_name=str(self.manifest.get("model", "")),
            device=self.config.device,
            batch_size=1,
            normalize=bool(self.manifest.get("normalize")),
        )
        vector = runtime.encode([self.config.query])[0]
        return vector.astype(np.float32), runtime.actual_device

    def _resolve_row_weight(self, row: dict[str, object]) -> float:
        if self.config.disable_row_weights:
            return 1.0
        raw = row.get("retrieval_weight")
        try:
            return float(raw) if raw is not None else 1.0
        except (TypeError, ValueError):
            return 1.0

    def _compute_scores(self, query_vec: np.ndarray, row_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        total = int(self.embeddings.shape[0])
        raw_scores = np.empty(total, dtype=np.float32)
        scores = np.empty(total, dtype=np.float32)
        query32 = query_vec.astype(np.float32, copy=False)
        for start in range(0, total, self.config.score_chunk_size):
            end = min(total, start + self.config.score_chunk_size)
            chunk = np.asarray(self.embeddings[start:end], dtype=np.float32)
            raw = chunk @ query32
            raw_scores[start:end] = raw
            scores[start:end] = raw * row_weights[start:end]
        return raw_scores, scores

    def _build_top_chunks(
        self,
        scores: np.ndarray,
        raw_scores: np.ndarray,
        row_weights: np.ndarray,
    ) -> list[dict[str, object]]:
        indices = np.argsort(scores)[::-1][: self.config.top_k_chunks]
        payload: list[dict[str, object]] = []
        for rank, idx in enumerate(indices.tolist(), start=1):
            row = self.rows[int(idx)]
            payload.append(
                {
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "raw_score": float(raw_scores[int(idx)]),
                    "applied_weight": float(row_weights[int(idx)]),
                    "document_id": row.get("document_id", ""),
                    "title": row.get("title", ""),
                    "section": row.get("section", ""),
                    "location_label": self._format_location(row),
                    "chunk_id": row.get("chunk_id", 0),
                    "preview": str(row.get("text", "") or "")[:300],
                }
            )
        return payload

    def _build_top_documents(self, scores: np.ndarray, raw_scores: np.ndarray) -> list[dict[str, object]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(self.rows):
            key = str(row.get("document_id", "") or row.get("title", "") or f"doc-{idx}")
            grouped[key].append(idx)

        docs: list[dict[str, object]] = []
        for document_id, indices in grouped.items():
            ranked = sorted(indices, key=lambda i: float(scores[i]), reverse=True)
            top = ranked[:3]
            if not top:
                continue
            aggregate = float(scores[top[0]])
            if len(top) > 1:
                aggregate += float(scores[top[1]]) * 0.2
            if len(top) > 2:
                aggregate += float(scores[top[2]]) * 0.1
            best = self.rows[top[0]]
            docs.append(
                {
                    "document_id": document_id,
                    "title": best.get("title", ""),
                    "aggregate_score": aggregate,
                    "best_chunk_score": float(scores[top[0]]),
                    "best_chunk_raw_score": float(raw_scores[top[0]]),
                    "best_location_label": self._format_location(best),
                    "best_preview": str(best.get("text", "") or "")[:300],
                    "hit_count": len(indices),
                }
            )
        docs.sort(key=lambda item: float(item["aggregate_score"]), reverse=True)
        return [{"rank": rank, **item} for rank, item in enumerate(docs[: self.config.top_k_documents], start=1)]

    @staticmethod
    def _format_location(row: dict[str, object]) -> str:
        value = str(row.get("location_label", "") or "").strip()
        if value:
            return value
        part = str(row.get("part_label", "") or row.get("section", "") or "").strip()
        heading = str(row.get("heading_path", "") or "").strip()
        return f"{part} > {heading}" if part and heading else part or heading
