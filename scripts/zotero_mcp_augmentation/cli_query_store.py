from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import QueryConfig
from .query_engine import LocalChunkStoreQueryEngine


DEFAULT_SCORE_CHUNK_SIZE = 4096


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Query a local chunk embedding store.")
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k-documents", type=int, default=8)
    parser.add_argument("--top-k-chunks", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-chunk-size", type=int, default=DEFAULT_SCORE_CHUNK_SIZE)
    parser.add_argument("--document-id", default="")
    parser.add_argument("--disable-row-weights", action="store_true")
    args = parser.parse_args()

    engine = LocalChunkStoreQueryEngine(
        config=QueryConfig(
            store_dir=args.store_dir.resolve(),
            query=args.query,
            top_k_documents=max(1, int(args.top_k_documents)),
            top_k_chunks=max(1, int(args.top_k_chunks)),
            device=args.device,
            score_chunk_size=max(1, int(args.score_chunk_size)),
            document_id=args.document_id,
            disable_row_weights=bool(args.disable_row_weights),
        )
    )
    print(json.dumps(engine.run(), ensure_ascii=False, indent=2))
