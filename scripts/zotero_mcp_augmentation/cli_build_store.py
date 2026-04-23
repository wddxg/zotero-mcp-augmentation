from __future__ import annotations

import argparse
import os
from pathlib import Path

from .models import StoreBuildConfig
from .store_builder import LocalChunkStoreBuilder


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_WRITE_CHUNK_SIZE = 256
DEFAULT_DTYPE = "float16"
DEFAULT_MIN_CHAR_COUNT = 80


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local chunk embedding store from chunks.jsonl.")
    parser.add_argument("--chunks-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--write-chunk-size", type=int, default=DEFAULT_WRITE_CHUNK_SIZE)
    parser.add_argument("--dtype", choices=("float16", "float32"), default=DEFAULT_DTYPE)
    parser.add_argument("--min-char-count", type=int, default=DEFAULT_MIN_CHAR_COUNT)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    builder = LocalChunkStoreBuilder(
        config=StoreBuildConfig(
            chunks_jsonl=args.chunks_jsonl.resolve(),
            out_dir=args.out_dir.resolve(),
            model_name=args.model,
            device=args.device,
            batch_size=max(1, int(args.batch_size)),
            write_chunk_size=max(1, int(args.write_chunk_size)),
            dtype=args.dtype,
            min_char_count=max(1, int(args.min_char_count)),
            normalize=bool(args.normalize),
            allow_resume=bool(args.resume),
        )
    )
    total_rows, dimension, actual_device = builder.run()

    print(f"store={args.out_dir.resolve()}")
    print(f"rows={total_rows}")
    print(f"dimension={dimension}")
    print(f"device={actual_device}")
