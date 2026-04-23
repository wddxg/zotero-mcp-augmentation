from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import ChunkingWorkflow, HeadingAwareChunkBuilder
from .cleaning import MarkdownCleaner
from .models import ChunkingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Markdown and build heading-aware retrieval chunks.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cleaned-out", type=Path, required=True)
    parser.add_argument("--chunks-out", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--document-id", default="")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--target-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=150)
    args = parser.parse_args()

    source_path = args.input.resolve()
    raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
    cleaner = MarkdownCleaner()
    workflow = ChunkingWorkflow(
        cleaner=cleaner,
        builder=HeadingAwareChunkBuilder(
            cleaner=cleaner,
            config=ChunkingConfig(
                max_chars=max(200, int(args.max_chars)),
                target_chars=max(100, int(args.target_chars)),
                overlap_chars=max(0, int(args.overlap_chars)),
            ),
        ),
    )
    document = workflow.run(
        source_path=source_path,
        raw_text=raw_text,
        requested_document_id=args.document_id.strip() or None,
        requested_title=args.title.strip() or None,
    )

    args.cleaned_out.parent.mkdir(parents=True, exist_ok=True)
    args.cleaned_out.write_text(document.cleaned_text, encoding="utf-8")
    args.chunks_out.parent.mkdir(parents=True, exist_ok=True)
    with args.chunks_out.open("w", encoding="utf-8") as f:
        for row in document.chunks:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"cleaned={args.cleaned_out}")
    print(f"chunks={args.chunks_out}")
    print(f"chunk_count={len(document.chunks)}")
    print(f"document_id={document.document_id}")
    print(f"title={document.title}")
