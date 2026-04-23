# Data Shape

## `chunks.jsonl`

Each line is one cleaned chunk candidate.

```json
{
  "chunk_id": 0,
  "document_id": "doc-001",
  "title": "Example Document",
  "section": "body",
  "part_label": "正文",
  "heading_path": "1 Introduction > 1.1 Background",
  "heading_title": "Background",
  "heading_level": 2,
  "location_label": "正文 > 1 Introduction > 1.1 Background",
  "char_count": 865,
  "text": "..."
}
```

Useful optional fields:

- `tags`
- `keywords`
- `source_path`
- `retrieval_weight`

## `rows.jsonl`

This is the enriched retrieval file used together with `embeddings.npy`.

Typical extra fields:

- `embedding_text`
- normalized `tags`
- normalized `keywords`

## `manifest.json`

Describes the built store:

- model
- device
- embedding dimension
- row count
- normalization flag
- source file path

## Query Output

The query script returns:

- `top_documents`: aggregated document candidates
- `top_chunks`: directly reusable chunk hits
