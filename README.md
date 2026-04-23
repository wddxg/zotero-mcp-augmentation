# Zotero MCP augmentation

一个给 `Zotero MCP` 使用的本地补强工具。它主要处理全文清洗、分段、嵌入和段落级检索。  
A local add-on for `Zotero MCP`. It focuses on full-text cleaning, chunking, local embeddings, and passage retrieval.

## Workflow

`item -> cleaned text -> chunks -> local embeddings -> top chunks`

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```powershell
python scripts/clean_and_chunk_markdown.py `
  --input examples/sample-source.md `
  --cleaned-out output/cleaned.md `
  --chunks-out output/chunks.jsonl `
  --title "Example Document" `
  --document-id example-doc
```

```powershell
python scripts/build_local_chunk_store.py `
  --chunks-jsonl output/chunks.jsonl `
  --out-dir output/store `
  --model BAAI/bge-m3 `
  --device cuda `
  --normalize
```

```powershell
python scripts/query_local_chunk_store.py `
  --store-dir output/store `
  --query "find paragraphs about tunnel temperature field evolution"
```

## Notes

  Clean the text before embedding.


