---
name: zotero-mcp-augmentation
description: Use this skill when the user wants to add local full-text cleaning, chunking, embedding, or passage retrieval to Zotero MCP. It is suitable for cleaned fulltext sidecars, chunk-level semantic search, and local RAG-style workflows built on local models instead of remote APIs.
---

# Zotero MCP augmentation

这个 skill 用来给 `Zotero MCP` 补一层本地全文处理和段落检索能力。

适用情况：

- 需要把 OCR 或 Markdown 文本清洗后再建库
- 需要把长文按标题结构切成可检索分段
- 需要用本地模型生成嵌入
- 需要按语义检索并返回段落级结果

## Commands

### 1. Clean and chunk

```powershell
python scripts/clean_and_chunk_markdown.py `
  --input source.md `
  --cleaned-out output/cleaned.md `
  --chunks-out output/chunks.jsonl `
  --title "My Document" `
  --document-id doc-001
```

### 2. Build the local store

```powershell
python scripts/build_local_chunk_store.py `
  --chunks-jsonl output/chunks.jsonl `
  --out-dir output/store `
  --model BAAI/bge-m3 `
  --device cuda `
  --normalize
```

### 3. Query

```powershell
python scripts/query_local_chunk_store.py `
  --store-dir output/store `
  --query "find paragraphs about coupled moisture and deformation effects"
```

## Expected Outputs

- `cleaned.md`
- `chunks.jsonl`
- `rows.jsonl`
- `embeddings.npy`
- `manifest.json`
- query result JSON with matched documents and chunks

## Output Style

执行这个 skill 时，返回内容尽量简短，优先给这些信息：

- 输出文件路径
- 实际执行的命令
- 清洗时去掉了什么类型的噪声
- 如果跑了查询，给 1 到 2 条命中示例

## Files

- `scripts/clean_and_chunk_markdown.py`: 清洗并分段
- `scripts/build_local_chunk_store.py`: 构建本地向量库
- `scripts/query_local_chunk_store.py`: 查询本地向量库
- `scripts/zotero_mcp_augmentation/`: 包内实现
- `references/data-shape.md`: 数据结构说明
