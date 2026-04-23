# Example Tunnel Paper

## Abstract

This example document is only a neutral fixture for testing local cleaning, chunking, and retrieval. It does not depend on Zotero state or any external API.

## Keywords

tunnel lining; moisture diffusion; deformation; local retrieval

## 1 Introduction

Segmented semantic retrieval works better when parsed text is cleaned before embedding. Front matter noise, repeated headers, and table-of-contents fragments often hurt recall quality.

## 1.1 Motivation

A chunk-oriented retrieval layer is useful when document-level search finds the right paper but cannot quickly locate the exact paragraph worth citing or reviewing.

## 2 Method

The workflow first cleans parsed Markdown, then splits it into heading-aware chunks, then builds local embeddings, and finally returns both top documents and top chunks for a query.

## 3 Conclusion

The augmentation design keeps Zotero MCP focused on bibliographic and item workflows while the chunk store handles local paragraph-level semantic recall.
