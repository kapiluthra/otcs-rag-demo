"""
ingester.py — Ingest document chunks into ChromaDB + Parquet cold store

Two storage layers:
  - ChromaDB: vector hot store, queried at retrieval time
  - Parquet:  cold store of raw text — lets you re-embed with a new model
              without re-crawling OTCS (very useful in practice)
"""

import logging
import os
from pathlib import Path

import chromadb
import pandas as pd

from chunker import Chunk

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
PARQUET_PATH = os.getenv("PARQUET_PATH", "./cold_store.parquet")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="otcs_docs",
        metadata={"hnsw:space": "cosine"},
    )


def _embed(text: str) -> list[float]:
    """Embed text using Ollama nomic-embed-text (768 dimensions)."""
    import requests
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def ingest_chunks(chunks: list[Chunk], batch_size: int = 50) -> None:
    """Ingest a list of Chunk objects into ChromaDB and append to Parquet cold store.

    Args:
        chunks: List of Chunk objects from chunker.chunk_document()
        batch_size: Number of chunks to embed and upsert per batch
    """
    if not chunks:
        return

    collection = _get_collection()
    cold_rows = []

    # Process in batches to avoid OOM on large documents
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids, docs, metas, embeds = [], [], [], []

        for chunk in batch:
            try:
                embedding = _embed(chunk.child_text)
            except Exception as e:
                logger.warning("Embedding failed for chunk %s: %s", chunk.chunk_id, e)
                continue

            ids.append(chunk.chunk_id)
            docs.append(chunk.child_text)
            metas.append({
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "parent_id": chunk.parent_id,
                "modify_date": chunk.modify_date,
                "mime_type": chunk.mime_type or "",
            })
            embeds.append(embedding)

            # Collect for cold store
            cold_rows.append({
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "child_text": chunk.child_text,
                "parent_text": chunk.parent_text,
                "modify_date": chunk.modify_date,
            })

        if ids:
            collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeds,
            )
            logger.info("Upserted %d chunks (batch %d/%d)", len(ids), i // batch_size + 1, -(-len(chunks) // batch_size))

    # Append to Parquet cold store
    if cold_rows:
        _append_parquet(cold_rows)


def _append_parquet(rows: list[dict]) -> None:
    """Append rows to the Parquet cold store (create if not exists)."""
    new_df = pd.DataFrame(rows)
    parquet_path = Path(PARQUET_PATH)

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        # Deduplicate by chunk_id (upsert semantics)
        combined = pd.concat([existing, new_df]).drop_duplicates(
            subset=["chunk_id"], keep="last"
        )
        combined.to_parquet(parquet_path, index=False)
    else:
        new_df.to_parquet(parquet_path, index=False)


def get_parent_text(parent_id: str) -> str:
    """Retrieve parent text from Parquet cold store for LLM context expansion."""
    parquet_path = Path(PARQUET_PATH)
    if not parquet_path.exists():
        return ""
    df = pd.read_parquet(parquet_path, columns=["parent_id", "parent_text"])
    matches = df[df["parent_id"] == parent_id]["parent_text"]
    return matches.iloc[0] if len(matches) > 0 else ""


def delete_doc_chunks(doc_id: str) -> None:
    """Remove all chunks for a document (used during incremental sync for modified docs)."""
    collection = _get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info("Deleted %d chunks for doc_id=%s", len(results["ids"]), doc_id)
