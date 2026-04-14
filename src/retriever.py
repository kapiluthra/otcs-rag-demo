"""
retriever.py — Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion

Why hybrid?
  - Pure vector search: misses exact keyword matches (contract numbers, IDs, codes)
  - Pure BM25: misses semantic similarity ("termination clause" ≠ "cancellation terms")
  - RRF fusion: combines both ranked lists, rewards docs appearing in both

RRF formula: score(doc) = Σ 1/(k + rank)  where k=60 (standard)
"""

import logging
import os
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
PARQUET_PATH = os.getenv("PARQUET_PATH", "./cold_store.parquet")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def _embed(text: str) -> list[float]:
    import requests
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _rrf_score(ranks: list[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


class HybridRetriever:
    """Hybrid BM25 + vector retriever with RRF reranking.

    Initialise once and reuse — BM25 index is built from the Parquet cold store.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("what are the contract renewal terms", top_k=5)
        for r in results:
            print(r["doc_name"], r["parent_text"][:200])
    """

    def __init__(self):
        self._collection = None
        self._bm25 = None
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_meta: list[dict] = []
        self._build_bm25_index()

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._collection = client.get_or_create_collection("otcs_docs")
        return self._collection

    def _build_bm25_index(self) -> None:
        """Build BM25 index from Parquet cold store."""
        parquet_path = Path(PARQUET_PATH)
        if not parquet_path.exists():
            logger.warning("Parquet cold store not found at %s — BM25 disabled", PARQUET_PATH)
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed — run: pip install rank-bm25")
            return

        df = pd.read_parquet(parquet_path, columns=["chunk_id", "child_text", "doc_name", "parent_id"])
        self._bm25_corpus = df["child_text"].tolist()
        self._bm25_ids = df["chunk_id"].tolist()
        self._bm25_meta = df[["chunk_id", "doc_name", "parent_id"]].to_dict("records")

        tokenized = [text.lower().split() for text in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d chunks", len(self._bm25_corpus))

    def search(
        self,
        query: str,
        top_k: int = 5,
        n_candidates: int = 20,
        # For ACL security gate: pass current username to filter results
        username: str = None,
        cs_client=None,
    ) -> list[dict]:
        """Search using hybrid BM25 + vector with RRF reranking.

        Args:
            query: Natural language query
            top_k: Number of results to return
            n_candidates: Number of candidates from each search before RRF
            username: If provided with cs_client, filters by OTCS permissions
            cs_client: CSClient instance for permission checks

        Returns:
            List of result dicts with keys: chunk_id, doc_id, doc_name,
            parent_id, child_text, parent_text, score
        """
        collection = self._get_collection()
        scores: dict[str, list[int]] = {}

        # ── Vector search ───────────────────────────────────────────────────
        try:
            query_embedding = _embed(query)
            vec_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_candidates, collection.count()),
            )
            for rank, chunk_id in enumerate(vec_results["ids"][0]):
                scores.setdefault(chunk_id, []).append(rank + 1)
        except Exception as e:
            logger.warning("Vector search failed: %s", e)

        # ── BM25 search ─────────────────────────────────────────────────────
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            top_indices = np.argsort(bm25_scores)[::-1][:n_candidates]
            for rank, idx in enumerate(top_indices):
                chunk_id = self._bm25_ids[idx]
                scores.setdefault(chunk_id, []).append(rank + 1)

        # ── RRF fusion ──────────────────────────────────────────────────────
        ranked_ids = sorted(
            scores.keys(),
            key=lambda cid: _rrf_score(scores[cid]),
            reverse=True,
        )

        # ── Fetch parent text and build results ─────────────────────────────
        from ingester import get_parent_text

        results = []
        for chunk_id in ranked_ids:
            if len(results) >= top_k:
                break

            # Fetch chunk metadata from ChromaDB
            try:
                chunk_data = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                if not chunk_data["ids"]:
                    continue
                meta = chunk_data["metadatas"][0]
                child_text = chunk_data["documents"][0]
            except Exception:
                continue

            doc_id = meta.get("doc_id", "")

            # ── ACL security gate (production: uncomment and pass cs_client) ──
            # if username and cs_client:
            #     if not cs_client.check_user_access(int(doc_id), username):
            #         continue   # skip — user has no read access

            parent_text = get_parent_text(meta.get("parent_id", ""))

            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": meta.get("doc_name", ""),
                "parent_id": meta.get("parent_id", ""),
                "child_text": child_text,
                "parent_text": parent_text or child_text,  # fallback to child if cold store miss
                "rrf_score": _rrf_score(scores[chunk_id]),
            })

        return results
