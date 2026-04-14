# Architecture Notes

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│  OpenText CS ──► cs_client.py ──► extractor.py ──► chunker.py  │
│                                                         │       │
│                                              ┌──────────┘       │
│                                              ▼                  │
│                                        ingester.py              │
│                                       /           \             │
│                                  ChromaDB       Parquet         │
│                                 (vectors)     (cold store)      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                          │
│                                                                 │
│  Query ──► embed ──► ChromaDB (vector search)  ─┐              │
│         └──────────► BM25 (keyword search)      ─┤              │
│                                                  ▼              │
│                                           RRF reranking         │
│                                                  │              │
│                                    ┌─────────────┘              │
│                                    ▼                            │
│                         fetch parent text (Parquet)             │
│                                    │                            │
│                                    ▼                            │
│                          ACL permission check                   │
│                                    │                            │
│                                    ▼                            │
│                            LLM (Ollama/llama3) ──► Answer       │
└─────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why two storage layers?

**ChromaDB (hot store):**
- Queried on every retrieval request
- Stores embeddings + small child chunks + metadata
- Fast cosine similarity search via HNSW

**Parquet (cold store):**
- Append-only, stores full text of both child and parent chunks
- Used to: (1) return parent text to LLM without re-embedding, (2) re-embed
  with a different model without re-crawling OTCS
- In practice, re-embedding the corpus is something you'll do as better models
  become available. Without the cold store this means re-downloading everything
  from OTCS, which is slow and adds server load.

### Why parent-child chunking?

The fundamental tension in RAG chunking:
- Small chunks → precise vector matching
- Large chunks → enough context for LLM reasoning

Parent-child gives you both:
- Child chunks (~400 tokens) are embedded and matched
- Parent chunks (~1500 tokens) are fetched from Parquet and sent to the LLM

### Why hybrid retrieval?

Pure vector search fails on exact keywords: a user searching for "contract
number PO-2024-0042" gets semantic matches to similar contracts but misses the
exact document. BM25 handles this but misses semantic similarity.

RRF combines both ranked lists by rewarding documents that appear highly ranked
in *both* searches: `score = Σ 1/(60 + rank)`. In testing against OTCS corpora,
hybrid + RRF improved recall on keyword-heavy queries by ~40%.

### Why k=60 in RRF?

k=60 is the standard constant from the original RRF paper (Cormack et al. 2009).
It dampens the effect of very high rankings while still rewarding consistently
good ranks across both systems. Values between 40-80 all perform similarly;
60 is a safe default.

### ACL security gates

In a production enterprise deployment, RAG without ACL enforcement is a data
leak waiting to happen. The pattern:
1. Store OTCS node ID with every chunk in ChromaDB metadata
2. At retrieval time, call `GET /api/v2/nodes/{id}/permissions/effective`
   for each candidate chunk before including it in the context
3. Cache permission results per (user, node_id) with a 5–15 min TTL

This adds latency (one API call per unique node in results), but correctness
matters more than latency in enterprise settings.

### Ticket renewal

OTCSTicket expires after a configurable timeout (typically 30 min). The
CSClient uses a property that transparently renews the ticket when it approaches
expiry (`ticket_ttl = 1700`s, renewing ~100s before the 30-min boundary).
Callers never need to handle auth — every `client.get()` call is transparently
authenticated.

## Incremental Sync

To keep the knowledge base current without re-processing everything:

1. Track `last_sync_time` in a JSON state file
2. On each sync run: walk OTCS nodes, check `OTModifyDate > last_sync_time`
3. For modified documents: delete existing chunks from ChromaDB and Parquet,
   re-extract, re-chunk, re-ingest
4. Update `last_sync_time` to now

This means each sync run only processes changed documents, keeping compute
costs manageable as the repository grows.
