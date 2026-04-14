# otcs-rag-demo

A sanitised demo scaffold for building a **RAG (Retrieval-Augmented Generation) pipeline on OpenText Content Server**.

This repo contains the patterns and skeleton code from a production system built against an OTCS repository of 500+ documents. The full production version is private (enterprise client); this scaffold strips client-specific details and demonstrates the core architecture.

---

## What's in here

```
otcs-rag-demo/
├── src/
│   ├── cs_client.py        # OTCS REST API client — auth, node traversal, download
│   ├── extractor.py        # Content extraction by MIME type (PDF, DOCX, TXT)
│   ├── chunker.py          # Parent-child chunking strategy
│   ├── ingester.py         # ChromaDB ingestion with Ollama embeddings
│   ├── retriever.py        # Hybrid BM25 + vector search with RRF reranking
│   └── app.py              # Streamlit chatbot UI
├── examples/
│   └── demo_crawl.py       # Walk a test OTCS node tree and print document metadata
├── docs/
│   └── architecture.md     # System design notes
├── requirements.txt
└── README.md
```

---

## Architecture

```
OpenText CS  →  cs_client.py  →  extractor.py  →  chunker.py
                                                        ↓
                                              ChromaDB (vector)
                                              Parquet   (cold store)
                                                        ↓
                                    hybrid_search() → RRF rerank → LLM → Answer
```

**Two storage layers:**
- **ChromaDB** — vector hot store, queried at retrieval time
- **Parquet** — cold store of raw text, lets you re-embed with a new model without re-crawling OTCS

---

## Key OTCS API Lessons

Two non-obvious quirks in the OTCS REST API that cost me two days of debugging:

**1. Children use `results`, not `data`**

```python
# Single node endpoint: GET /api/v2/nodes/{id}
response["data"]["properties"]  # ✓ correct

# Children endpoint: GET /api/v2/nodes/{id}/nodes  
response["results"]             # ✓ correct
response["data"]                # ✗ always empty
```

**2. Container detection via flag, not node_type**

```python
# ✗ Only catches folders (type 0), misses workspaces (848), compound docs (136)
is_container = node["data"]["properties"]["node_type"] == 0

# ✓ Correct — covers all container types
is_container = node["data"]["properties"].get("container", False)
```

See [this writeup](https://kapiluthra.github.io/blog-cs-client-debugging.html) for the full debugging story.

---

## Stack

| Component | Library |
|---|---|
| Embeddings | Ollama — `nomic-embed-text` (768 dim) |
| Vector store | ChromaDB (cosine similarity, HNSW) |
| Cold store | Parquet via `pandas` |
| Keyword search | `rank_bm25` |
| Reranking | Reciprocal Rank Fusion (RRF, k=60) |
| Content extraction | `pdfminer.six`, `python-docx` |
| Chunking | `langchain` `RecursiveCharacterTextSplitter` |
| UI | Streamlit |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/kapiluthra/otcs-rag-demo
cd otcs-rag-demo
pip install -r requirements.txt

# 2. Start Ollama and pull the embedding model
ollama pull nomic-embed-text

# 3. Set your OTCS connection (copy from .env.example)
cp .env.example .env
# edit .env with your OTCS_BASE_URL, OTCS_USERNAME, OTCS_PASSWORD, OTCS_ROOT_NODE

# 4. Run the demo crawl to verify connectivity
python examples/demo_crawl.py

# 5. Run ingestion
python src/ingester.py

# 6. Launch the Streamlit UI
streamlit run src/app.py
```

> **Note:** You need a running OpenText Content Server instance. For testing, you can point this at any OTCS 21.x+ environment — the demo crawl will work against any accessible node tree.

---

## Chunking Strategy

Parent-child chunking gives you the best of both worlds:

- **Child chunks (~400 tokens)** are what gets embedded and matched — small enough for precise retrieval
- **Parent chunks (~1500 tokens)** are what gets sent to the LLM — enough context to reason from

```python
# Child chunk retrieved → Parent chunk returned to LLM
child  = "...the contract renewal clause states 30 days notice..."   # matched
parent = "...Section 4.2 Termination. Either party may terminate...  # sent to LLM
         ...30 days written notice...Section 4.3 Survival..."
```

---

## Hybrid Retrieval + RRF

Pure vector search misses exact keyword matches (contract numbers, document IDs, codes). Pure BM25 misses semantic similarity. RRF fuses both ranked lists:

```
score(doc) = Σ 1/(k + rank_in_list)   where k=60
```

In testing against an OTCS corpus, hybrid + RRF improved recall on keyword-heavy queries by ~40% over pure vector search alone.

---

## ACL Security Gates

In production, every retrieved chunk is filtered through OTCS permission checks before being returned. This demo includes the hook but skips the live API call — enable it by uncommenting the permission check in `retriever.py`.

---

## Production Notes

Things that matter at scale that this demo simplifies:

- **Incremental sync**: production uses `OTModifyDate` to only re-ingest changed documents
- **Ticket renewal**: `OTCSTicket` expires after ~30 min; the client auto-renews
- **Parquet cold store**: lets you re-embed with a different model without re-crawling
- **Per-document error handling**: failed documents are logged and retried, not silently skipped

---

## Related Writing

- [Building a Production RAG Pipeline for OpenText Content Server](https://kapiluthra.github.io/blog-rag-opentext-pipeline.html)
- [How I Debugged 150M Document Download Failures in cs_client.py](https://kapiluthra.github.io/blog-cs-client-debugging.html)
- [Clean Architecture in Enterprise ECM](https://kapiluthra.github.io/blog-ecm-architecture.html)

---

## Author

**Kapil Uthra** — Solutions Architect, 13 years OpenText ECM  
[kapiluthra.github.io](https://kapiluthra.github.io) · [LinkedIn](https://linkedin.com/in/kapiluthra)

---

## License

MIT — use freely, attribution appreciated.
