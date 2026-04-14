"""
chunker.py — Parent-child document chunking for RAG

Strategy: split into ~1500-token parent sections, then ~400-token child chunks
with overlap. At retrieval time, child chunks are matched; parent chunks are
returned to the LLM for context.

Why this matters:
  - Child chunks: precise semantic matching (small = focused)
  - Parent chunks: enough context for the LLM to reason from
"""

from dataclasses import dataclass
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Parent splitter: larger sections, no overlap (sections should be coherent units)
_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " "],
)

# Child splitter: smaller chunks with overlap for context continuity
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n", ". ", " "],
)


@dataclass
class Chunk:
    chunk_id: str       # unique ID for this child chunk (stored in ChromaDB)
    parent_id: str      # ID of the parent section
    doc_id: str         # OTCS node ID as string
    doc_name: str       # filename for display
    child_text: str     # text that gets embedded and matched
    parent_text: str    # text returned to the LLM
    modify_date: str    # OTModifyDate for incremental sync
    mime_type: Optional[str] = None


def chunk_document(
    text: str,
    doc_id: str,
    doc_name: str,
    modify_date: str = "",
    mime_type: str = "",
) -> list[Chunk]:
    """Split document text into parent-child chunks.

    Args:
        text: Full extracted text of the document
        doc_id: OTCS node ID (as string)
        doc_name: Document filename (for display/citation)
        modify_date: OTModifyDate string for incremental sync tracking
        mime_type: Original MIME type

    Returns:
        List of Chunk objects ready for ingestion
    """
    if not text or not text.strip():
        return []

    chunks = []
    parents = _parent_splitter.split_text(text)

    for p_idx, parent_text in enumerate(parents):
        parent_id = f"{doc_id}_p{p_idx}"
        children = _child_splitter.split_text(parent_text)

        for c_idx, child_text in enumerate(children):
            if not child_text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{p_idx}_c{c_idx}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    child_text=child_text,
                    parent_text=parent_text,
                    modify_date=modify_date,
                    mime_type=mime_type,
                )
            )

    return chunks
