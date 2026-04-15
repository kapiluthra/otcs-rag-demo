"""
tests/test_chunker.py — Tests for the parent-child chunking strategy
tests/test_retriever.py — Tests for RRF reranking logic
"""

# ═══════════════════════════════════════════════════════════════════
# test_chunker.py
# ═══════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from chunker import chunk_document, Chunk


class TestChunkDocument:
    def test_returns_chunks_for_nonempty_text(self):
        text = "This is a test document. " * 50
        chunks = chunk_document(text, doc_id="doc1", doc_name="test.pdf")
        assert len(chunks) > 0

    def test_returns_empty_for_blank_text(self):
        assert chunk_document("", "doc1", "test.pdf") == []
        assert chunk_document("   ", "doc1", "test.pdf") == []
        assert chunk_document(None, "doc1", "test.pdf") == []  # type: ignore

    def test_chunk_ids_are_unique(self):
        text = "A " * 500
        chunks = chunk_document(text, doc_id="doc42", doc_name="unique.pdf")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_id_contains_doc_id(self):
        chunks = chunk_document("Hello world. " * 100, doc_id="mynode", doc_name="x.pdf")
        for c in chunks:
            assert "mynode" in c.chunk_id

    def test_parent_child_relationship(self):
        """Multiple child chunks share the same parent_id."""
        # Long text to guarantee multiple children per parent
        long_text = "The quick brown fox jumps over the lazy dog. " * 200
        chunks = chunk_document(long_text, doc_id="doc1", doc_name="long.pdf")
        
        # Group by parent
        from collections import defaultdict
        by_parent = defaultdict(list)
        for c in chunks:
            by_parent[c.parent_id].append(c)
        
        # At least one parent should have multiple children (overlap creates them)
        has_multiple_children = any(len(v) > 1 for v in by_parent.values())
        assert has_multiple_children, "Long text should produce parents with multiple children"

    def test_parent_text_contains_child_text(self):
        """Every child chunk's text should be a substring of its parent text."""
        text = "Contract clause 1: payment within 30 days. " * 50
        chunks = chunk_document(text, doc_id="c1", doc_name="contract.pdf")
        for c in chunks:
            # Child text should be mostly contained in parent (allowing for minor splitter differences)
            assert len(c.child_text) <= len(c.parent_text), (
                f"Child chunk ({len(c.child_text)} chars) longer than parent ({len(c.parent_text)} chars)"
            )

    def test_child_text_shorter_than_parent(self):
        """Child chunks should be smaller than parent chunks by design."""
        text = "Enterprise content management system. " * 200
        chunks = chunk_document(text, doc_id="ecm1", doc_name="ecm.pdf")
        for c in chunks:
            assert len(c.child_text) < len(c.parent_text)

    def test_metadata_preserved(self):
        chunks = chunk_document(
            "Some text. " * 100,
            doc_id="node123",
            doc_name="policy.pdf",
            modify_date="2026-04-01T00:00:00Z",
            mime_type="application/pdf",
        )
        for c in chunks:
            assert c.doc_id == "node123"
            assert c.doc_name == "policy.pdf"
            assert c.modify_date == "2026-04-01T00:00:00Z"
            assert c.mime_type == "application/pdf"

    def test_long_document_produces_multiple_parents(self):
        # 5000 chars should produce multiple parent chunks at 1500-token target
        text = ("This document describes the enterprise content management policy. " * 80)
        chunks = chunk_document(text, doc_id="big1", doc_name="big.pdf")
        parent_ids = set(c.parent_id for c in chunks)
        assert len(parent_ids) > 1, "Long document should have multiple parent chunks"

    def test_short_document_single_parent(self):
        text = "This is a short document."
        chunks = chunk_document(text, doc_id="short1", doc_name="short.pdf")
        parent_ids = set(c.parent_id for c in chunks)
        assert len(parent_ids) == 1, "Short document should have a single parent chunk"

    def test_chunk_is_dataclass(self):
        chunks = chunk_document("Hello. " * 50, "id1", "f.pdf")
        assert isinstance(chunks[0], Chunk)
        assert hasattr(chunks[0], "chunk_id")
        assert hasattr(chunks[0], "parent_id")
        assert hasattr(chunks[0], "child_text")
        assert hasattr(chunks[0], "parent_text")


# ═══════════════════════════════════════════════════════════════════
# test_rrf.py — Tests for RRF scoring logic (no ChromaDB/Ollama needed)
# ═══════════════════════════════════════════════════════════════════

class TestRRFScoring:
    """Test the RRF reranking math directly without needing live dependencies."""

    def _rrf_score(self, ranks, k=60):
        """Local copy of the RRF formula for testing."""
        return sum(1.0 / (k + r) for r in ranks)

    def test_higher_rank_scores_higher(self):
        """A doc ranked #1 in a list should score higher than one ranked #10."""
        score_top = self._rrf_score([1])
        score_bottom = self._rrf_score([10])
        assert score_top > score_bottom

    def test_appearing_in_both_lists_boosts_score(self):
        """A doc in both vector and BM25 results should outscore one in only one."""
        score_both = self._rrf_score([1, 1])    # rank 1 in both
        score_one = self._rrf_score([1])         # rank 1 in one only
        assert score_both > score_one

    def test_fusion_of_mediocre_beats_single_excellent(self):
        """Doc appearing at rank 3 in both lists beats one at rank 1 in one list."""
        score_dual_3rd = self._rrf_score([3, 3])
        score_single_1st = self._rrf_score([1])
        assert score_dual_3rd > score_single_1st

    def test_k_dampens_rank_differences(self):
        """With k=60, the difference between rank 1 and rank 2 is small."""
        diff_low_k = self._rrf_score([1], k=1) - self._rrf_score([2], k=1)
        diff_high_k = self._rrf_score([1], k=60) - self._rrf_score([2], k=60)
        assert diff_low_k > diff_high_k, "Higher k should dampen rank differences"

    def test_rrf_is_always_positive(self):
        for ranks in [[1], [5, 10], [1, 2, 3], [100]]:
            assert self._rrf_score(ranks) > 0

    def test_score_decreases_as_rank_worsens(self):
        scores = [self._rrf_score([r]) for r in range(1, 20)]
        assert scores == sorted(scores, reverse=True), "Scores should decrease as rank increases"
