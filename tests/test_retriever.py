"""Unit tests for Retriever (bm25 / dense / hybrid) and chunker."""
from __future__ import annotations

import pytest
from src.chunking.chunker import Chunk, chunk_document
from src.retrieval.retriever import Retriever, _rrf_merge


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_corpus() -> list[Chunk]:
    """Small deterministic corpus for retrieval tests."""
    docs = [
        ("doc1", "The KYC process requires identity verification documents."),
        ("doc2", "Annual fees and pricing structure for premium accounts."),
        ("doc3", "Fraud detection uses machine learning to flag suspicious activity."),
        ("doc4", "Credit scoring evaluates loan eligibility and creditworthiness."),
        ("doc5", "GDPR compliance mandates strict personal data protection policies."),
    ]
    chunks = []
    for doc_id, text in docs:
        chunks.append(Chunk(chunk_id=f"{doc_id}_0", doc_id=doc_id, text=text, metadata={}))
    return chunks


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def setup_method(self):
        self.chunks = _make_corpus()
        self.retriever = Retriever(self.chunks, retriever_type="bm25")

    def test_returns_results(self):
        results = self.retriever.retrieve("kyc identity", top_k=3)
        assert len(results) > 0

    def test_top_k_respected(self):
        results = self.retriever.retrieve("fee pricing", top_k=2)
        assert len(results) <= 2

    def test_relevant_doc_ranked_high(self):
        results = self.retriever.retrieve("kyc identity verification", top_k=5)
        top_doc_ids = [c.doc_id for c, _ in results]
        assert "doc1" in top_doc_ids[:2]

    def test_scores_are_floats(self):
        results = self.retriever.retrieve("fraud detection", top_k=3)
        for _, score in results:
            assert isinstance(score, float)

    def test_descending_scores(self):
        results = self.retriever.retrieve("credit loan", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query(self):
        results = self.retriever.retrieve("", top_k=3)
        assert isinstance(results, list)

    def test_metadata_entities_injected(self):
        """LightRAG风格：构建索引时应注入 metadata['entities']。"""
        for chunk in self.chunks:
            assert "entities" in chunk.metadata
            assert isinstance(chunk.metadata["entities"], list)


# ── Dense (TF-IDF fallback) Retriever ────────────────────────────────────────

class TestDenseRetriever:
    """Tests dense mode; in offline env falls back to TF-IDF which is fine."""

    def setup_method(self):
        self.chunks = _make_corpus()
        self.retriever = Retriever(self.chunks, retriever_type="dense", embedding_model="bge-small")

    def test_returns_results(self):
        results = self.retriever.retrieve("personal data protection", top_k=3)
        assert len(results) > 0

    def test_top_k_respected(self):
        results = self.retriever.retrieve("fraud", top_k=2)
        assert len(results) <= 2

    def test_descending_scores(self):
        results = self.retriever.retrieve("gdpr compliance", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_type(self):
        results = self.retriever.retrieve("credit scoring", top_k=3)
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)


# ── Hybrid (RRF) Retriever ────────────────────────────────────────────────────

class TestHybridRetriever:
    def setup_method(self):
        self.chunks = _make_corpus()
        self.retriever = Retriever(self.chunks, retriever_type="hybrid")

    def test_returns_results(self):
        results = self.retriever.retrieve("kyc fraud", top_k=3)
        assert len(results) > 0

    def test_top_k_respected(self):
        results = self.retriever.retrieve("fee", top_k=2)
        assert len(results) <= 2

    def test_descending_scores(self):
        results = self.retriever.retrieve("credit loan eligibility", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_covers_bm25_results(self):
        """Hybrid should retrieve doc1 for 'kyc' query (BM25 should surface it)."""
        results = self.retriever.retrieve("kyc identity", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "doc1" in doc_ids


# ── Metadata Filter ───────────────────────────────────────────────────────────

class TestMetadataFilter:
    def setup_method(self):
        chunks = [
            Chunk("a_0", "a", "kyc document", {"section_title": "kyc", "entities": ["regulation"]}),
            Chunk("b_0", "b", "fee structure", {"section_title": "fee", "entities": ["pricing"]}),
            Chunk("c_0", "c", "general info", {"section_title": "general", "entities": []}),
        ]
        self.retriever = Retriever(chunks, retriever_type="bm25")

    def test_filter_by_section_title(self):
        results = self.retriever.retrieve(
            "kyc", top_k=5, metadata_filter={"section_title": "kyc"}
        )
        for chunk, _ in results:
            assert "kyc" in str(chunk.metadata.get("section_title", "")).lower()

    def test_no_filter_returns_all(self):
        results = self.retriever.retrieve("info", top_k=10)
        assert len(results) <= 3


# ── chunk_document ────────────────────────────────────────────────────────────

class TestChunkDocument:
    def test_basic_sentence_chunking(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_document("doc1", text, strategy="sentence", size=50)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.doc_id == "doc1"
            assert c.text.strip()

    def test_chunk_ids_unique(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_document("doc1", text, strategy="token", size=50, overlap_size=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_fields_present(self):
        chunks = chunk_document("doc1", "Hello world.", strategy="sentence", size=100)
        for c in chunks:
            assert "doc_id" in c.metadata
            assert "chunk_index" in c.metadata
            assert "type" in c.metadata

    def test_empty_text_returns_chunk(self):
        chunks = chunk_document("doc1", "")
        assert len(chunks) == 1

    def test_semantic_strategy(self):
        text = " ".join([f"word{i}" for i in range(300)])
        chunks = chunk_document(
            "doc1", text,
            strategy="semantic",
            size=100,
            semantic_min_size=50,
            semantic_max_size=150,
        )
        assert len(chunks) >= 1

    def test_table_preserved_as_atomic(self):
        text = "Some text.\n| col1 | col2 |\n| a | b |\nMore text."
        chunks = chunk_document("doc1", text, strategy="sentence", size=50)
        types = [c.metadata.get("type") for c in chunks]
        assert "table" in types

    def test_overlap_injection(self):
        text = " ".join([f"token{i}" for i in range(100)])
        chunks_no_overlap = chunk_document("doc1", text, strategy="token", size=30, overlap_size=0)
        chunks_with_overlap = chunk_document("doc1", text, strategy="token", size=30, overlap_size=10)
        # chunks with overlap should have longer or equal texts (overlap adds tokens)
        avg_no = sum(len(c.text.split()) for c in chunks_no_overlap) / max(1, len(chunks_no_overlap))
        avg_with = sum(len(c.text.split()) for c in chunks_with_overlap) / max(1, len(chunks_with_overlap))
        assert avg_with >= avg_no
