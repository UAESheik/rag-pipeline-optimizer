"""Unit tests for evaluation metrics and retriever utilities."""
from __future__ import annotations

import pytest
from src.chunking.chunker import Chunk
from src.evaluation.metrics import (
    case1_metrics,
    case2_metrics,
    _token_overlap,
    _doc_recall,
    _groundedness,
)
from src.retrieval.retriever import _rrf_merge, _detect_entities


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}_0", doc_id=doc_id, text=text, metadata={})


def _retrieved(chunks_texts: list[tuple[str, str]]) -> list:
    """Build (Chunk, score) list from (doc_id, text) pairs."""
    return [(_make_chunk(d, t), 1.0) for d, t in chunks_texts]


# ── _token_overlap ────────────────────────────────────────────────────────────

class TestTokenOverlap:
    def test_identical(self):
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _token_overlap("foo bar", "baz qux") == 0.0

    def test_partial(self):
        score = _token_overlap("the cat sat", "the dog sat")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert _token_overlap("", "") == 0.0

    def test_one_empty(self):
        assert _token_overlap("hello", "") == 0.0


# ── _doc_recall ───────────────────────────────────────────────────────────────

class TestDocRecall:
    def test_full_recall(self):
        assert _doc_recall(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_recall(self):
        assert _doc_recall(["c"], ["a", "b"]) == 0.0

    def test_partial_recall(self):
        assert _doc_recall(["a", "c"], ["a", "b"]) == 0.5

    def test_empty_reference(self):
        assert _doc_recall(["a"], []) == 0.0

    def test_empty_retrieved(self):
        assert _doc_recall([], ["a"]) == 0.0


# ── _groundedness ─────────────────────────────────────────────────────────────

class TestGroundedness:
    def test_fully_grounded(self):
        retrieved = _retrieved([("d1", "the answer is forty two")])
        score = _groundedness("the answer is forty two", retrieved)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_not_grounded(self):
        retrieved = _retrieved([("d1", "completely unrelated text here")])
        score = _groundedness("nuclear fusion reactor", retrieved)
        assert score < 0.3

    def test_empty_answer(self):
        retrieved = _retrieved([("d1", "some text")])
        assert _groundedness("", retrieved) == 0.0

    def test_empty_retrieved(self):
        assert _groundedness("some answer", []) == 0.0


# ── _rrf_merge ────────────────────────────────────────────────────────────────

class TestRRFMerge:
    def test_single_list(self):
        ranked = [(0, 0.9), (1, 0.5), (2, 0.1)]
        result = _rrf_merge([ranked])
        indices = [idx for idx, _ in result]
        assert indices[0] == 0  # top ranked should win

    def test_two_lists_agreement(self):
        """Both lists agree on ranking → same order preserved."""
        l1 = [(0, 0.9), (1, 0.5)]
        l2 = [(0, 0.8), (1, 0.4)]
        result = _rrf_merge([l1, l2])
        assert result[0][0] == 0

    def test_two_lists_disagreement(self):
        """RRF should promote doc ranked high in both lists."""
        l1 = [(0, 0.9), (1, 0.1)]  # 0 wins in l1
        l2 = [(1, 0.9), (0, 0.1)]  # 1 wins in l2
        result = _rrf_merge([l1, l2])
        scores = {idx: sc for idx, sc in result}
        # Both should have equal RRF score (rank 0 in one, rank 1 in other)
        assert abs(scores[0] - scores[1]) < 1e-6

    def test_empty_lists(self):
        assert _rrf_merge([]) == []

    def test_scores_positive(self):
        ranked = [(i, float(10 - i)) for i in range(5)]
        result = _rrf_merge([ranked])
        for _, score in result:
            assert score > 0

    def test_k_parameter(self):
        """Lower k gives higher weight to top-ranked docs."""
        ranked = [(0, 1.0), (1, 0.5)]
        result_k1 = _rrf_merge([ranked], k=1)
        result_k100 = _rrf_merge([ranked], k=100)
        gap_k1 = result_k1[0][1] - result_k1[1][1]
        gap_k100 = result_k100[0][1] - result_k100[1][1]
        assert gap_k1 > gap_k100


# ── _detect_entities ──────────────────────────────────────────────────────────

class TestDetectEntities:
    def test_known_entity(self):
        entities = _detect_entities("The kyc process requires verification")
        assert "regulation" in entities

    def test_multiple_entities(self):
        entities = _detect_entities("kyc and aml compliance")
        assert "regulation" in entities

    def test_no_entity(self):
        entities = _detect_entities("the weather is nice today")
        assert entities == []

    def test_case_insensitive(self):
        """Entities detected case-insensitively via lower()."""
        entities = _detect_entities("KYC verification")
        # lowercased in _detect_entities so 'kyc' should match
        assert "regulation" in entities

    def test_fee_entity(self):
        entities = _detect_entities("what is the fee for this service")
        assert "pricing" in entities


# ── case1_metrics ─────────────────────────────────────────────────────────────

class TestCase1Metrics:
    def test_perfect_retrieval(self):
        retrieved = _retrieved([("doc1", "the capital of France is Paris")])
        m = case1_metrics(
            retrieved=retrieved,
            answer="Paris",
            reference_answer="Paris",
            reference_doc_ids=["doc1"],
            reference_context="the capital of France is Paris",
            query="What is the capital of France?",
        )
        assert m["composite"] > 0.5
        assert m["context_recall"] > 0.0
        assert m["faithfulness"] > 0.0

    def test_wrong_doc(self):
        retrieved = _retrieved([("doc2", "some unrelated text")])
        m = case1_metrics(
            retrieved=retrieved,
            answer="London",
            reference_answer="Paris",
            reference_doc_ids=["doc1"],
        )
        # doc2 not in reference → doc recall = 0
        assert m["context_recall"] < 0.5

    def test_output_keys(self):
        retrieved = _retrieved([("d1", "test text")])
        m = case1_metrics(
            retrieved=retrieved,
            answer="test",
            reference_answer="test",
            reference_doc_ids=["d1"],
        )
        for key in ("composite", "context_recall", "answer_similarity", "faithfulness"):
            assert key in m

    def test_composite_bounded(self):
        retrieved = _retrieved([("d1", "hello world answer")])
        m = case1_metrics(
            retrieved=retrieved,
            answer="answer",
            reference_answer="answer",
            reference_doc_ids=["d1"],
        )
        assert 0.0 <= m["composite"] <= 1.0


# ── case2_metrics ─────────────────────────────────────────────────────────────

class TestCase2Metrics:
    def test_good_coverage(self):
        retrieved = _retrieved([("doc1", "relevant evidence here")])
        m = case2_metrics(
            retrieved=retrieved,
            answer="relevant answer",
            reference_doc_ids=["doc1"],
        )
        assert m["retrieval_coverage_proxy"] == 1.0
        assert m["composite"] > 0.0

    def test_no_coverage(self):
        retrieved = _retrieved([("doc2", "some text")])
        m = case2_metrics(
            retrieved=retrieved,
            answer="answer",
            reference_doc_ids=["doc1"],
        )
        assert m["retrieval_coverage_proxy"] == 0.0

    def test_output_keys(self):
        retrieved = _retrieved([("d1", "text")])
        m = case2_metrics(
            retrieved=retrieved,
            answer="text",
            reference_doc_ids=["d1"],
        )
        for key in ("composite", "retrieval_coverage_proxy", "groundedness", "citation_quality"):
            assert key in m

    def test_composite_bounded(self):
        retrieved = _retrieved([("d1", "answer text here")])
        m = case2_metrics(
            retrieved=retrieved,
            answer="answer",
            reference_doc_ids=["d1"],
        )
        assert 0.0 <= m["composite"] <= 1.0
