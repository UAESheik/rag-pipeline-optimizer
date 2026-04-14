from __future__ import annotations

import re
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _token_overlap(a: str, b: str) -> float:
    sa = set(_tokens(a))
    sb = set(_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _token_f1(pred: str, ref: str) -> float:
    p = _tokens(pred)
    r = _tokens(ref)
    if not p or not r:
        return 0.0
    pset, rset = set(p), set(r)
    hit = len(pset & rset)
    precision = hit / max(1, len(pset))
    recall = hit / max(1, len(rset))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tfidf_cosine(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform([a, b])
        sim = (mat[0] @ mat[1].T).toarray()[0][0]
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0


def _bertscore_similarity(answer: str, reference_answer: str, use_bertscore: bool) -> float:
    proxy = round(0.7 * _token_f1(answer, reference_answer) + 0.3 * _tfidf_cosine(answer, reference_answer), 4)
    if not use_bertscore:
        return proxy
    try:
        from bert_score import score

        _, _, f1 = score([answer], [reference_answer], lang="en", verbose=False)
        return round(float(f1.mean().item()), 4)
    except Exception:
        return proxy


def _doc_recall(retrieved_ids: List[str], reference_ids: List[str]) -> float:
    if not reference_ids:
        return 0.0
    return len(set(retrieved_ids) & set(reference_ids)) / len(set(reference_ids))


def _groundedness(answer: str, retrieved: List[RetrievedChunk]) -> float:
    if not answer or not retrieved:
        return 0.0

    ctx = " ".join(c.text for c, _ in retrieved)
    support = _token_f1(answer, ctx)

    retrieved_ids = {c.doc_id for c, _ in retrieved}
    cited_ids = {t.strip("[]") for t in answer.split() if t.startswith("[") and t.endswith("]")}
    if cited_ids:
        citation_support = len(cited_ids & retrieved_ids) / len(cited_ids)
        score = 0.75 * support + 0.25 * citation_support
    else:
        score = support

    return round(float(max(0.0, min(1.0, score))), 4)


def _context_recall_text(retrieved: List[RetrievedChunk], reference_context: str) -> float:
    if not retrieved or not reference_context:
        return 0.0
    merged = " ".join(c.text for c, _ in retrieved)
    global_recall = _token_f1(merged, reference_context)
    best_chunk = max(_token_f1(c.text, reference_context) for c, _ in retrieved)
    return round(0.6 * global_recall + 0.4 * best_chunk, 4)


def _local_scores(query: str, answer: str, retrieved: List[RetrievedChunk]) -> Dict[str, float]:
    answer_relevancy = round(0.6 * _token_overlap(answer, query) + 0.4 * _tfidf_cosine(answer, query), 4)
    if not retrieved:
        context_relevancy = 0.0
    else:
        context_relevancy = round(
            sum(0.6 * _token_overlap(query, c.text) + 0.4 * _tfidf_cosine(query, c.text[:500]) for c, _ in retrieved)
            / len(retrieved),
            4,
        )
    faithfulness = _groundedness(answer, retrieved)
    return {
        "answer_relevancy": answer_relevancy,
        "context_relevancy": context_relevancy,
        "faithfulness": faithfulness,
    }


def case1_metrics(
    retrieved: List[RetrievedChunk],
    answer: str,
    reference_answer: str,
    reference_doc_ids: List[str],
    reference_context: str = "",
    query: str = "",
    use_bertscore: bool = False,
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    retrieved_ids = [c.doc_id for c, _ in retrieved]
    doc_hit_rate = _doc_recall(retrieved_ids, reference_doc_ids)
    text_sim = _context_recall_text(retrieved, reference_context) if reference_context else 0.0
    context_recall = round(0.6 * text_sim + 0.4 * doc_hit_rate, 4)
    answer_similarity = _bertscore_similarity(answer, reference_answer, use_bertscore=use_bertscore)

    local_scores = _local_scores(query=query, answer=answer, retrieved=retrieved)
    faithfulness_score = local_scores["faithfulness"]

    w = weights or {
        "context_recall": 0.35,
        "answer_similarity": 0.35,
        "faithfulness": 0.2,
        "answer_relevancy": 0.05,
        "context_relevancy": 0.05,
    }
    composite_raw = (
        float(w.get("context_recall", 0.35)) * context_recall
        + float(w.get("answer_similarity", 0.35)) * answer_similarity
        + float(w.get("faithfulness", 0.2)) * faithfulness_score
        + float(w.get("answer_relevancy", 0.05)) * local_scores["answer_relevancy"]
        + float(w.get("context_relevancy", 0.05)) * local_scores["context_relevancy"]
    )
    signal_quality = 1.0 if use_bertscore else 0.7
    signal_penalty = 0.75 + 0.25 * signal_quality
    composite = round(composite_raw * signal_penalty, 4)

    return {
        "composite": composite,
        "context_recall": context_recall,
        "answer_similarity": answer_similarity,
        "faithfulness": faithfulness_score,
        "answer_relevancy": local_scores["answer_relevancy"],
        "context_relevancy": local_scores["context_relevancy"],
        "signal_quality": signal_quality,
    }


def case2_metrics(
    retrieved: List[RetrievedChunk],
    answer: str,
    reference_doc_ids: List[str],
    query: str = "",
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    retrieved_ids = [c.doc_id for c, _ in retrieved]
    coverage = _doc_recall(retrieved_ids, reference_doc_ids)

    local_scores = _local_scores(query=query, answer=answer, retrieved=retrieved)
    groundedness = max(_groundedness(answer, retrieved), local_scores["faithfulness"])

    cited = [t.strip("[]") for t in answer.split() if t.startswith("[") and t.endswith("]")]
    citation_quality = 0.0 if not retrieved_ids else round(len(set(cited) & set(retrieved_ids)) / len(set(retrieved_ids)), 4)

    w = weights or {
        "retrieval_coverage_proxy": 0.4,
        "groundedness": 0.3,
        "citation_quality": 0.2,
        "answer_relevancy": 0.05,
        "context_relevancy": 0.05,
    }
    composite_raw = (
        float(w.get("retrieval_coverage_proxy", 0.4)) * coverage
        + float(w.get("groundedness", 0.3)) * groundedness
        + float(w.get("citation_quality", 0.2)) * citation_quality
        + float(w.get("answer_relevancy", 0.05)) * local_scores["answer_relevancy"]
        + float(w.get("context_relevancy", 0.05)) * local_scores["context_relevancy"]
    )

    signal_quality = round(0.6 + 0.4 * citation_quality, 4)
    signal_penalty = 0.75 + 0.25 * signal_quality
    composite = round(composite_raw * signal_penalty, 4)

    return {
        "composite": composite,
        "retrieval_coverage_proxy": round(coverage, 4),
        "groundedness": groundedness,
        "citation_quality": citation_quality,
        "answer_relevancy": local_scores["answer_relevancy"],
        "context_relevancy": local_scores["context_relevancy"],
        "signal_quality": signal_quality,
    }
