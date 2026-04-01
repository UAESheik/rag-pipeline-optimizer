from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

from src.chunking.chunker import Chunk
from src.utils.logging_utils import log_external_event

RetrievedChunk = Tuple[Chunk, float]


def _token_overlap(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _bertscore_similarity(answer: str, reference_answer: str, use_bertscore: bool) -> float:
    proxy = round(_token_overlap(answer, reference_answer), 4)
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


def _groundedness_proxy(answer: str, retrieved: List[RetrievedChunk]) -> float:
    if not answer or not retrieved:
        return 0.0
    ctx = " ".join(c.text for c, _ in retrieved).lower()
    tokens = answer.lower().split()
    hits = sum(1 for tok in tokens if tok in ctx)
    return round(min(1.0, hits / max(1, len(tokens))), 4)


def _context_recall_text(retrieved: List[RetrievedChunk], reference_context: str) -> float:
    if not retrieved or not reference_context:
        return 0.0
    best = max(_token_overlap(c.text, reference_context) for c, _ in retrieved)
    return round(best, 4)


def _ragas_scores(query: str, answer: str, retrieved: List[RetrievedChunk], use_ragas: bool) -> Dict[str, float]:
    proxy_answer_rel = round(_token_overlap(answer, query), 4)
    proxy_context_rel = 0.0 if not retrieved else round(sum(_token_overlap(query, c.text) for c, _ in retrieved) / len(retrieved), 4)
    proxy_faith = _groundedness_proxy(answer, retrieved)

    if not use_ragas:
        return {
            "answer_relevancy": proxy_answer_rel,
            "context_relevancy": proxy_context_rel,
            "faithfulness": proxy_faith,
            "ragas_used": 0.0,
        }

    try:
        log_external_event("metrics", "ragas", "start", "single_turn_ascore")
        from ragas import SingleTurnSample
        from ragas.metrics import answer_relevancy, context_relevancy, faithfulness

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=[c.text for c, _ in retrieved],
        )

        async def _score_async() -> Tuple[float, float, float]:
            a = await answer_relevancy.single_turn_ascore(sample)
            c = await context_relevancy.single_turn_ascore(sample)
            f = await faithfulness.single_turn_ascore(sample)
            return float(a), float(c), float(f)

        a_rel, c_rel, faith = asyncio.run(_score_async())
        log_external_event("metrics", "ragas", "success", "single_turn_ascore")
        return {
            "answer_relevancy": round(a_rel, 4),
            "context_relevancy": round(c_rel, 4),
            "faithfulness": round(faith, 4),
            "ragas_used": 1.0,
        }
    except Exception as e:
        log_external_event("metrics", "ragas", "fallback", str(e))
        return {
            "answer_relevancy": proxy_answer_rel,
            "context_relevancy": proxy_context_rel,
            "faithfulness": proxy_faith,
            "ragas_used": 0.0,
        }


def case1_metrics(
    retrieved: List[RetrievedChunk],
    answer: str,
    reference_answer: str,
    reference_doc_ids: List[str],
    reference_context: str = "",
    query: str = "",
    use_ragas: bool = False,
    use_bertscore: bool = False,
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    retrieved_ids = [c.doc_id for c, _ in retrieved]
    doc_hit_rate = _doc_recall(retrieved_ids, reference_doc_ids)
    text_sim = _context_recall_text(retrieved, reference_context) if reference_context else 0.0
    context_recall = round(0.6 * text_sim + 0.4 * doc_hit_rate, 4)
    answer_similarity = _bertscore_similarity(answer, reference_answer, use_bertscore=use_bertscore)

    ragas_scores = _ragas_scores(query=query, answer=answer, retrieved=retrieved, use_ragas=use_ragas)
    faithfulness_score = ragas_scores["faithfulness"]

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
        + float(w.get("answer_relevancy", 0.05)) * ragas_scores["answer_relevancy"]
        + float(w.get("context_relevancy", 0.05)) * ragas_scores["context_relevancy"]
    )
    # 降低“虚假信号”风险：真实评估信号不足时下调综合分
    signal_quality = round(0.6 * ragas_scores["ragas_used"] + 0.4 * (1.0 if use_bertscore else 0.0), 4)
    signal_penalty = 0.7 + 0.3 * signal_quality
    composite = round(composite_raw * signal_penalty, 4)

    return {
        "composite": composite,
        "context_recall": context_recall,
        "answer_similarity": answer_similarity,
        "faithfulness": faithfulness_score,
        "answer_relevancy": ragas_scores["answer_relevancy"],
        "context_relevancy": ragas_scores["context_relevancy"],
        "ragas_used": ragas_scores["ragas_used"],
        "signal_quality": signal_quality,
    }


def case2_metrics(
    retrieved: List[RetrievedChunk],
    answer: str,
    reference_doc_ids: List[str],
    query: str = "",
    use_ragas: bool = False,
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    retrieved_ids = [c.doc_id for c, _ in retrieved]
    coverage = _doc_recall(retrieved_ids, reference_doc_ids)

    ragas_scores = _ragas_scores(query=query, answer=answer, retrieved=retrieved, use_ragas=use_ragas)
    groundedness = max(_groundedness_proxy(answer, retrieved), ragas_scores["faithfulness"])

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
        + float(w.get("answer_relevancy", 0.05)) * ragas_scores["answer_relevancy"]
        + float(w.get("context_relevancy", 0.05)) * ragas_scores["context_relevancy"]
    )

    signal_quality = round(0.7 * ragas_scores["ragas_used"] + 0.3 * (1.0 if citation_quality > 0 else 0.0), 4)
    signal_penalty = 0.75 + 0.25 * signal_quality
    composite = round(composite_raw * signal_penalty, 4)

    return {
        "composite": composite,
        "retrieval_coverage_proxy": round(coverage, 4),
        "groundedness": groundedness,
        "citation_quality": citation_quality,
        "answer_relevancy": ragas_scores["answer_relevancy"],
        "context_relevancy": ragas_scores["context_relevancy"],
        "ragas_used": ragas_scores["ragas_used"],
        "signal_quality": signal_quality,
    }
