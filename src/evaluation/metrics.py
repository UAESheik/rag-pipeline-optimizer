from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]


def _token_overlap(a: str, b: str) -> float:
    """计算两个文本的 token 交并比。"""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _bertscore_similarity(answer: str, reference_answer: str, use_bertscore: bool) -> float:
    """优先使用 BERTScore，失败时回退 token overlap。"""
    proxy = round(_token_overlap(answer, reference_answer), 4)
    if not use_bertscore:
        return proxy
    try:
        from bert_score import score

        _, _, f1 = score(
            [answer],
            [reference_answer],
            lang="en",
            verbose=False,
        )
        return round(float(f1.mean().item()), 4)
    except Exception:
        return proxy


def _doc_recall(retrieved_ids: List[str], reference_ids: List[str]) -> float:
    """按文档 ID 计算召回率。"""
    if not reference_ids:
        return 0.0
    return len(set(retrieved_ids) & set(reference_ids)) / len(set(reference_ids))


def _groundedness(answer: str, retrieved: List[RetrievedChunk]) -> float:
    """答案 token 在检索上下文中的支撑比例。"""
    if not answer or not retrieved:
        return 0.0
    ctx = " ".join(c.text for c, _ in retrieved).lower()
    tokens = answer.lower().split()
    hits = sum(1 for tok in tokens if tok in ctx)
    return round(min(1.0, hits / max(1, len(tokens))), 4)


def _context_recall_text(retrieved: List[RetrievedChunk], reference_context: str) -> float:
    """Case1 文本级上下文召回（与参考上下文的最高相似）。"""
    if not retrieved or not reference_context:
        return 0.0
    best = max(_token_overlap(c.text, reference_context) for c, _ in retrieved)
    return round(best, 4)


def _ragas_or_proxy_relevancy(
    query: str,
    answer: str,
    retrieved: List[RetrievedChunk],
    use_ragas: bool,
) -> Tuple[float, float]:
    """优先走 RAGAS，失败时回退到代理 token overlap。"""
    proxy_answer_rel = round(_token_overlap(answer, query), 4)
    if not retrieved:
        return proxy_answer_rel, 0.0
    proxy_context_rel = round(sum(_token_overlap(query, c.text) for c, _ in retrieved) / len(retrieved), 4)

    if not use_ragas:
        return proxy_answer_rel, proxy_context_rel

    try:
        from ragas import SingleTurnSample
        from ragas.metrics import answer_relevancy, context_relevancy

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=[c.text for c, _ in retrieved],
        )

        async def _score_async() -> Tuple[float, float]:
            ans = await answer_relevancy.single_turn_ascore(sample)
            ctx = await context_relevancy.single_turn_ascore(sample)
            return float(ans), float(ctx)

        ans_rel, ctx_rel = asyncio.run(_score_async())
        return round(ans_rel, 4), round(ctx_rel, 4)
    except Exception:
        return proxy_answer_rel, proxy_context_rel


def case1_metrics(
    retrieved: List[RetrievedChunk],
    answer: str,
    reference_answer: str,
    reference_doc_ids: List[str],
    reference_context: str = "",
    query: str = "",
    use_ragas: bool = False,
    use_bertscore: bool = False,
) -> Dict[str, float]:
    """Case1（有监督）指标。"""
    retrieved_ids = [c.doc_id for c, _ in retrieved]
    doc_hit_rate = _doc_recall(retrieved_ids, reference_doc_ids)
    text_sim = _context_recall_text(retrieved, reference_context) if reference_context else 0.0
    context_recall = round(0.6 * text_sim + 0.4 * doc_hit_rate, 4)
    answer_similarity = _bertscore_similarity(answer, reference_answer, use_bertscore=use_bertscore)
    faithfulness = _groundedness(answer, retrieved)
    answer_relevancy, context_relevancy = _ragas_or_proxy_relevancy(
        query=query,
        answer=answer,
        retrieved=retrieved,
        use_ragas=use_ragas,
    )
    composite = round(0.4 * context_recall + 0.4 * answer_similarity + 0.2 * faithfulness, 4)
    return {
        "composite": composite,
        "context_recall": context_recall,
        "answer_similarity": answer_similarity,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_relevancy": context_relevancy,
    }


def case2_metrics(
    retrieved: List[RetrievedChunk],
    answer: str,
    reference_doc_ids: List[str],
    query: str = "",
    use_ragas: bool = False,
) -> Dict[str, float]:
    """Case2（弱监督）指标，包含 RAGAS 项（失败时回退代理）。"""
    retrieved_ids = [c.doc_id for c, _ in retrieved]
    coverage = _doc_recall(retrieved_ids, reference_doc_ids)
    groundedness = _groundedness(answer, retrieved)
    cited = [t.strip("[]") for t in answer.split() if t.startswith("[") and t.endswith("]")]
    citation_quality = 0.0 if not retrieved_ids else round(len(set(cited) & set(retrieved_ids)) / len(set(retrieved_ids)), 4)
    answer_relevancy, context_relevancy = _ragas_or_proxy_relevancy(
        query=query,
        answer=answer,
        retrieved=retrieved,
        use_ragas=use_ragas,
    )

    composite = round(0.5 * coverage + 0.3 * groundedness + 0.2 * citation_quality, 4)
    return {
        "composite": composite,
        "retrieval_coverage_proxy": round(coverage, 4),
        "groundedness": groundedness,
        "citation_quality": citation_quality,
        "answer_relevancy": answer_relevancy,
        "context_relevancy": context_relevancy,
    }
