from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]

# 全局 CrossEncoder 缓存
_CE_CACHE: Dict[str, object] = {}


def _load_cross_encoder(model_name: str):
    """懒加载 CrossEncoder 模型（仅使用本地缓存，不触发网络下载）。"""
    if model_name not in _CE_CACHE:
        try:
            import os
            from sentence_transformers import CrossEncoder
            # local_files_only=True：仅加载已缓存模型，避免超时
            _CE_CACHE[model_name] = CrossEncoder(
                model_name, max_length=512,
                tokenizer_kwargs={"local_files_only": True},
            )
        except Exception:
            _CE_CACHE[model_name] = None
    return _CE_CACHE[model_name]


def rerank(
    results: List[RetrievedChunk],
    enabled: bool,
    query: str = "",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[RetrievedChunk]:
    """重排序：优先使用真实 CrossEncoder，不可用时回退轻量 TF-IDF CE。"""
    if not enabled or not results:
        return results

    if query:
        ce_model = _load_cross_encoder(model_name)
        if ce_model is not None:
            # 真实 CrossEncoder：query-chunk 对联合打分
            pairs = [[query, chunk.text] for chunk, _ in results]
            ce_scores = ce_model.predict(pairs)
            rescored: List[RetrievedChunk] = []
            for (chunk, orig_score), ce_score in zip(results, ce_scores):
                # 原始分数（BM25/dense）与 CE 分数加权融合
                combined = round(0.3 * float(orig_score) + 0.7 * float(ce_score), 6)
                rescored.append((chunk, combined))
            return sorted(rescored, key=lambda x: x[1], reverse=True)

        # 回退：TF-IDF CE 代理
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        def _ce_proxy(q: str, text: str) -> float:
            combined = q + " [SEP] " + text
            try:
                vect = TfidfVectorizer().fit([combined])
                return float(cosine_similarity(vect.transform([q]), vect.transform([text]))[0][0])
            except Exception:
                return 0.0

        rescored = []
        for chunk, orig_score in results:
            ce_score = _ce_proxy(query, chunk.text)
            combined = round(0.3 * orig_score + 0.7 * ce_score, 6)
            rescored.append((chunk, combined))
        return sorted(rescored, key=lambda x: x[1], reverse=True)

    # 无 query 时退化为按分数排序
    return sorted(results, key=lambda x: x[1], reverse=True)
