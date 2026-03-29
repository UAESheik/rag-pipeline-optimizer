from __future__ import annotations

"""
retriever.py

借鉴两个框架的核心思想：

1. FlashRAG（标准化检索接口 + 多路融合）：
   - 统一 Retriever 接口，支持 bm25 / dense / hybrid 三种模式
   - hybrid 模式使用 RRF（Reciprocal Rank Fusion）融合，而非简单加权
     RRF 公式：score(d) = Σ 1/(k + rank_i(d))，k=60（Robertson et al.）
   - 与 FlashRAG 区别：自实现 RRF，不依赖外部框架，逻辑完全透明

2. LightRAG（实体感知元数据注入）：
   - 构建索引时，对每个 chunk 检测命名实体并注入 metadata["entities"]
   - 检索时支持基于实体的元数据过滤，模拟 LightRAG 图谱关联查询入口
   - 与 LightRAG 区别：当前为轻量规则实体识别，真实部署可替换为 spaCy/NER 模型
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]

# 全局 embedding 模型缓存，避免重复加载
_EMBEDDING_CACHE: Dict[str, object] = {}

# embedding_model 参数 → 真实模型名称映射
_MODEL_MAP = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
}

# LightRAG 风格：领域关键词实体词典（可替换为 spaCy NER 模型）
_DOMAIN_ENTITIES: Dict[str, str] = {
    "kyc": "regulation",
    "aml": "regulation",
    "fee": "pricing",
    "credit": "lending",
    "fraud": "risk",
    "collateral": "lending",
    "chargeback": "dispute",
    "sanctions": "compliance",
    "gdpr": "privacy",
    "pii": "privacy",
}


def _detect_entities(text: str) -> List[str]:
    """
    轻量实体检测（LightRAG 实体感知思路）。
    检测文本中出现的领域关键词，返回对应实体类别列表。
    真实部署可替换为：spaCy ner / transformers NER pipeline。
    """
    tokens = text.lower().split()
    found = list({_DOMAIN_ENTITIES[tok] for tok in tokens if tok in _DOMAIN_ENTITIES})
    return found


def _load_encoder(model_name: str):
    """加载 sentence-transformers 模型（仅本地缓存，不触发网络下载）。"""
    if model_name not in _EMBEDDING_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_CACHE[model_name] = SentenceTransformer(
                model_name,
                tokenizer_kwargs={"local_files_only": True},
            )
        except Exception:
            _EMBEDDING_CACHE[model_name] = None
    return _EMBEDDING_CACHE[model_name]


def _tokenize(text: str) -> List[str]:
    """轻量分词（小写 + split），与 BM25Okapi 保持一致。"""
    return text.lower().split()


def _rrf_merge(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion（FlashRAG hybrid 核心算法）。

    公式：score(d) = Σ_i  1 / (k + rank_i(d))
    k=60 为 Cormack et al. 推荐默认值，平衡高排名文档权重。

    参数：
        ranked_lists: 多路检索结果，每路为 (chunk_idx, score) 列表（已按 score 降序）
        k: RRF 超参数（默认 60）

    返回：融合后的 (chunk_idx, rrf_score) 列表（按 rrf_score 降序）
    """
    rrf_scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    """
    统一检索接口（借鉴 FlashRAG + LightRAG 设计）。

    检索模式：
      - bm25  : BM25Okapi 稀疏检索
      - dense : sentence-transformers 稠密检索（降级为 TF-IDF）
      - hybrid: BM25 + dense 双路 RRF 融合（FlashRAG 核心策略）

    元数据增强（LightRAG 风格）：
      - 构建索引时自动检测实体，注入 metadata["entities"]
      - retrieve() 支持 metadata_filter，可按实体类别过滤候选集
    """

    def __init__(
        self,
        chunks: List[Chunk],
        retriever_type: str = "hybrid",
        embedding_model: str = "bge-small",
    ):
        self.chunks = chunks
        self.retriever_type = retriever_type
        self.embedding_model = embedding_model

        # LightRAG 风格：构建索引时注入实体元数据
        for chunk in self.chunks:
            if "entities" not in chunk.metadata:
                chunk.metadata["entities"] = _detect_entities(chunk.text)

        corpus_texts = [c.text for c in self.chunks] or [""]

        # BM25 索引（稀疏检索主力）
        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)

        # TF-IDF 向量索引（dense 降级 fallback）
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = self._tfidf.fit_transform(corpus_texts)

        # sentence-transformers 稠密索引
        self._st_model = None
        self._st_embeddings: Optional[np.ndarray] = None
        if retriever_type in ("dense", "hybrid"):
            model_name = _MODEL_MAP.get(embedding_model, embedding_model)
            self._st_model = _load_encoder(model_name)
            if self._st_model is not None:
                self._st_embeddings = self._st_model.encode(
                    corpus_texts,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

    def _ranked_bm25(self, query: str) -> List[Tuple[int, float]]:
        """BM25 排序列表：(chunk_idx, normalized_score)，按分数降序。"""
        raw = self._bm25.get_scores(_tokenize(query))
        max_score = float(np.max(raw)) if raw.max() > 0 else 1.0
        normed = [float(s) / max_score for s in raw]
        return sorted(enumerate(normed), key=lambda x: x[1], reverse=True)

    def _ranked_dense(self, query: str) -> List[Tuple[int, float]]:
        """稠密检索排序列表：优先 ST，降级 TF-IDF，按分数降序。"""
        if self._st_model is not None and self._st_embeddings is not None:
            q_emb = self._st_model.encode(
                [query], show_progress_bar=False, normalize_embeddings=True
            )
            sims = cosine_similarity(q_emb, self._st_embeddings).flatten()
        else:
            q_vec = self._tfidf.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        return sorted(enumerate(float(s) for s in sims), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedChunk]:
        """
        执行检索并返回 top_k 结果。

        hybrid 模式使用 RRF 融合（FlashRAG 策略），而非简单加权平均：
        - 对排名不敏感的分数缩放具有鲁棒性
        - 高排名文档获得更高融合权重
        """
        metadata_filter = metadata_filter or {}

        # 候选集过滤（支持 metadata_filter，LightRAG 实体过滤入口）
        valid_indices = [
            idx for idx, c in enumerate(self.chunks)
            if all(
                str(c.metadata.get(k, "")).lower().find(str(v).lower()) >= 0
                for k, v in metadata_filter.items()
            )
        ]
        valid_set = set(valid_indices)

        if self.retriever_type == "bm25":
            ranked = [(i, s) for i, s in self._ranked_bm25(query) if i in valid_set]

        elif self.retriever_type == "dense":
            ranked = [(i, s) for i, s in self._ranked_dense(query) if i in valid_set]

        else:  # hybrid — RRF 融合（FlashRAG 核心）
            bm25_ranked = [(i, s) for i, s in self._ranked_bm25(query) if i in valid_set]
            dense_ranked = [(i, s) for i, s in self._ranked_dense(query) if i in valid_set]
            ranked = _rrf_merge([bm25_ranked, dense_ranked])

        return [(self.chunks[idx], score) for idx, score in ranked[:top_k]]
