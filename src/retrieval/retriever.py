from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]


@dataclass
class RetrievalProvenance:
    source: str
    bm25_rank: Optional[int] = None
    bm25_score: float = 0.0
    dense_rank: Optional[int] = None
    dense_score: float = 0.0
    rrf_score: float = 0.0
    metadata_bonus: float = 0.0
    matched_entities: List[str] = field(default_factory=list)
    matched_fields: List[str] = field(default_factory=list)


@dataclass
class RetrievalStageResult:
    stage: str
    candidates: List[Tuple[int, float]]
    source: str


@dataclass
class RetrievedItem:
    chunk: Chunk
    score: float
    provenance: RetrievalProvenance


_EMBEDDING_CACHE: Dict[str, object] = {}
_MODEL_MAP = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
}

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
    tokens = text.lower().split()
    return list({_DOMAIN_ENTITIES[tok] for tok in tokens if tok in _DOMAIN_ENTITIES})


def _load_encoder(model_name: str):
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
    return text.lower().split()


def _rrf_merge(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    rrf_scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(
        self,
        chunks: List[Chunk],
        retriever_type: str = "hybrid",
        embedding_model: str = "bge-small",
        metadata_enrichment: bool = True,
    ):
        self.chunks = chunks
        self.retriever_type = retriever_type
        self.embedding_model = embedding_model
        self.metadata_enrichment = metadata_enrichment

        if self.metadata_enrichment:
            for chunk in self.chunks:
                chunk.metadata.setdefault("entities", _detect_entities(chunk.text))

        corpus_texts = [c.text for c in self.chunks] or [""]
        self._bm25 = BM25Okapi([_tokenize(t) for t in corpus_texts])
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = self._tfidf.fit_transform(corpus_texts)

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
        raw = self._bm25.get_scores(_tokenize(query))
        max_score = float(np.max(raw)) if raw.max() > 0 else 1.0
        normed = [float(s) / max_score for s in raw]
        return sorted(enumerate(normed), key=lambda x: x[1], reverse=True)

    def _ranked_dense(self, query: str) -> List[Tuple[int, float]]:
        if self._st_model is not None and self._st_embeddings is not None:
            q_emb = self._st_model.encode([query], show_progress_bar=False, normalize_embeddings=True)
            sims = cosine_similarity(q_emb, self._st_embeddings).flatten()
        else:
            q_vec = self._tfidf.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        return sorted(enumerate(float(s) for s in sims), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _metadata_bonus(query: str, chunk: Chunk) -> Tuple[float, List[str], List[str]]:
        q = query.lower()
        bonus = 0.0
        matched_entities: List[str] = []
        matched_fields: List[str] = []

        entities = [str(x).lower() for x in chunk.metadata.get("entities", [])]
        overlap_entities = [e for e in entities if e and e in q]
        if overlap_entities:
            bonus += 0.08 * min(2, len(overlap_entities))
            matched_entities = overlap_entities
            matched_fields.append("entities")

        for field in ("section_title", "source", "title", "doc_title"):
            val = str(chunk.metadata.get(field, "")).strip().lower()
            if val and val in q:
                bonus += 0.05
                matched_fields.append(field)

        return min(0.2, bonus), matched_entities, list(dict.fromkeys(matched_fields))

    def _generate_candidates(self, query: str, metadata_filter: Optional[Dict[str, str]] = None) -> Dict[str, List[Tuple[int, float]]]:
        metadata_filter = metadata_filter or {}
        valid_indices = [
            idx
            for idx, c in enumerate(self.chunks)
            if all(str(c.metadata.get(k, "")).lower().find(str(v).lower()) >= 0 for k, v in metadata_filter.items())
        ]
        valid_set = set(valid_indices)
        bm25_ranked = [(i, s) for i, s in self._ranked_bm25(query) if i in valid_set]
        dense_ranked = [(i, s) for i, s in self._ranked_dense(query) if i in valid_set]
        return {"bm25": bm25_ranked, "dense": dense_ranked}

    def _fuse_candidates(self, candidate_map: Dict[str, List[Tuple[int, float]]]) -> Tuple[List[Tuple[int, float]], str]:
        bm25_ranked = candidate_map.get("bm25", [])
        dense_ranked = candidate_map.get("dense", [])
        if self.retriever_type == "bm25":
            return bm25_ranked, "bm25"
        if self.retriever_type == "dense":
            return dense_ranked, "dense"
        return _rrf_merge([bm25_ranked, dense_ranked]), "hybrid_rrf"

    def _apply_metadata_scoring(self, query: str, fused: List[Tuple[int, float]], source: str) -> List[RetrievedItem]:
        candidate_map = {"bm25": self._ranked_bm25(query), "dense": self._ranked_dense(query)}
        bm25_pos = {idx: (r + 1, score) for r, (idx, score) in enumerate(candidate_map["bm25"])}
        dense_pos = {idx: (r + 1, score) for r, (idx, score) in enumerate(candidate_map["dense"])}
        items: List[RetrievedItem] = []
        for idx, base_score in fused:
            chunk = self.chunks[idx]
            bonus, matched_entities, matched_fields = self._metadata_bonus(query, chunk)
            final_score = float(base_score) + float(bonus)
            bp = bm25_pos.get(idx, (None, 0.0))
            dp = dense_pos.get(idx, (None, 0.0))
            items.append(
                RetrievedItem(
                    chunk=chunk,
                    score=final_score,
                    provenance=RetrievalProvenance(
                        source=source,
                        bm25_rank=bp[0],
                        bm25_score=float(bp[1]),
                        dense_rank=dp[0],
                        dense_score=float(dp[1]),
                        rrf_score=float(base_score) if source == "hybrid_rrf" else 0.0,
                        metadata_bonus=float(bonus),
                        matched_entities=matched_entities,
                        matched_fields=matched_fields,
                    ),
                )
            )
        return sorted(items, key=lambda x: x.score, reverse=True)

    def retrieve_with_provenance(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedItem]:
        candidate_map = self._generate_candidates(query, metadata_filter)
        fused, source = self._fuse_candidates(candidate_map)
        items = self._apply_metadata_scoring(query, fused, source)
        return items[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedChunk]:
        items = self.retrieve_with_provenance(query=query, top_k=top_k, metadata_filter=metadata_filter)
        return [(it.chunk, it.score) for it in items]
