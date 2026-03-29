from __future__ import annotations

from typing import List, Tuple

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]


def generate_answer(query: str, retrieved: List[RetrievedChunk], answer_style: str = "concise") -> str:
    """基于检索上下文生成答案（轻量抽取式基线）。"""
    if not retrieved:
        return "未检索到足够证据，无法给出可靠答案。"

    evidence = retrieved[0][0].text[:400]
    if answer_style == "citation_first":
        cited = sorted({c.doc_id for c, _ in retrieved})
        cites = " ".join(f"[{d}]" for d in cited)
        return f"{cites} {evidence}"
    return evidence
