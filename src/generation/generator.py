from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Tuple

from src.chunking.chunker import Chunk
from src.utils.logging_utils import log_external_event


def _format_context(retrieved: List[RetrievedChunk], max_chunks: int = 6) -> str:
    parts: List[str] = []
    for chunk, score in retrieved[:max_chunks]:
        meta = chunk.metadata or {}
        meta_bits = []
        for key in ("section_title", "source", "title", "type", "page_number"):
            val = meta.get(key, "")
            if val not in ("", None):
                meta_bits.append(f"{key}={val}")
        entities = meta.get("entities", [])
        if entities:
            meta_bits.append(f"entities={','.join(map(str, entities))}")
        prefix = f"[{chunk.doc_id}|score={round(float(score),4)}"
        if meta_bits:
            prefix += "|" + "|".join(meta_bits)
        prefix += "]"
        parts.append(f"{prefix} {chunk.text[:800]}")
    return "\n\n".join(parts)

RetrievedChunk = Tuple[Chunk, float]


def _build_extract_answer(retrieved: List[RetrievedChunk], answer_style: str) -> str:
    """离线兜底：抽取式答案。"""
    evidence = retrieved[0][0].text[:500]
    if answer_style == "citation_first":
        cited = sorted({c.doc_id for c, _ in retrieved})
        cites = " ".join(f"[{d}]" for d in cited)
        return f"{cites} {evidence}"
    return evidence


def _call_ollama(messages: list, model: str, temperature: float, max_new_tokens: int) -> str:
    """调用本地 Ollama（免费开源）。"""
    base_url = os.getenv("RAG_OPT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    endpoint = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(max_new_tokens)},
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    log_external_event("generator", "ollama.chat", "start", endpoint, {"model": model})
    with urllib.request.urlopen(req, timeout=90) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    log_external_event("generator", "ollama.chat", "success", endpoint)
    return obj["message"]["content"].strip()


def generate_answer(
    query: str,
    retrieved: List[RetrievedChunk],
    answer_style: str = "concise",
    temperature: float = 0.0,
    llm_model: str = "qwen2.5:3b-instruct",
    use_llm: bool = True,
    prompt_template: str = "standard",
    max_new_tokens: int = 256,
) -> str:
    """基于检索上下文生成答案（真实调用优先，失败回退抽取式）。"""
    if not retrieved:
        return "未检索到足够证据，无法给出可靠答案。"
    if not use_llm:
        return _build_extract_answer(retrieved, answer_style)

    context = _format_context(retrieved, max_chunks=6)
    style_rule = (
        "Please answer with explicit citations like [DOC_ID] for each key claim. Prioritize evidence-first organization."
        if answer_style == "citation_first"
        else "Please answer concisely with at least one citation like [DOC_ID], and mention when evidence is weak."
    )
    prompt_template = str(prompt_template or os.getenv("RAG_OPT_PROMPT_TEMPLATE", "standard")).lower()
    system_prompt = (
        "You are a strict evidence-first RAG assistant. Answer only from context. "
        "If evidence is insufficient, explicitly say so and avoid speculation."
        if prompt_template == "strict_no_hallucination"
        else "You are a factual RAG assistant. Only answer based on provided context. If evidence is insufficient, explicitly say so."
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nContext:\n{context}\n\nInstruction:\n{style_rule}",
        },
    ]

    try:
        return _call_ollama(messages=messages, model=llm_model, temperature=temperature, max_new_tokens=max_new_tokens)
    except Exception as e:
        log_external_event("generator", "ollama.chat", "fallback", str(e), {"model": llm_model})
        return _build_extract_answer(retrieved, answer_style)
