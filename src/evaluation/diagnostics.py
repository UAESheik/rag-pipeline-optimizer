from __future__ import annotations

"""RAGChecker 风格诊断 + 真实 LLM Judge（Ollama 优先）。"""

import json
import os
import urllib.request
from typing import Dict, List, Tuple

from src.chunking.chunker import Chunk
from src.utils.logging_utils import log_external_event

RetrievedChunk = Tuple[Chunk, float]


def _token_set(text: str) -> set:
    return set(text.lower().split())


def _overlap(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def retrieval_bias_score(retrieved: List[RetrievedChunk]) -> Dict[str, object]:
    if not retrieved:
        return {"bias_detected": False, "top_doc": "", "top_doc_fraction": 0.0}
    doc_counts: Dict[str, int] = {}
    for chunk, _ in retrieved:
        doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
    top_doc = max(doc_counts, key=doc_counts.__getitem__)
    top_frac = round(doc_counts[top_doc] / len(retrieved), 3)
    return {"bias_detected": top_frac > 0.6 and len(retrieved) >= 3, "top_doc": top_doc, "top_doc_fraction": top_frac}


def hallucination_score(answer: str, retrieved: List[RetrievedChunk]) -> Dict[str, object]:
    if not answer or not retrieved:
        return {"hallucination_risk": "unknown", "unsupported_fraction": 1.0}
    ctx = " ".join(c.text for c, _ in retrieved).lower()
    tokens = [t for t in answer.lower().split() if len(t) > 3]
    if not tokens:
        return {"hallucination_risk": "low", "unsupported_fraction": 0.0}
    unsupported = sum(1 for t in tokens if t not in ctx)
    frac = round(unsupported / len(tokens), 3)
    risk = "low" if frac < 0.2 else ("medium" if frac < 0.5 else "high")
    return {"hallucination_risk": risk, "unsupported_fraction": frac}


def query_drift_score(original_query: str, answer: str) -> Dict[str, object]:
    ov = round(_overlap(original_query, answer), 3)
    return {"drift_detected": ov < 0.1, "query_answer_overlap": ov}


def _judge_prompt(query: str, answer: str, retrieved: List[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{c.doc_id}] {c.text[:700]}" for c, _ in retrieved[:6])
    return (
        "Evaluate groundedness of the answer using only the given context. "
        "Return ONLY a JSON object like {\"score\": 0.0}.\n\n"
        f"Question:\n{query}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"
    )


def _call_judge_ollama(prompt: str, model: str) -> float:
    base_url = os.getenv("RAG_OPT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    endpoint = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    log_external_event("judge", "ollama.chat", "start", endpoint, {"model": model})
    with urllib.request.urlopen(req, timeout=90) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    content = obj["message"]["content"].strip()
    parsed = json.loads(content)
    score = float(parsed["score"])
    log_external_event("judge", "ollama.chat", "success", endpoint)
    return max(0.0, min(1.0, score))


def _proxy_judge(answer: str, retrieved: List[RetrievedChunk]) -> float:
    if not answer or not retrieved:
        return 0.0
    ctx = " ".join(c.text for c, _ in retrieved).lower()
    tokens = [t for t in answer.lower().split() if len(t) > 2]
    if not tokens:
        return 1.0
    supported = sum(1 for t in tokens if t in ctx)
    return round(supported / len(tokens), 4)


def judge_groundedness_score(
    query: str,
    answer: str,
    retrieved: List[RetrievedChunk],
    use_llm_judge: bool = False,
) -> Dict[str, object]:
    if not use_llm_judge:
        return {
            "judge_score": _proxy_judge(answer, retrieved),
            "judge_method": "token_coverage_proxy",
            "judge_warning": "未启用 LLM judge，使用代理评分",
        }

    model = os.getenv("RAG_OPT_JUDGE_MODEL", os.getenv("RAG_OPT_LLM_MODEL", "qwen2.5:3b-instruct"))
    prompt = _judge_prompt(query, answer, retrieved)
    try:
        score = _call_judge_ollama(prompt=prompt, model=model)
        return {"judge_score": round(score, 4), "judge_method": "llm_judge_ollama", "judge_warning": ""}
    except Exception as e:
        proxy = _proxy_judge(answer, retrieved)
        log_external_event("judge", "ollama.chat", "fallback", str(e), {"model": model})
        return {
            "judge_score": proxy,
            "judge_method": "token_coverage_proxy_fallback",
            "judge_warning": f"LLM judge 失败，已回退代理评分: {e}",
        }


def full_ragchecker_report(
    query: str,
    answer: str,
    retrieved: List[RetrievedChunk],
    use_llm_judge: bool = False,
) -> Dict[str, object]:
    bias = retrieval_bias_score(retrieved)
    hallucination = hallucination_score(answer, retrieved)
    drift = query_drift_score(query, answer)
    judge = judge_groundedness_score(query, answer, retrieved, use_llm_judge=use_llm_judge)

    findings: List[str] = []
    if bias["bias_detected"]:
        findings.append(f"检索偏差: {bias['top_doc']} 占比 {bias['top_doc_fraction']}")
    if hallucination["hallucination_risk"] in ("medium", "high"):
        findings.append(f"幻觉风险 {hallucination['hallucination_risk']}: 无支撑 token 占比 {hallucination['unsupported_fraction']}")
    if drift["drift_detected"]:
        findings.append(f"查询偏移: 答案与查询重叠仅 {drift['query_answer_overlap']}")
    if not findings:
        findings.append("无明显问题")

    return {
        "retrieval_bias": bias,
        "hallucination": hallucination,
        "query_drift": drift,
        "judge": judge,
        "findings_summary": " | ".join(findings),
    }


def diagnose_case(case_num: int, metrics: Dict[str, float]) -> str:
    if case_num == 1:
        if metrics.get("context_recall", 0) < 0.5:
            return "检索结果不充分"
        if metrics.get("faithfulness", 0) < 0.6:
            return "生成存在幻觉风险"
        return "通过"
    if metrics.get("retrieval_coverage_proxy", 0) < 0.5:
        return "弱监督下覆盖不足"
    if metrics.get("groundedness", 0) < 0.6:
        return "答案与证据绑定不足"
    return "通过"
