from __future__ import annotations

import json
import os
import re
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


def _extract_claims(answer: str) -> List[str]:
    parts = re.split(r"[。.!?]\s+", answer.strip())
    claims = [p.strip() for p in parts if len(p.strip().split()) >= 4]
    return claims[:8]


def _citation_doc_ids(answer: str) -> List[str]:
    return [t.strip("[]") for t in answer.split() if t.startswith("[") and t.endswith("]")]


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
        return {"hallucination_risk": "unknown", "unsupported_fraction": 1.0, "claim_support": 0.0}

    context = " ".join(c.text for c, _ in retrieved).lower()
    claims = _extract_claims(answer)
    if not claims:
        return {"hallucination_risk": "low", "unsupported_fraction": 0.0, "claim_support": 1.0}

    supported = 0
    for cl in claims:
        cl_tokens = [t for t in re.findall(r"[a-zA-Z0-9_]+", cl.lower()) if len(t) > 2]
        if not cl_tokens:
            continue
        hit = sum(1 for t in cl_tokens if t in context) / len(cl_tokens)
        if hit >= 0.5:
            supported += 1

    claim_support = round(supported / max(1, len(claims)), 3)
    unsupported = round(1.0 - claim_support, 3)
    risk = "low" if unsupported < 0.25 else ("medium" if unsupported < 0.55 else "high")
    return {"hallucination_risk": risk, "unsupported_fraction": unsupported, "claim_support": claim_support}


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


def _failure_taxonomy(query: str, answer: str, retrieved: List[RetrievedChunk], report: Dict[str, object]) -> Dict[str, object]:
    retrieval_failure_type = "none"
    if not retrieved:
        retrieval_failure_type = "no_retrieval"
    elif bool(report.get("retrieval_bias", {}).get("bias_detected", False)):
        retrieval_failure_type = "retrieval_bias"
    elif all(float(score) < 0.1 for _, score in retrieved):
        retrieval_failure_type = "low_signal_retrieval"

    citation_ids = _citation_doc_ids(answer)
    retrieved_ids = {c.doc_id for c, _ in retrieved}
    missing_citations = [cid for cid in citation_ids if cid not in retrieved_ids]
    citation_failure_type = "none"
    if not citation_ids:
        citation_failure_type = "missing_citation"
    elif missing_citations:
        citation_failure_type = "citation_mismatch"
    elif len(set(citation_ids)) < len(citation_ids):
        citation_failure_type = "duplicate_citation"

    grounding_failure_type = "none"
    hall = report.get("hallucination", {})
    if hall.get("hallucination_risk") in ("medium", "high"):
        grounding_failure_type = "unsupported_claims"
    elif float(hall.get("claim_support", 1.0)) < 0.7:
        grounding_failure_type = "weak_claim_support"

    query_processing_failure_type = "none"
    if report.get("query_drift", {}).get("drift_detected", False):
        query_processing_failure_type = "query_drift"
    elif len(query.split()) <= 2 and len(answer.split()) > 40:
        query_processing_failure_type = "over_expansion"
    elif len(query.split()) > 8 and len(_extract_claims(answer)) <= 1:
        query_processing_failure_type = "under_decomposition"

    claims = _extract_claims(answer)
    citation_completeness = 0.0 if not answer else round(len(citation_ids) / max(1, len(claims)), 4)
    citation_binding = 0.0
    if citation_ids:
        citation_binding = round(len([x for x in citation_ids if x in retrieved_ids]) / len(citation_ids), 4)
    claim_coverage = 0.0
    if claims:
        claim_coverage = round(sum(1 for cl in claims if _overlap(cl, " ".join(c.text for c, _ in retrieved))) / len(claims), 4)

    return {
        "retrieval_failure_type": retrieval_failure_type,
        "grounding_failure_type": grounding_failure_type,
        "citation_failure_type": citation_failure_type,
        "query_processing_failure_type": query_processing_failure_type,
        "citation_completeness": citation_completeness,
        "citation_binding": citation_binding,
        "claim_coverage": claim_coverage,
        "retrieval_signal_strength": round(sum(float(score) for _, score in retrieved) / max(1, len(retrieved)), 4) if retrieved else 0.0,
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

    base = {
        "retrieval_bias": bias,
        "hallucination": hallucination,
        "query_drift": drift,
        "judge": judge,
    }
    taxonomy = _failure_taxonomy(query=query, answer=answer, retrieved=retrieved, report=base)

    findings: List[str] = []
    if taxonomy["retrieval_failure_type"] != "none":
        findings.append(f"retrieval={taxonomy['retrieval_failure_type']}")
    if taxonomy["grounding_failure_type"] != "none":
        findings.append(f"grounding={taxonomy['grounding_failure_type']}")
    if taxonomy["citation_failure_type"] != "none":
        findings.append(f"citation={taxonomy['citation_failure_type']}")
    if taxonomy["query_processing_failure_type"] != "none":
        findings.append(f"query_proc={taxonomy['query_processing_failure_type']}")
    if not findings:
        findings.append("no_major_issue")

    return {
        **base,
        **taxonomy,
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
