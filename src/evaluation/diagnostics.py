from __future__ import annotations

"""
diagnostics.py

RAGChecker 风格诊断模块，同时引入裁判模型（Judge）接地性检查占位。

借鉴思路：
- RAGChecker：从检索偏差、幻觉风险、查询偏移三个维度做自动化诊断
- LLM-as-Judge：judge_groundedness_score() 为裁判模型评分占位接口
  当前使用 token 覆盖率代理，真实部署可替换为 LLM 打分调用
  注意：LLM Judge 评分存在不稳定性与成本问题，使用时需附加以下谨慎说明

⚠️  裁判模型使用说明（重要）：
  1. LLM Judge 分数与人工评分相关性在 0.7-0.85 之间，不等同于黄金标准
  2. 同一 prompt 不同批次可能产生最高 ±0.1 的分数波动
  3. 本项目的 judge_score 为"代理信号"而非"真实标签"，优化目标中权重应小于真实监督指标
  4. 若用 judge_score 选择最优配置，建议同时输出 holdout 上的人工抽查结果
"""

from typing import Dict, List, Tuple

from src.chunking.chunker import Chunk

RetrievedChunk = Tuple[Chunk, float]


def _token_set(text: str) -> set:
    return set(text.lower().split())


def _overlap(a: str, b: str) -> float:
    """文本 token 交并比。"""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─── RAGChecker 风格三项诊断 ────────────────────────────────────────────────────

def retrieval_bias_score(retrieved: List[RetrievedChunk]) -> Dict[str, object]:
    """检索偏差：是否过度集中在同一文档。"""
    if not retrieved:
        return {"bias_detected": False, "top_doc": "", "top_doc_fraction": 0.0}

    doc_counts: Dict[str, int] = {}
    for chunk, _ in retrieved:
        doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

    top_doc = max(doc_counts, key=doc_counts.__getitem__)
    top_frac = round(doc_counts[top_doc] / len(retrieved), 3)
    bias_detected = top_frac > 0.6 and len(retrieved) >= 3

    return {
        "bias_detected": bias_detected,
        "top_doc": top_doc,
        "top_doc_fraction": top_frac,
    }


def hallucination_score(answer: str, retrieved: List[RetrievedChunk]) -> Dict[str, object]:
    """幻觉风险：答案中无证据 token 比例。"""
    if not answer or not retrieved:
        return {"hallucination_risk": "unknown", "unsupported_fraction": 1.0}

    ctx = " ".join(c.text for c, _ in retrieved).lower()
    tokens = [t for t in answer.lower().split() if len(t) > 3]
    if not tokens:
        return {"hallucination_risk": "low", "unsupported_fraction": 0.0}

    unsupported = sum(1 for t in tokens if t not in ctx)
    frac = round(unsupported / len(tokens), 3)

    if frac < 0.2:
        risk = "low"
    elif frac < 0.5:
        risk = "medium"
    else:
        risk = "high"

    return {"hallucination_risk": risk, "unsupported_fraction": frac}


def query_drift_score(original_query: str, answer: str) -> Dict[str, object]:
    """查询偏移：答案是否明显偏离原问题。"""
    overlap = round(_overlap(original_query, answer), 3)
    drift_detected = overlap < 0.1
    return {"drift_detected": drift_detected, "query_answer_overlap": overlap}


# ─── LLM-as-Judge 接地性评分占位 ──────────────────────────────────────────────

def judge_groundedness_score(
    query: str,
    answer: str,
    retrieved: List[RetrievedChunk],
    use_llm_judge: bool = False,
) -> Dict[str, object]:
    """
    裁判模型接地性评分（Judge Groundedness）。

    设计说明：
    - 当前实现：token 覆盖率代理评分（可离线运行，无需 LLM）
    - 生产替换：将 _proxy_judge() 替换为 LLM API 调用即可，接口不变

    ⚠️  使用注意（见模块文档）：
    - judge_score 为代理信号，不等同于真实标签
    - 在优化目标中建议权重 ≤ 0.1，避免优化器过度拟合裁判模型偏好
    - 建议对 judge_score 高但 holdout 低的配置做人工抽查

    参数：
        use_llm_judge: True 时尝试调用外部 LLM 打分（需配置 API_KEY），
                       False 时使用 token 覆盖代理（默认，离线可用）
    """
    proxy = _proxy_judge(answer, retrieved)

    if not use_llm_judge:
        return {
            "judge_score": proxy,
            "judge_method": "token_coverage_proxy",
            "judge_warning": "代理评分，非真实 LLM Judge，仅作辅助参考",
        }

    # LLM Judge 占位（真实调用替换此处）
    try:
        llm_score = _llm_judge_call(query, answer, retrieved)
        return {
            "judge_score": llm_score,
            "judge_method": "llm_judge",
            "judge_warning": "LLM Judge 存在批次间波动，建议结合 holdout 人工抽查",
        }
    except Exception:
        return {
            "judge_score": proxy,
            "judge_method": "token_coverage_proxy_fallback",
            "judge_warning": "LLM Judge 调用失败，已回退代理评分",
        }


def _proxy_judge(answer: str, retrieved: List[RetrievedChunk]) -> float:
    """token 覆盖率代理：答案中有证据支撑的 token 比例。"""
    if not answer or not retrieved:
        return 0.0
    ctx = " ".join(c.text for c, _ in retrieved).lower()
    tokens = [t for t in answer.lower().split() if len(t) > 2]
    if not tokens:
        return 1.0
    supported = sum(1 for t in tokens if t in ctx)
    return round(supported / len(tokens), 4)


def _llm_judge_call(
    query: str,
    answer: str,
    retrieved: List[RetrievedChunk],
) -> float:
    """
    LLM Judge 调用占位（生产替换位）。

    替换示例（OpenAI）：
        import openai
        ctx = "\n".join(c.text for c, _ in retrieved[:3])
        prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer: {answer}\n"
                 f"Rate the groundedness of the answer on a scale of 0 to 1."
        response = openai.chat.completions.create(...)
        return float(response.choices[0].message.content.strip())
    """
    raise NotImplementedError("LLM Judge 未配置，请替换 _llm_judge_call() 实现")


# ─── 全量诊断汇总 ──────────────────────────────────────────────────────────────

def full_ragchecker_report(
    query: str,
    answer: str,
    retrieved: List[RetrievedChunk],
    use_llm_judge: bool = False,
) -> Dict[str, object]:
    """汇总 RAGChecker 风格诊断 + Judge 接地性评分。"""
    bias = retrieval_bias_score(retrieved)
    hallucination = hallucination_score(answer, retrieved)
    drift = query_drift_score(query, answer)
    judge = judge_groundedness_score(query, answer, retrieved, use_llm_judge=use_llm_judge)

    findings: List[str] = []
    if bias["bias_detected"]:
        findings.append(f"检索偏差: {bias['top_doc']} 占比 {bias['top_doc_fraction']}")
    if hallucination["hallucination_risk"] in ("medium", "high"):
        findings.append(
            f"幻觉风险 {hallucination['hallucination_risk']}: "
            f"无支撑 token 占比 {hallucination['unsupported_fraction']}"
        )
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
    """逐查询失败原因归因。"""
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
