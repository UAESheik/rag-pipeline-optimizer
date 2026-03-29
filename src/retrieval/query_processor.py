from __future__ import annotations

"""
query_processor.py

借鉴 DSPy 的核心设计思想：将查询改写视为"声明式程序"，
通过 QuerySignature 描述输入/输出语义，通过 QueryModule 实现执行逻辑。

与直接使用 DSPy 框架的区别：
- 未引入 dspy 包，避免黑盒 LLM 调用与联网依赖
- 自实现 Signature / Module 模式，保持可解释性与可审计性
- 真实部署时可将 QueryModule.forward() 替换为 LLM 调用，接口不变
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── DSPy 风格：声明式 Signature ────────────────────────────────────────────────

@dataclass
class QuerySignature:
    """
    描述查询改写任务的输入/输出语义（DSPy Signature 思路）。

    Fields:
        name        : 任务名称
        description : 任务目标描述
        input_fields: 输入字段说明
        output_fields: 输出字段说明
        strategy    : 实际执行策略
    """
    name: str
    description: str
    input_fields: Dict[str, str] = field(default_factory=dict)
    output_fields: Dict[str, str] = field(default_factory=dict)
    strategy: str = "expand"


# 三种预定义 Signature（可扩展）
EXPAND_SIG = QuerySignature(
    name="QueryExpand",
    description="通过同义词/上位词扩展查询，提升稀疏检索覆盖率",
    input_fields={"query": "原始查询文本"},
    output_fields={"queries": "扩展后的查询列表（含原始查询）"},
    strategy="expand",
)

DECOMPOSE_SIG = QuerySignature(
    name="QueryDecompose",
    description="将复合问题分解为多个独立子问题，分别检索后合并",
    input_fields={"query": "复合查询文本"},
    output_fields={"queries": "子问题列表"},
    strategy="decompose",
)

HYDE_SIG = QuerySignature(
    name="QueryHyDE",
    description="生成假想文档描述（HyDE 风格），引导语义检索对齐",
    input_fields={"query": "原始查询文本"},
    output_fields={"queries": "包含 HyDE 前缀的查询列表"},
    strategy="hypothetical",
)


# ─── DSPy 风格：Module（执行逻辑与签名解耦）──────────────────────────────────────

class QueryModule:
    """
    执行查询改写的 Module（借鉴 DSPy Module 设计）。

    - forward() 对应 DSPy 的 __call__ / predict 逻辑
    - 当前为规则实现（可替换为 dspy.Predict(sig) 调用）
    - 规则透明可审计，便于面试讲解与离线运行
    """

    def __init__(self, sig: QuerySignature):
        self.sig = sig

    def forward(self, query: str) -> List[str]:
        """根据 Signature.strategy 执行对应改写逻辑。"""
        query = query.strip()
        if not query:
            return [query]

        if self.sig.strategy == "decompose":
            return self._decompose(query)
        if self.sig.strategy == "hypothetical":
            return self._hypothetical(query)
        return self._expand(query)  # 默认 expand

    # ── 内部实现（规则实现，可替换 LLM 调用）────────────────────────────────────

    @staticmethod
    def _expand(query: str) -> List[str]:
        """
        关键词扩展（AutoRAG 候选查询生成思路）。
        扩展表为显式映射，面试可展示业务知识注入位置。
        """
        _EXPANSION_MAP: Dict[str, str] = {
            "kyc": "know your customer identity verification",
            "fee": "charge cost pricing tariff",
            "document": "file record paper",
            "policy": "rule regulation guideline",
            "requirement": "criteria condition prerequisite",
            "credit": "loan lending creditworthiness",
            "fraud": "suspicious activity money laundering",
        }
        tokens = query.lower().split()
        extras = [_EXPANSION_MAP[tok] for tok in tokens if tok in _EXPANSION_MAP]
        if extras:
            expanded = query + " " + " ".join(extras)
            return [query, expanded]
        return [query]

    @staticmethod
    def _decompose(query: str) -> List[str]:
        """按连接词拆分复合问题为子问题。"""
        import re
        parts = re.split(
            r"\band\b|\bor\b|\bas well as\b|，|、",
            query,
            flags=re.IGNORECASE,
        )
        sub_queries = [p.strip() for p in parts if p.strip()]
        return sub_queries if len(sub_queries) > 1 else [query]

    @staticmethod
    def _hypothetical(query: str) -> List[str]:
        """
        HyDE：构造假想文档前缀，引导稠密检索语义对齐。
        参考 Gao et al. (2022) Precise Zero-Shot Dense Retrieval without Relevance Labels。
        """
        return [f"A document that answers the question: {query}"]


# ─── 预实例化三个模块（复用，避免重复构造）────────────────────────────────────────

_EXPAND_MODULE = QueryModule(EXPAND_SIG)
_DECOMPOSE_MODULE = QueryModule(DECOMPOSE_SIG)
_HYDE_MODULE = QueryModule(HYDE_SIG)


# ─── 向后兼容接口（optimizer.py 调用入口不变）──────────────────────────────────────

def rewrite_query(query: str, strategy: str = "expand") -> List[str]:
    """
    查询改写接口（保持向后兼容）。

    strategy 选项：
      - "expand"      : QueryExpand Signature
      - "decompose"   : QueryDecompose Signature
      - "hypothetical": QueryHyDE Signature
    """
    sig_map = {
        "expand": _EXPAND_MODULE,
        "decompose": _DECOMPOSE_MODULE,
        "hypothetical": _HYDE_MODULE,
    }
    module = sig_map.get(strategy, _EXPAND_MODULE)
    return module.forward(query)


def apply_query_processor(query: str, rewrite: bool, decompose: bool) -> List[str]:
    """统一查询处理入口：按配置决定是否改写/分解。"""
    queries = [query]
    if rewrite:
        queries = rewrite_query(query, strategy="expand")
    if decompose:
        decomposed: List[str] = []
        for q in queries:
            decomposed.extend(rewrite_query(q, strategy="decompose"))
        queries = list(dict.fromkeys(decomposed))  # 去重保序
    return queries if queries else [query]
