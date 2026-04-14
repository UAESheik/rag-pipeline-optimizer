from __future__ import annotations

"""
query_processor.py

借鉴 DSPy 的核心设计思想：将查询处理视为声明式 program，
每一步都保留结构化中间结果，便于审计、诊断与优化。
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class QuerySignature:
    name: str
    description: str
    input_fields: Dict[str, str] = field(default_factory=dict)
    output_fields: Dict[str, str] = field(default_factory=dict)
    strategy: str = "expand"


@dataclass
class QueryStepResult:
    step_name: str
    strategy: str
    input_queries: List[str]
    output_queries: List[str]
    applied: bool
    notes: str = ""


@dataclass
class QueryProgramResult:
    original_query: str
    final_queries: List[str]
    rewritten_queries: List[str] = field(default_factory=list)
    decomposed_queries: List[str] = field(default_factory=list)
    hypothetical_queries: List[str] = field(default_factory=list)
    filtered_queries: List[str] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    steps: List[QueryStepResult] = field(default_factory=list)


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

INTENT_FILTER_SIG = QuerySignature(
    name="IntentFilter",
    description="过滤过泛或偏离问题意图的查询分支",
    input_fields={"queries": "待过滤查询列表"},
    output_fields={"queries": "保留后的查询列表"},
    strategy="intent_filter",
)


class QueryModule:
    def __init__(self, sig: QuerySignature):
        self.sig = sig

    def run(self, queries: List[str]) -> QueryStepResult:
        inputs = list(queries)
        if self.sig.strategy == "decompose":
            outputs = self._decompose_many(inputs)
        elif self.sig.strategy == "hypothetical":
            outputs = self._hypothetical_many(inputs)
        elif self.sig.strategy == "intent_filter":
            outputs = self._intent_filter(inputs)
        else:
            outputs = self._expand_many(inputs)
        outputs = list(dict.fromkeys(q for q in outputs if q.strip()))
        return QueryStepResult(
            step_name=self.sig.name,
            strategy=self.sig.strategy,
            input_queries=inputs,
            output_queries=outputs or inputs,
            applied=outputs != inputs,
        )

    @staticmethod
    def _expand(query: str) -> List[str]:
        expansion_map: Dict[str, str] = {
            "kyc": "know your customer identity verification",
            "fee": "charge cost pricing tariff",
            "document": "file record paper",
            "policy": "rule regulation guideline",
            "requirement": "criteria condition prerequisite",
            "credit": "loan lending creditworthiness",
            "fraud": "suspicious activity money laundering",
            "dispute": "chargeback complaint card dispute",
        }
        tokens = query.lower().split()
        extras = [expansion_map[tok] for tok in tokens if tok in expansion_map]
        if extras:
            return [query, query + " " + " ".join(extras)]
        return [query]

    @classmethod
    def _expand_many(cls, queries: List[str]) -> List[str]:
        out: List[str] = []
        for query in queries:
            out.extend(cls._expand(query.strip()))
        return out

    @staticmethod
    def _decompose(query: str) -> List[str]:
        import re

        parts = re.split(r"\band\b|\bor\b|\bas well as\b|，|、", query, flags=re.IGNORECASE)
        sub_queries = [p.strip() for p in parts if p.strip()]
        return sub_queries if len(sub_queries) > 1 else [query]

    @classmethod
    def _decompose_many(cls, queries: List[str]) -> List[str]:
        out: List[str] = []
        for query in queries:
            out.extend(cls._decompose(query.strip()))
        return out

    @staticmethod
    def _hypothetical(query: str) -> List[str]:
        return [query, f"A document that answers the question: {query}"]

    @classmethod
    def _hypothetical_many(cls, queries: List[str]) -> List[str]:
        out: List[str] = []
        for query in queries:
            out.extend(cls._hypothetical(query.strip()))
        return out

    @staticmethod
    def _intent_filter(queries: List[str]) -> List[str]:
        kept: List[str] = []
        for query in queries:
            q = query.strip()
            if len(q.split()) < 2:
                continue
            if q.lower().startswith("a document that answers") and len(q.split()) < 8:
                continue
            kept.append(q)
        return kept or queries


_EXPAND_MODULE = QueryModule(EXPAND_SIG)
_DECOMPOSE_MODULE = QueryModule(DECOMPOSE_SIG)
_HYDE_MODULE = QueryModule(HYDE_SIG)
_INTENT_FILTER_MODULE = QueryModule(INTENT_FILTER_SIG)


def run_query_program(
    query: str,
    rewrite: bool,
    decompose: bool,
    use_hyde: bool = False,
    intent_driven_filter: bool = False,
) -> QueryProgramResult:
    result = QueryProgramResult(original_query=query, final_queries=[query])
    current_queries = [query]

    if rewrite:
        step = _EXPAND_MODULE.run(current_queries)
        result.steps.append(step)
        result.execution_path.append(step.step_name)
        result.rewritten_queries = [q for q in step.output_queries if q != query]
        current_queries = step.output_queries

    if decompose:
        step = _DECOMPOSE_MODULE.run(current_queries)
        result.steps.append(step)
        result.execution_path.append(step.step_name)
        result.decomposed_queries = [q for q in step.output_queries if q not in result.rewritten_queries and q != query]
        current_queries = step.output_queries

    if use_hyde:
        step = _HYDE_MODULE.run(current_queries)
        result.steps.append(step)
        result.execution_path.append(step.step_name)
        result.hypothetical_queries = [q for q in step.output_queries if q.lower().startswith("a document that answers")]
        current_queries = step.output_queries

    if intent_driven_filter:
        step = _INTENT_FILTER_MODULE.run(current_queries)
        result.steps.append(step)
        result.execution_path.append(step.step_name)
        result.filtered_queries = step.output_queries
        current_queries = step.output_queries

    result.final_queries = list(dict.fromkeys(q for q in current_queries if q.strip())) or [query]
    return result


def rewrite_query(query: str, strategy: str = "expand") -> List[str]:
    if strategy == "decompose":
        return _DECOMPOSE_MODULE.run([query]).output_queries
    if strategy == "hypothetical":
        return _HYDE_MODULE.run([query]).output_queries
    return _EXPAND_MODULE.run([query]).output_queries


def apply_query_processor(query: str, rewrite: bool, decompose: bool) -> List[str]:
    return run_query_program(query, rewrite=rewrite, decompose=decompose).final_queries
