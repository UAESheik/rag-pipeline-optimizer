from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from src.chunking.chunker import chunk_document
from src.evaluation.diagnostics import diagnose_case, full_ragchecker_report
from src.evaluation.metrics import case1_metrics, case2_metrics
from src.generation.generator import generate_answer
from src.retrieval.query_processor import run_query_program
from src.retrieval.reranker import rerank
from src.retrieval.retriever import Retriever
from src.report.report_generator import generate_final_report
from src.utils.cache import SimpleCache
from src.utils.io import read_csv, read_json, read_jsonl, write_csv, write_json


class RAGOptimizer:
    def __init__(self, config_path: Path, data_dir: Path, output_dir: Path, top_k: int = 5):
        self.config = read_json(config_path)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.top_k = top_k
        cache_enabled = bool(self.config.get("run", {}).get("cache_enabled", False))
        self.cache = SimpleCache(output_dir / ".cache", enabled=cache_enabled)
        if not cache_enabled:
            cache_dir = output_dir / ".cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        self.corpus = read_jsonl(data_dir / "reference_corpus.jsonl")
        self.case1 = read_csv(data_dir / "case1_eval_dataset.csv")
        self.case2 = read_csv(data_dir / "case2_query_doc_dataset.csv")
        self.case_results: Dict[int, Dict[str, Any]] = {}

    def _iter_configs(self, max_trials: int) -> List[Dict[str, Any]]:
        """生成候选配置列表（grid）。"""
        ss = self.config["search_space"]

        primary = list(itertools.product(
            ss["retrieval"]["retriever"],
            ss["chunking"]["strategy"],
            ss["chunking"]["size"],
            ss["generation"]["answer_style"],
            ss["reranking"]["enabled"],
        ))

        from itertools import cycle as _cycle
        emb_cyc = _cycle(ss["retrieval"]["embedding_model"])
        mfil_cyc = _cycle(ss["retrieval"]["metadata_filter_enabled"])
        menr_cyc = _cycle(ss["retrieval"].get("metadata_enrichment", [True]))
        rew_cyc = _cycle(ss["query_processor"]["rewrite"])
        dec_cyc = _cycle(ss["query_processor"]["decompose"])
        hyd_cyc = _cycle(ss["query_processor"].get("use_hyde", [False]))
        idf_cyc = _cycle(ss["query_processor"].get("intent_driven_filter", [False]))
        ovt_cyc = _cycle(ss["chunking"]["overlap_type"])
        ovs_cyc = _cycle(ss["chunking"]["overlap_size"])
        temp_cyc = _cycle(ss["generation"].get("temperature", [0.0]))
        prompt_cyc = _cycle(ss["generation"].get("prompt_template", ["standard"]))
        maxtok_cyc = _cycle(ss["generation"].get("max_new_tokens", [256]))
        rm_cyc = _cycle(ss["reranking"]["model"])
        rtop_cyc = _cycle(ss["reranking"].get("rerank_top_k", [self.top_k]))

        out: List[Dict[str, Any]] = []
        for idx, (retriever, strategy, size, rerank_style, rerank_en) in enumerate(primary):
            out.append({
                "config_id": f"cfg_{idx:05d}",
                "retrieval": {
                    "retriever": retriever,
                    "embedding_model": next(emb_cyc),
                    "metadata_filter_enabled": next(mfil_cyc),
                    "metadata_enrichment": next(menr_cyc),
                    "metadata_filter_fields": ss["retrieval"].get("metadata_filter_fields", ["section_title", "entities", "type"]),
                },
                "chunking": {
                    "strategy": strategy,
                    "size": size,
                    "overlap_type": next(ovt_cyc),
                    "overlap_size": next(ovs_cyc),
                    "semantic_min_size": ss["chunking"]["semantic_min_size"][0],
                    "semantic_max_size": ss["chunking"]["semantic_max_size"][0],
                    "window_size": ss["chunking"].get("window_size", [3])[0],
                    "similarity_threshold": ss["chunking"].get("similarity_threshold", [0.65])[0],
                    "preserve_table_as_markdown": True,
                    "generate_image_caption": False,
                },
                "generation": {
                    "answer_style": rerank_style,
                    "temperature": next(temp_cyc),
                    "prompt_template": next(prompt_cyc),
                    "max_new_tokens": next(maxtok_cyc),
                },
                "reranking": {
                    "enabled": rerank_en,
                    "model": next(rm_cyc),
                    "rerank_top_k": next(rtop_cyc),
                },
                "query_processor": {
                    "rewrite": next(rew_cyc),
                    "decompose": next(dec_cyc),
                    "use_hyde": next(hyd_cyc),
                    "intent_driven_filter": next(idf_cyc),
                },
            })
            if len(out) >= max_trials:
                break
        return out

    def _sample_config_by_trial(self, trial: Any) -> Dict[str, Any]:
        """贝叶斯搜索时由 Optuna 采样一组配置。"""
        ss = self.config["search_space"]
        active_case = getattr(self, "_active_case_num", None)

        retriever_choices = ss["retrieval"]["retriever"]
        rerank_enabled_choices = ss["reranking"]["enabled"]
        rerank_top_k_choices = ss["reranking"].get("rerank_top_k", [self.top_k])
        answer_style_choices = ss["generation"]["answer_style"]
        prompt_template_choices = ss["generation"].get("prompt_template", ["standard"])
        temperature_choices = ss["generation"].get("temperature", [0.0])
        max_new_tokens_choices = ss["generation"].get("max_new_tokens", [256])

        if active_case == 2:
            retriever_choices = [x for x in retriever_choices if x in ("bm25", "hybrid")] or ["bm25"]
            rerank_enabled_choices = [False]
            rerank_top_k_choices = [self.top_k]
            answer_style_choices = ["citation_first", "concise_with_evidence"]
            prompt_template_choices = ["strict_no_hallucination"]
            temperature_choices = [0.0]
            max_new_tokens_choices = [128]

        return {
            "config_id": f"cfg_{trial.number:05d}",
            "chunking": {
                "strategy": trial.suggest_categorical("chunking_strategy", ss["chunking"]["strategy"]),
                "size": trial.suggest_categorical("chunking_size", ss["chunking"]["size"]),
                "overlap_type": trial.suggest_categorical("chunking_overlap_type", ss["chunking"]["overlap_type"]),
                "overlap_size": trial.suggest_categorical("chunking_overlap_size", ss["chunking"]["overlap_size"]),
                "semantic_min_size": trial.suggest_categorical("chunking_semantic_min_size", ss["chunking"]["semantic_min_size"]),
                "semantic_max_size": trial.suggest_categorical("chunking_semantic_max_size", ss["chunking"]["semantic_max_size"]),
                "window_size": trial.suggest_categorical("chunking_window_size", ss["chunking"].get("window_size", [3])),
                "similarity_threshold": trial.suggest_categorical("chunking_similarity_threshold", ss["chunking"].get("similarity_threshold", [0.65])),
                "preserve_table_as_markdown": trial.suggest_categorical("chunking_preserve_table_as_markdown", ss["chunking"].get("preserve_table_as_markdown", [True])),
                "generate_image_caption": trial.suggest_categorical("chunking_generate_image_caption", ss["chunking"].get("generate_image_caption", [False])),
            },
            "retrieval": {
                "retriever": trial.suggest_categorical("retrieval_retriever", retriever_choices),
                "embedding_model": trial.suggest_categorical("retrieval_embedding_model", ss["retrieval"]["embedding_model"]),
                "metadata_filter_enabled": trial.suggest_categorical("retrieval_metadata_filter_enabled", ss["retrieval"]["metadata_filter_enabled"]),
                "metadata_enrichment": trial.suggest_categorical("retrieval_metadata_enrichment", ss["retrieval"].get("metadata_enrichment", [True])),
                "metadata_filter_fields": ss["retrieval"].get("metadata_filter_fields", ["section_title", "entities", "type"]),
            },
            "query_processor": {
                "rewrite": trial.suggest_categorical("query_rewrite", ss["query_processor"]["rewrite"]),
                "decompose": trial.suggest_categorical("query_decompose", ss["query_processor"]["decompose"]),
                "use_hyde": trial.suggest_categorical("query_use_hyde", ss["query_processor"].get("use_hyde", [False])),
                "intent_driven_filter": trial.suggest_categorical("query_intent_driven_filter", ss["query_processor"].get("intent_driven_filter", [False])),
            },
            "generation": {
                "answer_style": trial.suggest_categorical("generation_answer_style", answer_style_choices),
                "temperature": trial.suggest_categorical("generation_temperature", temperature_choices),
                "prompt_template": trial.suggest_categorical("generation_prompt_template", prompt_template_choices),
                "max_new_tokens": trial.suggest_categorical("generation_max_new_tokens", max_new_tokens_choices),
            },
            "reranking": {
                "enabled": trial.suggest_categorical("reranking_enabled", rerank_enabled_choices),
                "model": trial.suggest_categorical("reranking_model", ss["reranking"]["model"]),
                "rerank_top_k": trial.suggest_categorical("reranking_rerank_top_k", rerank_top_k_choices),
            },
        }

    def _sample_random_config(self, trial_idx: int, seed: int = 42) -> Dict[str, Any]:
        """随机采样一组配置（random search 模式）。

        每个超参数从其候选列表中独立均匀采样，确保 20 个 trial 内
        就能覆盖到不同 retriever / chunk_strategy / chunk_size 组合，
        不受网格截断影响。
        seed 与 trial_idx 组合保证采样可复现。
        """
        import random as _random
        rng = _random.Random(seed + trial_idx)

        def _pick(lst):
            return rng.choice(lst)

        ss = self.config["search_space"]
        retriever = _pick(ss["retrieval"]["retriever"])
        strategy  = _pick(ss["chunking"]["strategy"])
        size      = _pick(ss["chunking"]["size"])

        return {
            "config_id": f"cfg_{trial_idx:05d}",
            "retrieval": {
                "retriever": retriever,
                "embedding_model": _pick(ss["retrieval"]["embedding_model"]),
                "metadata_filter_enabled": _pick(ss["retrieval"]["metadata_filter_enabled"]),
                "metadata_enrichment": _pick(ss["retrieval"].get("metadata_enrichment", [True])),
                "metadata_filter_fields": ss["retrieval"].get(
                    "metadata_filter_fields", ["section_title", "entities", "type"]
                ),
            },
            "chunking": {
                "strategy": strategy,
                "size": size,
                "overlap_type": _pick(ss["chunking"]["overlap_type"]),
                "overlap_size": _pick(ss["chunking"]["overlap_size"]),
                "semantic_min_size": ss["chunking"]["semantic_min_size"][0],
                "semantic_max_size": ss["chunking"]["semantic_max_size"][0],
                "window_size": _pick(ss["chunking"].get("window_size", [3])),
                "similarity_threshold": _pick(ss["chunking"].get("similarity_threshold", [0.65])),
                "preserve_table_as_markdown": _pick(ss["chunking"].get("preserve_table_as_markdown", [True])),
                "generate_image_caption": _pick(ss["chunking"].get("generate_image_caption", [False])),
            },
            "generation": {
                "answer_style": _pick(ss["generation"]["answer_style"]),
                "temperature": _pick(ss["generation"].get("temperature", [0.0])),
                "prompt_template": _pick(ss["generation"].get("prompt_template", ["standard"])),
                "max_new_tokens": _pick(ss["generation"].get("max_new_tokens", [256])),
            },
            "reranking": {
                "enabled": _pick(ss["reranking"]["enabled"]),
                "model": _pick(ss["reranking"]["model"]),
                "rerank_top_k": _pick(ss["reranking"].get("rerank_top_k", [self.top_k])),
            },
            "query_processor": {
                "rewrite": _pick(ss["query_processor"]["rewrite"]),
                "decompose": _pick(ss["query_processor"]["decompose"]),
                "use_hyde": _pick(ss["query_processor"].get("use_hyde", [False])),
                "intent_driven_filter": _pick(ss["query_processor"].get("intent_driven_filter", [False])),
            },
        }

    def _build_chunks(self, cfg: Dict[str, Any]):
        chunks = []
        for d in self.corpus:
            base_metadata = {}
            if bool(cfg["retrieval"].get("metadata_enrichment", False)):
                base_metadata = {
                    "title": d.get("title", ""),
                    "source": d.get("source", ""),
                    "page_number": d.get("page_number", None),
                }
            chunks.extend(
                chunk_document(
                    doc_id=d["doc_id"],
                    text=d["text"],
                    strategy=cfg["chunking"]["strategy"],
                    size=cfg["chunking"]["size"],
                    overlap_type=cfg["chunking"]["overlap_type"],
                    overlap_size=cfg["chunking"]["overlap_size"],
                    semantic_min_size=cfg["chunking"]["semantic_min_size"],
                    semantic_max_size=cfg["chunking"]["semantic_max_size"],
                    preserve_table_as_markdown=bool(cfg["chunking"].get("preserve_table_as_markdown", True)),
                    generate_image_caption=bool(cfg["chunking"].get("generate_image_caption", False)),
                    window_size=int(cfg["chunking"].get("window_size", 3)),
                    similarity_threshold=float(cfg["chunking"].get("similarity_threshold", 0.65)),
                    base_metadata=base_metadata,
                )
            )
        return chunks

    def _build_metadata_filter(self, query: str, cfg: Dict[str, Any], chunks: List[Any]) -> Dict[str, str]:
        """基于配置字段从 query 中自动匹配 metadata 过滤条件。"""
        if not cfg["retrieval"].get("metadata_filter_enabled", False):
            return {}
        if not cfg["retrieval"].get("metadata_enrichment", False):
            return {}

        fields = cfg["retrieval"].get("metadata_filter_fields", [])
        q = query.lower()
        out: Dict[str, str] = {}

        for field in fields:
            values = {
                str(c.metadata.get(field, "")).strip()
                for c in chunks
                if str(c.metadata.get(field, "")).strip()
            }
            # 候选值按长度倒序，优先更具体短语
            for val in sorted(values, key=len, reverse=True):
                if val.lower() in q:
                    out[field] = val
                    break
        return out

    def _split_dataset(self, dataset: list, holdout_ratio: float = 0.3, seed: int = 42) -> tuple:
        """将数据集按比例分为训练集和保留集，用于防止优化过拟合。"""
        import random
        rng = random.Random(seed)
        data = list(dataset)
        rng.shuffle(data)
        n_holdout = max(1, int(len(data) * holdout_ratio))
        holdout = data[:n_holdout]
        train = data[n_holdout:] if len(data) > n_holdout else data
        return train, holdout

    def _estimate_external_signal_availability(self) -> Dict[str, bool]:
        """估计外部评估链路是否可用（依赖+本地服务连通性）。"""
        out = {"ragas": False, "bertscore": False, "llm_judge": False, "llm_generator": False}
        try:
            import ragas  # noqa: F401
            out["ragas"] = True
        except Exception:
            out["ragas"] = False

        try:
            import bert_score  # noqa: F401
            out["bertscore"] = True
        except Exception:
            out["bertscore"] = False

        ollama_ok = False
        try:
            base_url = os.getenv("RAG_OPT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                ollama_ok = True
        except Exception:
            ollama_ok = False

        out["llm_judge"] = bool(self.config.get("run", {}).get("use_llm_judge", False)) and ollama_ok
        out["llm_generator"] = bool(self.config.get("run", {}).get("use_llm_generator", False)) and ollama_ok
        return out

    @staticmethod
    def _spearman_corr(xs: List[float], ys: List[float]) -> float:
        n = min(len(xs), len(ys))
        if n < 2:
            return 0.0

        def _ranks(vals: List[float]) -> List[float]:
            pairs = sorted(enumerate(vals), key=lambda t: t[1])
            ranks = [0.0] * len(vals)
            i = 0
            while i < len(pairs):
                j = i
                while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[pairs[k][0]] = avg_rank
                i = j + 1
            return ranks

        rx = _ranks(xs[:n])
        ry = _ranks(ys[:n])
        mx = sum(rx) / n
        my = sum(ry) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        vx = sum((a - mx) ** 2 for a in rx)
        vy = sum((b - my) ** 2 for b in ry)
        if vx <= 0 or vy <= 0:
            return 0.0
        return float(max(-1.0, min(1.0, cov / math.sqrt(vx * vy))))

    def _write_metric_sanity_check(self, out_dir: Path, case_num: int, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        cols = [
            "composite", "context_recall", "answer_similarity", "faithfulness",
            "retrieval_coverage_proxy", "groundedness", "citation_quality",
            "answer_relevancy", "context_relevancy", "judge_score",
        ]
        composite_vals = [float(r.get("composite", 0.0)) for r in rows]
        table: List[Dict[str, Any]] = []
        for c in cols:
            metric_vals = [float(r.get(c, 0.0)) for r in rows]
            table.append({
                "case_num": case_num,
                "metric": c,
                "spearman_vs_composite": round(self._spearman_corr(metric_vals, composite_vals), 4),
                "mean": round(sum(metric_vals) / max(1, len(metric_vals)), 4),
                "min": round(min(metric_vals), 4),
                "max": round(max(metric_vals), 4),
            })

        write_csv(
            out_dir / f"metric_sanity_case{case_num}.csv",
            table,
            ["case_num", "metric", "spearman_vs_composite", "mean", "min", "max"],
        )

    def _run_case(self, case_num: int, cfg: Dict[str, Any], dataset_override: list | None = None) -> Dict[str, Any]:
        dataset = dataset_override if dataset_override is not None else (self.case1 if case_num == 1 else self.case2)
        # Case2 定向策略：固定生成与检索配置，降低幻觉噪声
        effective_cfg = json.loads(json.dumps(cfg))
        effective_top_k = self.top_k
        if case_num == 2:
            effective_cfg.setdefault("generation", {})["answer_style"] = "citation_first"
            effective_cfg["generation"]["temperature"] = 0.0
            if effective_cfg.get("retrieval", {}).get("retriever") not in ("bm25", "hybrid"):
                effective_cfg["retrieval"]["retriever"] = "bm25"
            effective_cfg.setdefault("reranking", {})["enabled"] = False
            effective_top_k = min(self.top_k, 6)
        cfg = effective_cfg

        chunks = self._build_chunks(cfg)
        retriever = Retriever(
            chunks,
            retriever_type=cfg["retrieval"]["retriever"],
            embedding_model=cfg["retrieval"].get("embedding_model", "tfidf"),
            metadata_enrichment=bool(cfg["retrieval"].get("metadata_enrichment", False)),
        )

        per_query: List[Dict[str, Any]] = []
        ragchecker_findings: List[str] = []

        for row in dataset:
            qid = row["query_id"]
            query = row["query"]
            reference_ids = [x for x in row["reference_doc_ids"].split("|") if x]

            cache_key = self.cache.make_key({"case": case_num, "qid": qid, "cfg": cfg})
            cached = self.cache.get(cache_key)
            if cached:
                per_query.append(cached)
                continue

            metadata_filter = self._build_metadata_filter(query=query, cfg=cfg, chunks=chunks)

            # 查询改写/分解：对多个子查询分别检索后合并去重
            qp = run_query_program(
                query=query,
                rewrite=cfg["query_processor"].get("rewrite", False),
                decompose=cfg["query_processor"].get("decompose", False),
                use_hyde=cfg["query_processor"].get("use_hyde", False),
                intent_driven_filter=cfg["query_processor"].get("intent_driven_filter", False),
            )
            sub_queries = qp.final_queries

            merged_items = retriever.retrieve_with_provenance(query, top_k=effective_top_k, metadata_filter=metadata_filter)
            if len(sub_queries) > 1:
                seen_ids: set = {it.chunk.chunk_id for it in merged_items}
                for sq in sub_queries[1:]:
                    for it in retriever.retrieve_with_provenance(sq, top_k=effective_top_k, metadata_filter=metadata_filter):
                        if it.chunk.chunk_id not in seen_ids:
                            merged_items.append(it)
                            seen_ids.add(it.chunk.chunk_id)

            merged_items = sorted(merged_items, key=lambda x: x.score, reverse=True)[:effective_top_k]
            retrieved = [(it.chunk, it.score) for it in merged_items]
            retrieved = rerank(
                retrieved,
                enabled=cfg["reranking"]["enabled"],
                query=query,
                model_name=cfg["reranking"].get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_k=int(cfg["reranking"].get("rerank_top_k", self.top_k)),
            )
            answer = generate_answer(
                query=query,
                retrieved=retrieved,
                answer_style=cfg["generation"]["answer_style"],
                temperature=cfg["generation"].get("temperature", 0.0),
                llm_model=str(self.config.get("run", {}).get("llm_model", "qwen2.5:3b-instruct")),
                use_llm=bool(self.config.get("run", {}).get("use_llm_generator", True)),
                prompt_template=str(cfg["generation"].get("prompt_template", "standard")),
                max_new_tokens=int(cfg["generation"].get("max_new_tokens", 256)),
            )
            if case_num == 2:
                tokens = answer.split()
                if len(tokens) > 120:
                    answer = " ".join(tokens[:120])
                if "[" not in answer and retrieved:
                    answer = f"{answer} [{retrieved[0][0].doc_id}]"
            use_bertscore = bool(self.config.get("run", {}).get("use_bertscore", False))
            case1_w = self.config.get(
                "objective", {}
            ).get("case1_weights", {
                "context_recall": 0.35,
                "answer_similarity": 0.35,
                "faithfulness": 0.2,
                "answer_relevancy": 0.05,
                "context_relevancy": 0.05,
            })
            case2_w = self.config.get(
                "objective", {}
            ).get("case2_weights", {
                "retrieval_coverage_proxy": 0.4,
                "groundedness": 0.3,
                "citation_quality": 0.2,
                "answer_relevancy": 0.05,
                "context_relevancy": 0.05,
            })

            if case_num == 1:
                ref_ctx = row.get("reference_relevant_context", "")
                m = case1_metrics(
                    retrieved=retrieved,
                    answer=answer,
                    reference_answer=row.get("reference_answer", ""),
                    reference_doc_ids=reference_ids,
                    reference_context=ref_ctx,
                    query=query,
                    use_bertscore=use_bertscore,
                    weights=case1_w,
                )
            else:
                m = case2_metrics(
                    retrieved=retrieved,
                    answer=answer,
                    reference_doc_ids=reference_ids,
                    query=query,
                    weights=case2_w,
                )

            # RAGChecker 风格诊断 + Judge 接地性评分
            use_llm_judge = bool(self.config.get("run", {}).get("use_llm_judge", False))
            rc = full_ragchecker_report(query=query, answer=answer, retrieved=retrieved, use_llm_judge=use_llm_judge)
            ragchecker_findings.append(f"{qid}: {rc['findings_summary']}")
            judge_score = rc.get("judge", {}).get("judge_score", 0.0)
            judge_method = rc.get("judge", {}).get("judge_method", "token_coverage_proxy")
            judge_warning = rc.get("judge", {}).get("judge_warning", "")

            judge_weight = float(self.config.get("objective", {}).get("judge_weight", 0.03))
            m["composite_no_judge"] = float(m.get("composite", 0.0))
            m["judge_weight"] = judge_weight
            m["composite"] = round((1.0 - judge_weight) * float(m.get("composite", 0.0)) + judge_weight * float(judge_score), 4)

            rewrite_used = 1 if qp.rewritten_queries else 0
            decompose_used = len(qp.decomposed_queries)
            hyde_used = 1 if qp.hypothetical_queries else 0
            metadata_bonus_avg = round(
                sum(float(it.provenance.metadata_bonus) for it in merged_items) / max(1, len(merged_items)), 4
            ) if merged_items else 0.0
            entity_match_rate = round(
                sum(1 for it in merged_items if it.provenance.matched_entities) / max(1, len(merged_items)), 4
            ) if merged_items else 0.0
            rrf_fusion_gain = round(
                sum(float(it.provenance.rrf_score) for it in merged_items) / max(1, len(merged_items)), 4
            ) if merged_items else 0.0
            metadata_filter_hit_rate = 1.0 if metadata_filter else 0.0
            query_program_depth = len(qp.steps)

            result: Dict[str, Any] = {
                "config_id": cfg["config_id"],
                "query_id": qid,
                "query": query,
                "retrieved_doc_ids": "|".join([c.doc_id for c, _ in retrieved]),
                "answer": answer,
                **m,
                "failure_reason": diagnose_case(case_num, m),
                "ragchecker_findings": rc["findings_summary"],
                "hallucination_risk": rc["hallucination"]["hallucination_risk"],
                "retrieval_bias": str(rc["retrieval_bias"]["bias_detected"]),
                "query_drift": str(rc["query_drift"]["drift_detected"]),
                "judge_score": judge_score,
                "judge_method": judge_method,
                "judge_warning": judge_warning,
                "retrieval_failure_type": rc.get("retrieval_failure_type", "none"),
                "grounding_failure_type": rc.get("grounding_failure_type", "none"),
                "citation_failure_type": rc.get("citation_failure_type", "none"),
                "query_processing_failure_type": rc.get("query_processing_failure_type", "none"),
                "citation_completeness": rc.get("citation_completeness", 0.0),
                "citation_binding": rc.get("citation_binding", 0.0),
                "query_program_final_queries": " || ".join(qp.final_queries),
                "query_program_execution_path": " > ".join(qp.execution_path),
                "query_program_depth": query_program_depth,
                "query_rewrite_usage": rewrite_used,
                "decompose_branch_count": decompose_used,
                "hyde_usage": hyde_used,
                "metadata_filter_hit_rate": metadata_filter_hit_rate,
                "metadata_bonus_avg": metadata_bonus_avg,
                "entity_match_rate": entity_match_rate,
                "rrf_fusion_gain": rrf_fusion_gain,
            }
            self.cache.set(cache_key, result)
            per_query.append(result)

        mean_composite = round(
            sum(float(r["composite"]) for r in per_query) / max(1, len(per_query)), 4
        )
        return {
            "mean_composite": mean_composite,
            "per_query": per_query,
            "ragchecker_findings": ragchecker_findings,
        }

    def optimize(self, case_num: int, max_trials: int) -> None:
        out_dir = self.output_dir
        (out_dir / "retrieval_examples").mkdir(parents=True, exist_ok=True)
        (out_dir / "answer_examples").mkdir(parents=True, exist_ok=True)

        # 将数据集划分为训练集（优化用）和保留集（最终评估用，防过拟合）
        seed = self.config.get("run", {}).get("random_seed", 42)
        full_dataset = self.case1 if case_num == 1 else self.case2
        train_set, holdout_set = self._split_dataset(full_dataset, holdout_ratio=0.3, seed=seed)

        method = str(self.config.get("run", {}).get("search_method", "grid")).lower()
        self._active_case_num = case_num
        if method == "bayes":
            configs = []
        elif method == "random":
            seed_val = int(self.config.get("run", {}).get("random_seed", 42))
            configs = [self._sample_random_config(i, seed=seed_val) for i in range(max_trials)]
        else:  # grid (default)
            configs = self._iter_configs(max_trials)
        run_summary_rows: List[Dict[str, Any]] = []
        best_cfg: Dict[str, Any] = {}
        best_score = -1.0
        best_per_query: List[Dict[str, Any]] = []
        all_ragchecker: List[str] = []

        metric_cols = [
            "context_recall", "answer_similarity", "faithfulness",
            "retrieval_coverage_proxy", "groundedness", "citation_quality",
            "answer_relevancy", "context_relevancy", "signal_quality",
            "composite_no_judge", "judge_weight", "judge_score",
            "query_program_depth", "query_rewrite_usage", "decompose_branch_count", "hyde_usage",
            "metadata_filter_hit_rate", "metadata_bonus_avg", "entity_match_rate",
            "rrf_fusion_gain", "citation_completeness", "citation_binding",
        ]

        def _avg_metric(rows: List[Dict[str, Any]], key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in rows if r.get(key, "") not in ("", None)]
            return round(sum(vals) / max(1, len(vals)), 4) if vals else 0.0

        # 在训练集上搜索最优配置
        if method == "bayes":
            import optuna

            seed = int(self.config.get("run", {}).get("random_seed", 42))
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial: Any) -> float:
                nonlocal best_cfg, best_score, best_per_query, all_ragchecker
                cfg = self._sample_config_by_trial(trial)
                t0 = time.perf_counter()
                res = self._run_case(case_num, cfg, dataset_override=train_set)
                trial_seconds = round(time.perf_counter() - t0, 4)
                guarded_score = float(res["mean_composite"])

                row = {
                    "config_id": cfg["config_id"],
                    "mean_composite": guarded_score,
                    "trial_seconds": trial_seconds,
                    "retriever": cfg["retrieval"]["retriever"],
                    "embedding_model": cfg["retrieval"].get("embedding_model", "tfidf"),
                    "chunk_strategy": cfg["chunking"]["strategy"],
                    "chunk_size": cfg["chunking"]["size"],
                    "rerank_enabled": cfg["reranking"]["enabled"],
                    "query_rewrite_enabled": cfg["query_processor"].get("rewrite", False),
                    "query_decompose_enabled": cfg["query_processor"].get("decompose", False),
                    "query_hyde_enabled": cfg["query_processor"].get("use_hyde", False),
                    "intent_driven_filter": cfg["query_processor"].get("intent_driven_filter", False),
                    "metadata_filter_enabled": cfg["retrieval"].get("metadata_filter_enabled", False),
                    "metadata_enrichment": cfg["retrieval"].get("metadata_enrichment", False),
                    "rerank_top_k": cfg["reranking"].get("rerank_top_k", self.top_k),
                    "prompt_template": cfg["generation"].get("prompt_template", "standard"),
                    "max_new_tokens": cfg["generation"].get("max_new_tokens", 256),
                    **{k: _avg_metric(res["per_query"], k) for k in metric_cols},
                }
                run_summary_rows.append(row)
                if guarded_score > best_score:
                    best_score = guarded_score
                    best_cfg = cfg
                    best_per_query = res["per_query"]
                    all_ragchecker = res["ragchecker_findings"]
                return float(guarded_score)

            study.optimize(objective, n_trials=max_trials)
        else:
            for cfg in configs:
                t0 = time.perf_counter()
                res = self._run_case(case_num, cfg, dataset_override=train_set)
                trial_seconds = round(time.perf_counter() - t0, 4)
                guarded_score = float(res["mean_composite"])

                row = {
                    "config_id": cfg["config_id"],
                    "mean_composite": guarded_score,
                    "trial_seconds": trial_seconds,
                    "retriever": cfg["retrieval"]["retriever"],
                    "embedding_model": cfg["retrieval"].get("embedding_model", "tfidf"),
                    "chunk_strategy": cfg["chunking"]["strategy"],
                    "chunk_size": cfg["chunking"]["size"],
                    "rerank_enabled": cfg["reranking"]["enabled"],
                    "query_rewrite_enabled": cfg["query_processor"].get("rewrite", False),
                    "query_decompose_enabled": cfg["query_processor"].get("decompose", False),
                    "query_hyde_enabled": cfg["query_processor"].get("use_hyde", False),
                    "intent_driven_filter": cfg["query_processor"].get("intent_driven_filter", False),
                    "metadata_filter_enabled": cfg["retrieval"].get("metadata_filter_enabled", False),
                    "metadata_enrichment": cfg["retrieval"].get("metadata_enrichment", False),
                    "rerank_top_k": cfg["reranking"].get("rerank_top_k", self.top_k),
                    "prompt_template": cfg["generation"].get("prompt_template", "standard"),
                    "max_new_tokens": cfg["generation"].get("max_new_tokens", 256),
                    **{k: _avg_metric(res["per_query"], k) for k in metric_cols},
                }
                run_summary_rows.append(row)
                if guarded_score > best_score:
                    best_score = guarded_score
                    best_cfg = cfg
                    best_per_query = res["per_query"]
                    all_ragchecker = res["ragchecker_findings"]

        if (not best_cfg) and run_summary_rows:
            fallback_row = max(run_summary_rows, key=lambda r: float(r.get("mean_composite", -1.0)))
            fallback_id = str(fallback_row.get("config_id", ""))
            candidates = configs if method != "bayes" else []
            for cfg_candidate in candidates:
                if str(cfg_candidate.get("config_id", "")) == fallback_id:
                    best_cfg = cfg_candidate
                    break
            if not best_cfg and method == "bayes":
                # bayes 场景下若 guard 全部判无效，使用 study best_trial 对应配置兜底
                try:
                    best_cfg = self._sample_config_by_trial(study.best_trial)
                except Exception:
                    best_cfg = {}

        # 在保留集上验证最优配置（防过拟合检验）
        holdout_res = self._run_case(case_num, best_cfg, dataset_override=holdout_set) if best_cfg else None
        holdout_score = holdout_res["mean_composite"] if holdout_res else 0.0

        best_path = out_dir / "best_config.json"
        best_data: Dict[str, Any] = {}
        if best_path.exists():
            try:
                best_data = read_json(best_path)
            except Exception:
                best_data = {}
        best_data[f"case{case_num}"] = {
            **best_cfg,
            "train_score": best_score,
            "holdout_score": holdout_score,
            "overfit_gap": round(best_score - holdout_score, 4),
        }
        write_json(best_path, best_data)

        run_summary_path = out_dir / "run_summary.csv"
        run_rows_with_case = [{**r, "case_num": case_num} for r in run_summary_rows]
        if run_summary_path.exists() and case_num != 1:
            try:
                prev = read_csv(run_summary_path)
            except Exception:
                prev = []
            run_rows_with_case = prev + run_rows_with_case
        write_csv(
            run_summary_path,
            run_rows_with_case,
            [
                "case_num", "config_id", "mean_composite", "trial_seconds",
                "retriever", "embedding_model", "chunk_strategy", "chunk_size", "rerank_enabled", "rerank_top_k",
                "query_rewrite_enabled", "query_decompose_enabled", "query_hyde_enabled", "intent_driven_filter",
                "metadata_filter_enabled", "metadata_enrichment", "prompt_template", "max_new_tokens",
                *metric_cols,
            ],
        )

        # 多目标 Pareto 分析（AutoRAG 风格：质量 vs 延迟权衡，默认启用）
        # 参考 AutoRAG 的 multi-objective trial selection 思路：
        # 不只选 mean_composite 最高的配置，而是找出质量-延迟 Pareto 前沿
        # 让用户在"更好质量"与"更低延迟"之间按需权衡
        pareto_rows: List[Dict[str, Any]] = []
        for row in run_summary_rows:
            dominated = False
            for other in run_summary_rows:
                if other["config_id"] == row["config_id"]:
                    continue
                better_or_equal_quality = float(other["mean_composite"]) >= float(row["mean_composite"])
                lower_or_equal_latency = float(other["trial_seconds"]) <= float(row["trial_seconds"])
                strictly_better = (
                    float(other["mean_composite"]) > float(row["mean_composite"])
                    or float(other["trial_seconds"]) < float(row["trial_seconds"])
                )
                if better_or_equal_quality and lower_or_equal_latency and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto_rows.append(row)

        # Pareto 前沿始终输出到 outputs（不依赖 mlflow 开关）
        pareto_rows.sort(key=lambda x: (-float(x["mean_composite"]), float(x["trial_seconds"])))
        pareto_csv_path = out_dir / f"pareto_frontier_case{case_num}.csv"
        write_csv(
            pareto_csv_path,
            pareto_rows,
            ["config_id", "mean_composite", "trial_seconds", "retriever",
             "embedding_model", "chunk_strategy", "chunk_size", "rerank_enabled", "rerank_top_k",
             "prompt_template", "max_new_tokens", *metric_cols],
        )
        self._write_pareto_plot(out_dir, pareto_rows, case_num, best_cfg)

        if bool(self.config.get("run", {}).get("mlflow_enabled", False)):
            self._log_mlflow_case(
                case_num=case_num,
                best_cfg=best_cfg,
                train_score=best_score,
                holdout_score=holdout_score,
                run_summary_path=out_dir / "run_summary.csv",
                pareto_csv_path=pareto_csv_path,
            )


        if best_per_query:
            diag_fields = [
                "config_id", "query_id", "query", "retrieved_doc_ids", "answer",
                "composite", "context_recall", "answer_similarity", "faithfulness",
                "retrieval_coverage_proxy", "groundedness", "citation_quality",
                "answer_relevancy", "context_relevancy", "signal_quality", "composite_no_judge", "judge_weight",
                "failure_reason", "ragchecker_findings", "hallucination_risk",
                "retrieval_bias", "query_drift",
                "judge_score", "judge_method", "judge_warning",
                "retrieval_failure_type", "grounding_failure_type", "citation_failure_type", "query_processing_failure_type",
                "citation_completeness", "citation_binding",
                "query_program_final_queries", "query_program_execution_path",
                "query_program_depth", "query_rewrite_usage", "decompose_branch_count", "hyde_usage",
                "metadata_filter_hit_rate", "metadata_bonus_avg", "entity_match_rate", "rrf_fusion_gain",
            ]
            per_query_path = out_dir / "per_query_diagnostics.csv"
            diag_rows_with_case = [{**r, "case_num": case_num} for r in best_per_query]
            if per_query_path.exists() and case_num != 1:
                try:
                    prev_diag = read_csv(per_query_path)
                except Exception:
                    prev_diag = []
                diag_rows_with_case = prev_diag + diag_rows_with_case

            write_csv(per_query_path, diag_rows_with_case, ["case_num", *diag_fields])
            self._write_metric_sanity_check(out_dir, case_num, best_per_query)

            for r in best_per_query:
                (out_dir / "retrieval_examples" / f"case{case_num}_{r['query_id']}.txt").write_text(
                    f"Query: {r['query']}\n\nRetrieved: {r['retrieved_doc_ids']}\n", encoding="utf-8"
                )
                (out_dir / "answer_examples" / f"case{case_num}_{r['query_id']}.txt").write_text(
                    f"Query: {r['query']}\n\n"
                    f"Answer: {r['answer']}\n\n"
                    f"Composite: {r['composite']}\n"
                    f"Failure: {r['failure_reason']}\n"
                    f"RAGChecker: {r['ragchecker_findings']}\n",
                    encoding="utf-8",
                )

        self.case_results[case_num] = {
            "best_cfg": best_cfg,
            "train_score": best_score,
            "holdout_score": holdout_score,
            "overfit_gap": round(best_score - holdout_score, 4),
            "external_signal_availability": self._estimate_external_signal_availability(),
        }
        self._active_case_num = None


    def _write_pareto_plot(self, out_dir: Path, rows: List[Dict[str, Any]], case_num: int, best_cfg: Dict[str, Any]) -> None:
        """
        输出 Pareto 散点图（质量 vs 延迟），并同时标出最终选择点。
        - 每个点标注 config_id
        - 颜色映射 mean_composite 高低（越黄越优）
        - 标注最优质量点（Best Quality）与最低延迟点（Fastest）
        - 标注最终选择点（Selected）
        - 横轴/纵轴带参考线（均值）
        - 图注说明 Pareto 前沿含义
        """
        if not rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np

            xs = [float(r["trial_seconds"]) for r in rows]
            ys = [float(r["mean_composite"]) for r in rows]
            labels = [str(r.get("config_id", i)) for i, r in enumerate(rows)]
            retrievers = [str(r.get("retriever", "")) for r in rows]
            chunks = [f"{r.get('chunk_strategy','')}/{r.get('chunk_size','')}" for r in rows]

            # 颜色映射：quality 越高越黄
            norm = plt.Normalize(min(ys), max(ys))
            colors = cm.RdYlGn(norm(ys))

            fig, ax = plt.subplots(figsize=(9, 5.5))
            sc = ax.scatter(xs, ys, c=ys, cmap="RdYlGn", s=80, zorder=3,
                            vmin=min(ys), vmax=max(ys), edgecolors="#333", linewidths=0.5)
            plt.colorbar(sc, ax=ax, label="mean_composite (quality)")

            # 标注每个点
            for x, y, lab, ret, chk in zip(xs, ys, labels, retrievers, chunks):
                ax.annotate(
                    f"{lab}\n{ret}|{chk}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=6.5,
                    color="#333",
                )

            # 标注最优质量点
            best_q_idx = int(np.argmax(ys))
            ax.scatter([xs[best_q_idx]], [ys[best_q_idx]], s=180, marker="*",
                       color="gold", zorder=5, label=f"Best Quality: {labels[best_q_idx]}")

            # 标注最快点
            best_t_idx = int(np.argmin(xs))
            ax.scatter([xs[best_t_idx]], [ys[best_t_idx]], s=120, marker="D",
                       color="steelblue", zorder=5, label=f"Fastest: {labels[best_t_idx]}")

            # 标注最终选择点
            selected_id = str(best_cfg.get("config_id", ""))
            for x, y, lab in zip(xs, ys, labels):
                if lab == selected_id:
                    ax.scatter([x], [y], s=220, marker="P", color="crimson", zorder=6,
                               label=f"Selected: {selected_id}")
                    ax.annotate(
                        f"Selected\n{selected_id}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(10, -14),
                        fontsize=7.5,
                        fontweight="bold",
                        color="crimson",
                    )
                    break

            # 均值参考线
            ax.axhline(np.mean(ys), color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Avg quality")
            ax.axvline(np.mean(xs), color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="Avg latency")

            case_label = "Case1 (Supervised)" if case_num == 1 else "Case2 (Weakly-supervised)"
            ax.set_xlabel("trial_seconds (lower = faster)", fontsize=10)
            ax.set_ylabel("mean_composite (higher = better)", fontsize=10)
            ax.set_title(
                f"Pareto Frontier: Quality vs Latency  [{case_label}]\n"
                f"* = Best Quality config    D = Fastest config    P = Selected config",
                fontsize=10,
            )
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / f"_pareto_frontier_case{case_num}.png", dpi=150)
            plt.close(fig)
        except Exception:
            return

    def _log_mlflow_case(
        self,
        case_num: int,
        best_cfg: Dict[str, Any],
        train_score: float,
        holdout_score: float,
        run_summary_path: Path,
        pareto_csv_path: Path,
    ) -> None:
        """按 Case 记录一次 MLflow run（可选）。"""
        if not bool(self.config.get("run", {}).get("mlflow_enabled", False)):
            return
        try:
            import mlflow

            tracking_uri = str(self.config.get("run", {}).get("mlflow_tracking_uri", "")).strip()
            if not tracking_uri:
                db_path = (self.output_dir / "mlflow.db").resolve().as_posix()
                tracking_uri = f"sqlite:///{db_path}"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("rag_optimizer")
            with mlflow.start_run(run_name=f"case{case_num}_{best_cfg.get('config_id', 'na')}"):
                mlflow.log_param("case_num", case_num)
                mlflow.log_param("config_id", best_cfg.get("config_id", "N/A"))
                mlflow.log_param("retriever", best_cfg.get("retrieval", {}).get("retriever", "N/A"))
                mlflow.log_param("chunk_strategy", best_cfg.get("chunking", {}).get("strategy", "N/A"))
                mlflow.log_param("chunk_size", best_cfg.get("chunking", {}).get("size", "N/A"))
                mlflow.log_param("rerank_enabled", best_cfg.get("reranking", {}).get("enabled", "N/A"))

                mlflow.log_metric("train_score", float(train_score))
                mlflow.log_metric("holdout_score", float(holdout_score))
                mlflow.log_metric("overfit_gap", float(train_score - holdout_score))

                if run_summary_path.exists():
                    mlflow.log_artifact(str(run_summary_path))
                if pareto_csv_path.exists():
                    mlflow.log_artifact(str(pareto_csv_path))
                pareto_png = pareto_csv_path.with_suffix(".png")
                if pareto_png.exists():
                    mlflow.log_artifact(str(pareto_png))
        except Exception:
            return

    def write_final_report(self) -> None:
        """生成中文最终报告（委托给 report_generator 模块）。"""
        generate_final_report(self.output_dir, self.case_results)
