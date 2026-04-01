from __future__ import annotations

import itertools
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from src.chunking.chunker import chunk_document
from src.evaluation.diagnostics import diagnose_case, full_ragchecker_report
from src.evaluation.metrics import case1_metrics, case2_metrics
from src.generation.generator import generate_answer
from src.retrieval.query_processor import apply_query_processor
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
        """生成候选配置列表。

        策略：先对最重要的维度做完整笛卡尔积（primary grid），
        再对次要维度做循环填充（cycling），保证即使 max_trials 较小
        时也能覆盖到不同 retriever / chunk_strategy / chunk_size 组合。

        Primary（逐一组合）：
          retriever x chunk_strategy x chunk_size x answer_style x rerank_enabled
          共 3x3x4x2x2 = 144 个主要组合，完整覆盖关键维度搜索空间。

        Secondary（循环填充，不展开组合爆炸）：
          embedding_model、metadata_filter_enabled、rewrite、decompose、
          overlap_type、overlap_size、temperature、reranking_model
        """
        ss = self.config["search_space"]

        # Primary grid: 最关键维度全量组合
        primary = list(itertools.product(
            ss["retrieval"]["retriever"],     # bm25 / dense / hybrid
            ss["chunking"]["strategy"],       # token / sentence / semantic
            ss["chunking"]["size"],           # 128 / 256 / 384 / 512
            ss["generation"]["answer_style"], # concise / citation_first
            ss["reranking"]["enabled"],       # False / True
        ))

        # Secondary: 次要维度按 cycle 补全
        from itertools import cycle as _cycle
        emb_cyc  = _cycle(ss["retrieval"]["embedding_model"])
        mfil_cyc = _cycle(ss["retrieval"]["metadata_filter_enabled"])
        rew_cyc  = _cycle(ss["query_processor"]["rewrite"])
        dec_cyc  = _cycle(ss["query_processor"]["decompose"])
        ovt_cyc  = _cycle(ss["chunking"]["overlap_type"])
        ovs_cyc  = _cycle(ss["chunking"]["overlap_size"])
        temp_cyc = _cycle(ss["generation"].get("temperature", [0.0]))
        rm_cyc   = _cycle(ss["reranking"]["model"])

        out: List[Dict[str, Any]] = []
        for idx, (retriever, strategy, size, answer_style, rerank_en) in enumerate(primary):
            out.append({
                "config_id": f"cfg_{idx:05d}",
                "retrieval": {
                "retriever": trial.suggest_categorical("retrieval_retriever", ss["retrieval"]["retriever"]),
                "embedding_model": trial.suggest_categorical("retrieval_embedding_model", ss["retrieval"]["embedding_model"]),
                "metadata_filter_enabled": trial.suggest_categorical("retrieval_metadata_filter_enabled", ss["retrieval"]["metadata_filter_enabled"]),
                "metadata_enrichment": True,
                "metadata_filter_fields": ss["retrieval"].get("metadata_filter_fields", ["section_title", "entities", "type"]),
            },
            "query_processor": {
                "rewrite": trial.suggest_categorical("query_rewrite", ss["query_processor"]["rewrite"]),
                "decompose": trial.suggest_categorical("query_decompose", ss["query_processor"]["decompose"]),
                "intent_driven_filter": trial.suggest_categorical("query_intent_driven_filter", ss["query_processor"].get("intent_driven_filter", [False])),
            },
            "generation": {
                "answer_style": trial.suggest_categorical("generation_answer_style", ss["generation"]["answer_style"]),
                "temperature": trial.suggest_categorical("generation_temperature", ss["generation"].get("temperature", [0.0])),
            },
            "reranking": {
                "enabled": trial.suggest_categorical("reranking_enabled", ss["reranking"]["enabled"]),
                "model": trial.suggest_categorical("reranking_model", ss["reranking"]["model"]),
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
                "metadata_enrichment": True,
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
                "preserve_table_as_markdown": True,
                "generate_image_caption": False,
            },
            "generation": {
                "answer_style": _pick(ss["generation"]["answer_style"]),
                "temperature": _pick(ss["generation"].get("temperature", [0.0])),
            },
            "reranking": {
                "enabled": _pick(ss["reranking"]["enabled"]),
                "model": _pick(ss["reranking"]["model"]),
            },
            "query_processor": {
                "rewrite": _pick(ss["query_processor"]["rewrite"]),
                "decompose": _pick(ss["query_processor"]["decompose"]),
                "intent_driven_filter": _pick(ss["query_processor"].get("intent_driven_filter", [False])),
            },
        }

    def _build_chunks(self, cfg: Dict[str, Any]):
        chunks = []
        for d in self.corpus:
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
                    base_metadata={
                        "title": d.get("title", ""),
                        "source": d.get("source", ""),
                        "page_number": d.get("page_number", None),
                    },
                )
            )
        return chunks

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

    def _run_case(self, case_num: int, cfg: Dict[str, Any], dataset_override: list | None = None) -> Dict[str, Any]:
        dataset = dataset_override if dataset_override is not None else (self.case1 if case_num == 1 else self.case2)
        chunks = self._build_chunks(cfg)
        retriever = Retriever(
            chunks,
            retriever_type=cfg["retrieval"]["retriever"],
            embedding_model=cfg["retrieval"].get("embedding_model", "tfidf"),
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

            metadata_filter = {}
            if cfg["retrieval"]["metadata_filter_enabled"]:
                lowered = query.lower()
                if "kyc" in lowered:
                    metadata_filter = {"section_title": "kyc"}
                elif "fee" in lowered:
                    metadata_filter = {"section_title": "fee"}

            retrieved = retriever.retrieve(query, top_k=self.top_k, metadata_filter=metadata_filter)
            # 查询改写/分解：对多个子查询分别检索后合并去重
            sub_queries = apply_query_processor(
                query,
                rewrite=cfg["query_processor"].get("rewrite", False),
                decompose=cfg["query_processor"].get("decompose", False),
            )
            if len(sub_queries) > 1:
                seen_ids: set = {c.chunk_id for c, _ in retrieved}
                for sq in sub_queries[1:]:
                    for chunk, score in retriever.retrieve(sq, top_k=self.top_k, metadata_filter=metadata_filter):
                        if chunk.chunk_id not in seen_ids:
                            retrieved.append((chunk, score))
                            seen_ids.add(chunk.chunk_id)
                retrieved = sorted(retrieved, key=lambda x: x[1], reverse=True)[:self.top_k]
            retrieved = rerank(retrieved, enabled=cfg["reranking"]["enabled"], query=query)
            answer = generate_answer(
                query=query,
                retrieved=retrieved,
                answer_style=cfg["generation"]["answer_style"],
                temperature=cfg["generation"].get("temperature", 0.0),
                llm_model=str(self.config.get("run", {}).get("llm_model", "qwen2.5:7b-instruct")),
                use_llm=bool(self.config.get("run", {}).get("use_llm_generator", True)),
            )
            use_ragas = bool(self.config.get("run", {}).get("use_ragas", False))
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
                    use_ragas=use_ragas,
                    use_bertscore=use_bertscore,
                    weights=case1_w,
                )
            else:
                m = case2_metrics(
                    retrieved=retrieved,
                    answer=answer,
                    reference_doc_ids=reference_ids,
                    query=query,
                    use_ragas=use_ragas,
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
                avg_ragas_used = round(
                    sum(float(r.get("ragas_used", 0.0)) for r in res["per_query"]) / max(1, len(res["per_query"])), 4
                )
                use_ragas = bool(self.config.get("run", {}).get("use_ragas", False))
                ragas_min_usage_threshold = float(self.config.get("run", {}).get("ragas_min_usage_threshold", 0.6))
                guard_mode = str(self.config.get("run", {}).get("ragas_guard_mode", "invalid")).lower()
                guard_penalty = float(self.config.get("run", {}).get("ragas_guard_penalty", 0.2))
                signal_guard_triggered = use_ragas and (avg_ragas_used < ragas_min_usage_threshold)

                guarded_score = float(res["mean_composite"])
                if signal_guard_triggered:
                    if guard_mode == "invalid":
                        guarded_score = -1.0
                    else:
                        guarded_score = round(guarded_score * guard_penalty, 4)

                row = {
                    "config_id": cfg["config_id"],
                    "mean_composite": guarded_score,
                    "raw_mean_composite": res["mean_composite"],
                    "avg_ragas_used": avg_ragas_used,
                    "signal_guard_triggered": str(signal_guard_triggered),
                    "trial_seconds": trial_seconds,
                    "retriever": cfg["retrieval"]["retriever"],
                    "embedding_model": cfg["retrieval"].get("embedding_model", "tfidf"),
                    "chunk_strategy": cfg["chunking"]["strategy"],
                    "chunk_size": cfg["chunking"]["size"],
                    "rerank_enabled": cfg["reranking"]["enabled"],
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
                avg_ragas_used = round(
                    sum(float(r.get("ragas_used", 0.0)) for r in res["per_query"]) / max(1, len(res["per_query"])), 4
                )
                use_ragas = bool(self.config.get("run", {}).get("use_ragas", False))
                ragas_min_usage_threshold = float(self.config.get("run", {}).get("ragas_min_usage_threshold", 0.6))
                guard_mode = str(self.config.get("run", {}).get("ragas_guard_mode", "invalid")).lower()
                guard_penalty = float(self.config.get("run", {}).get("ragas_guard_penalty", 0.2))
                signal_guard_triggered = use_ragas and (avg_ragas_used < ragas_min_usage_threshold)

                guarded_score = float(res["mean_composite"])
                if signal_guard_triggered:
                    if guard_mode == "invalid":
                        guarded_score = -1.0
                    else:
                        guarded_score = round(guarded_score * guard_penalty, 4)

                row = {
                    "config_id": cfg["config_id"],
                    "mean_composite": guarded_score,
                    "raw_mean_composite": res["mean_composite"],
                    "avg_ragas_used": avg_ragas_used,
                    "signal_guard_triggered": str(signal_guard_triggered),
                    "trial_seconds": trial_seconds,
                    "retriever": cfg["retrieval"]["retriever"],
                    "embedding_model": cfg["retrieval"].get("embedding_model", "tfidf"),
                    "chunk_strategy": cfg["chunking"]["strategy"],
                    "chunk_size": cfg["chunking"]["size"],
                    "rerank_enabled": cfg["reranking"]["enabled"],
                }
                run_summary_rows.append(row)
                if guarded_score > best_score:
                    best_score = guarded_score
                    best_cfg = cfg
                    best_per_query = res["per_query"]
                    all_ragchecker = res["ragchecker_findings"]

        if (not best_cfg) and run_summary_rows:
            fallback_row = max(run_summary_rows, key=lambda r: float(r.get("raw_mean_composite", r.get("mean_composite", -1.0))))
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
            ["case_num", "config_id", "mean_composite", "raw_mean_composite", "avg_ragas_used", "signal_guard_triggered", "trial_seconds", "retriever", "embedding_model", "chunk_strategy", "chunk_size", "rerank_enabled"],
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
            ["config_id", "mean_composite", "raw_mean_composite", "avg_ragas_used", "signal_guard_triggered", "trial_seconds", "retriever",
             "embedding_model", "chunk_strategy", "chunk_size", "rerank_enabled"],
        )
        self._write_pareto_plot(out_dir, pareto_rows, case_num)

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
                "answer_relevancy", "context_relevancy", "ragas_used", "signal_quality", "composite_no_judge", "judge_weight",
                "failure_reason", "ragchecker_findings", "hallucination_risk",
                "retrieval_bias", "query_drift",
                "judge_score", "judge_method", "judge_warning",
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
        }

    def _write_pareto_plot(self, out_dir: Path, rows: List[Dict[str, Any]], case_num: int) -> None:
        """
        输出 Pareto 散点图（质量 vs 延迟），含以下增强可视化：
        - 每个点标注 config_id
        - 颜色映射 mean_composite 高低（越黄越优）
        - 标注最优质量点（Best Quality）与最低延迟点（Fastest）
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

            # 均值参考线
            ax.axhline(np.mean(ys), color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Avg quality")
            ax.axvline(np.mean(xs), color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="Avg latency")

            case_label = "Case1 (Supervised)" if case_num == 1 else "Case2 (Weakly-supervised)"
            ax.set_xlabel("trial_seconds (lower = faster)", fontsize=10)
            ax.set_ylabel("mean_composite (higher = better)", fontsize=10)
            ax.set_title(
                f"Pareto Frontier: Quality vs Latency  [{case_label}]\n"
                f"* = Best Quality config    D = Fastest config",
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

            tracking_uri = str(self.output_dir / "mlruns")
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
