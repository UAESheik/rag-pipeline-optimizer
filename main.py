from __future__ import annotations

import argparse
from pathlib import Path

from src.optimizer.optimizer import RAGOptimizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="RAG 流水线自动优化器")
    parser.add_argument("--case", type=int, choices=[1, 2], default=None, help="仅运行 Case1 或 Case2")
    parser.add_argument("--max-trials", type=int, default=20, help="每个 Case 最大实验次数")
    parser.add_argument("--top-k", type=int, default=None, help="每次检索返回的块数量（不传则用配置 run.top_k）")
    parser.add_argument("--config", type=str, default="config/optimizer_config.json", help="配置文件路径")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    cfg = Path(args.config)
    cfg_obj = None
    try:
        import json
        cfg_obj = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        cfg_obj = {}

    resolved_top_k = args.top_k
    if resolved_top_k is None:
        resolved_top_k = int(cfg_obj.get("run", {}).get("top_k", 5))

    optimizer = RAGOptimizer(
        config_path=cfg,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        top_k=resolved_top_k,
    )

    if args.case in (None, 1):
        optimizer.optimize(case_num=1, max_trials=args.max_trials)
    if args.case in (None, 2):
        optimizer.optimize(case_num=2, max_trials=args.max_trials)

    optimizer.write_final_report()


if __name__ == "__main__":
    main()
