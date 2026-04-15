# RAG 优化器最终报告

## 1. 项目目标
本项目通过统一优化框架，在 Case1（有监督）与 Case2（弱监督）下自动搜索并选择最优 RAG 配置，基于本地代码指标输出可解释的推荐结果。

## 2. Case1 推荐配置（有监督）
- 配置ID：`cfg_00044`
- 检索器：bm25
- 分块策略：semantic / size=384
- 重排：False
- 训练分数：0.6281，保留集分数：0.5554，过拟合差值：0.0727

**推荐原因：**
Case1 以 context_recall + answer_similarity + faithfulness 为目标，并用本地代码代理指标约束答案相关性与上下文相关性，最优配置在训练集与保留集上表现稳定，说明在有参考答案场景下泛化较好。

## 3. Case2 推荐配置（弱监督）
- 配置ID：`cfg_00008`
- 检索器：dense
- 分块策略：sentence / size=384
- 重排：False
- 训练分数：0.3102，保留集分数：0.2913，过拟合差值：0.0189

**推荐原因：**
Case2 无参考答案，优化目标切换为 retrieval_coverage_proxy + groundedness + citation_quality。该配置在弱监督代理指标下综合最优，且能够保持答案与检索证据的一致性。

## 4. Case1 vs Case2 对比总览
| 对比维度 | Case1（有监督） | Case2（弱监督） |
|---|---|---|
| 优化目标 | 提升答案与参考的一致性，同时保证证据支撑 | 在无参考答案条件下最大化证据覆盖与可溯源性 |
| 核心指标 | context_recall / answer_similarity / faithfulness | retrieval_coverage_proxy / groundedness / citation_quality |
| 推荐配置ID | cfg_00044 | cfg_00008 |
| train_score | 0.6281 | 0.3102 |
| holdout_score | 0.5554 | 0.2913 |
| overfit_gap | 0.0727 | 0.0189 |

## 5. Case1 到 Case2 的迁移优化思路
- 保留 Case1 中已验证有效的结构化分块、检索与重排搜索空间；
- 将优化目标从“答案匹配度”迁移到“证据覆盖+可溯源+ groundedness”；
- 保持同一优化器框架，仅替换评分函数，实现从有监督到弱监督的平滑迁移。

## 6. 结论
统一优化器在两种监督条件下都能产出可解释的最优配置，证明该优化框架具备较强通用性。
