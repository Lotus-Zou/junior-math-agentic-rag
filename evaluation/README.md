# 自动化评测

本目录包含覆盖七至九年级代数、几何、函数、统计与概率的 1000 条标注测试集：334 条普通样例、333 条幻觉高危样例和 333 条口语化样例。

## 有效资产

- `math_benchmark_1000.csv`：含 case 类型、年级、章节、知识点、错误作答、错因、参考答案、目标上下文关键词和来源。
- `generate_dataset.py`：确定性重建并校验数据集。
- `evaluation.py`：支持 `dense`、`hybrid`、`hybrid_graph` 检索 A/B、端到端评测、RAGAS 和质量门禁。
- `chaos_cases.yaml`：模型超时、供应商异常、空检索、Critic 拒绝、练习过期和缓存异常的可执行故障用例。
- `reports/`、`latest_summary.json`：逐题 Trace 报告和最新汇总指标。
- `benchmark_history.json`：用户提供的历史项目数据，明确标记是否在当前工作区复现。

旧的 `golden_dataset.csv`、路由混淆矩阵和早期 RAGAS 报告仅保留作通用 RAG 版本的历史对照，不再参与当前评测。

## 使用

```powershell
# Skill 契约与 YAML Pipeline 回归
python evaluation/skill_harness.py --report
python evaluation/pipeline_harness.py

# 产品 bad-case、六类 chaos、1000 条确定性覆盖与延迟 SLA 门禁
python evaluation/bad_case_harness.py --report

# 自适应练习有效性、隐藏答案隔离和每配置 100 种子去重门禁
python evaluation/exercise_quality_harness.py --report

# 只校验条数、字段、唯一 ID 和空标注
python evaluation/evaluation.py --mode validate

# 对比纯向量、向量 + BM25、向量 + BM25 + GraphRAG
python evaluation/evaluation.py --mode ab --limit 100 --k 5 --version dev

# 跑 100 条端到端业务指标，不调用 RAGAS Judge
python evaluation/evaluation.py --mode e2e --limit 100 --skip-ragas --version dev

# 全量 RAGAS（会调用外部模型并产生费用）
python evaluation/evaluation.py --mode e2e --limit 0 --k 5 --version release

# 用 latest_summary.json 执行质量门禁
python evaluation/evaluation.py --mode gate
```

`skill_harness.py` 与 `pipeline_harness.py` 同时支持 `expected`、`contains` 和 `answer_contains` 断言，以及 `expected_paths` 点路径断言。Skill 的前两者检查顶层 Skill 结果，Pipeline 的前两者检查 `expected_node`，两者的 `answer_contains` 都检查其公开答案。点路径可以检查嵌套的 Skill 响应，例如 `response.response_type`；Pipeline case 则针对完整执行 state，例如 `curriculum_solve.response.response_type`。`absent_paths` 要求路径真正不存在（存在但为 `null` 仍会失败）；`not_contains` 扫描完整公开响应序列化（Skill 为 `response`，Pipeline 为 `response_render`），以阻止内部报告、验证证据、隐藏答案或内部失败文案泄露。本地对话控制回归覆盖 `几何`、`代数`、`一次函数`、`换个问题`、`难一点` 和 `简单一点`，并固定其零工具调用路径。

`exercise_quality_harness.py` 对 `exercise_cases.jsonl` 中每个年级、主题、难度和题型组合生成 100 个确定性种子，要求数学验证失败数与公开答案泄漏数均为 0，并以最差配置口径要求题目及提示唯一率不低于 90%。

`bad_case_harness.py` 会先执行 `chaos_cases.yaml`，要求所有故障只返回批准的公开响应类型、明确的下一步和 Trace ID，同时禁止“复杂推理服务”“8 秒”“bad case”“知识库没有召回”、内部 Critic/异常及供应商详情进入学生回答。

检索模式使用透明的关键词 `Context Precision/Recall` 与知识点匹配率；E2E 模式增加意图、错因、步骤、直接答案违规和幻觉检出指标；RAGAS 负责 `Context Precision`、`Context Recall`、`Faithfulness`、`Answer Relevance`。

历史口径为 Recall 提升 15%、幻觉率 `35% -> 10%`、Answer Relevance `0.50 -> 0.78`，当前工作区尚未全量复现。10 条 smoke A/B 的三种策略 Context Recall 均为 `0.9667`，不构成提升证据。
