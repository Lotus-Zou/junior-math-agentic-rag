# 评测工具

本目录提供可重建的数据集、检索 A/B、端到端质量评测和故障测试。工程验证入口见
[测试与评测](../docs/Testing.md)，运行环境见 [运行指南](../docs/Quickstart.md)。

## 数据与指标

| 资产 | 用途 |
| --- | --- |
| `math_benchmark_1000.csv` | 1000 条去重的程序化标注回归题，10 类知识点 |
| `generate_dataset.py`、`dataset_audit.py` | 重建标签、来源关联与 SHA-256 审计 |
| `ragas_math_knowledge.csv` | 知识问答专项样例，按自身样本数报告 |
| `evaluation.py` | 检索 A/B、端到端业务指标、RAGAS 与质量门禁 |
| `paired_evaluation.py` | 同题单、同评分器的逐题配对比较 |
| `public_benchmarks.py`、`prm800k_diagnosis.py` | 公开数学题集与首错定位评测 |
| `mteb_retrieval.py` | BEIR / C-MTEB 检索评测 |
| `skill_harness.py`、`pipeline_harness.py` | Skill 与 YAML Pipeline 合同检查 |
| `bad_case_harness.py`、`chaos_cases.yaml` | 异常响应与故障恢复检查 |
| `exercise_quality_harness.py` | 练习数学验证、去重与隐藏答案隔离 |

公开数据源的 split、revision 与摘要见 `dataset_manifest.json`。不同数据集与指标
分别保存；检索分数、RAGAS、数学正确率与工程测试数量互不替代。

## 常用命令

```bash
python -m evaluation.dataset_audit --output evaluation/reports/dataset-audit.json
python evaluation/evaluation.py --mode validate
python evaluation/skill_harness.py --report
python evaluation/pipeline_harness.py
python evaluation/bad_case_harness.py --report
python evaluation/exercise_quality_harness.py --report

# 相同题单比较 Dense、Hybrid 与 Hybrid + Graph
python evaluation/evaluation.py --mode ab --limit 0 --k 5 --version retrieval
python evaluation/analyze_reproduction.py --version retrieval

# 端到端评测；会使用已配置的模型服务
python evaluation/evaluation.py --mode e2e --dataset ragas --limit 0 --version before
python evaluation/evaluation.py --mode e2e --dataset ragas --limit 0 --version after
python -m evaluation.paired_evaluation \
  --before evaluation/reports/before_paired.json \
  --after evaluation/reports/after_paired.json \
  --output evaluation/reports/comparison.json

# 使用新生成的完整 RAGAS 汇总执行质量门禁
python evaluation/evaluation.py --mode gate
```

## 实验约定

- 每次使用独立版本名保存报告，原始记录输出到忽略的 `evaluation/reports/`。
- 配对比较固定题单、Judge、embedding、评分器与指标定义；系统版本可以变化。
- 不完整答案保留在业务分母，Judge 缺分使相应比较不完整。
- 断点记录绑定题目、回答、上下文、参考答案与评分配置指纹。
- 模型评测可能产生费用；先验证端点、预算和小样本，再执行完整实验。
- 自包含数学题和检索知识问答分别分析，依据各自所需上下文解释 Faithfulness。
