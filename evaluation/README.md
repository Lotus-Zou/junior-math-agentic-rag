# 自动化评测

本目录包含覆盖七至九年级代数、几何、函数、统计与概率的 1000 条标注测试集：334 条普通样例、333 条幻觉高危样例和 333 条口语化样例。

## 有效资产

- `math_benchmark_1000.csv`：含 case 类型、年级、章节、知识点、错误作答、错因、参考答案、目标上下文关键词和来源。
- `generate_dataset.py`：确定性重建并校验数据集。
- `evaluation.py`：支持 `dense`、`hybrid`、`hybrid_graph` 检索 A/B、端到端评测、RAGAS 和质量门禁。
- `public_benchmarks.py`：通过生产 `/ask` 接口分别运行 C-Eval 初中数学与 GSM8K，分别保存准确率，禁止混合成一个分数。
- `mteb_retrieval.py`：使用生产配置的 embedding 模型运行 BEIR `SciFact` 与 C-MTEB `CovidRetrieval`，分别保存 NDCG@10、MAP@10、Recall@10。
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
python evaluation/evaluation.py --mode e2e --dataset ragas --limit 0 --k 5 --version release

# 只复测一个 bad case；报告同时保存期望/实际意图与知识点
python evaluation/evaluation.py --mode e2e --dataset ragas --case-id RAG-002 --skip-ragas --version rag-002

# 公开数学题集小样本；两个数据集分别报告 accuracy
python evaluation/public_benchmarks.py --benchmark all --limit 5 --version smoke

# 只从完整 checkpoint 重建指标，不调用 API、不重试失败样例
python evaluation/public_benchmarks.py --benchmark gsm8k --limit 0 --version full --report-only

# 同一 version 只能有一个进程；崩溃遗留锁且确认无进程后才可显式解锁
python evaluation/public_benchmarks.py --benchmark gsm8k --limit 0 --workers 2 --version full --force-unlock

# PRM800K 人工步骤标签：固定 seed 300 条首错定位，再对同一批回答运行完整 RAGAS
python evaluation/prm800k_diagnosis.py --limit 300 --workers 2 --version prm300
python evaluation/prm800k_diagnosis.py --limit 300 --workers 2 --version prm300 --ragas

# 单独安装较重的公开检索评测依赖
uv pip install --python .venv/Scripts/python.exe -r requirements-eval.txt

# BEIR + C-MTEB；会下载公开数据和 embedding 模型，CPU 全量运行耗时较长
python evaluation/mteb_retrieval.py --tasks SciFact CovidRetrieval --version release

# 用 latest_summary.json 执行质量门禁
python evaluation/evaluation.py --mode gate
```

公开端点下载失败时先固定相同 revision，再使用国内镜像，不允许换成来源不明的二次打包数据：

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
python evaluation/public_benchmarks.py --benchmark gsm8k --limit 0 --version full
```

每个真实数据源的 split、规模、cache fingerprint、commit 或逐文件 SHA256 记录在
`evaluation/dataset_manifest.json`。C-Eval 只报告公开有标签的中学数学 `val`；隐藏
`test` 没有公开标签，不能写成官方 test 成绩。PRM800K 是英文竞赛数学人工步骤标签，
默认排除显式超出初中范围的主题；它评估首错定位，不等于中文初中错因类别准确率。

E2E 会在调用 RAGAS Judge 前先保存 `<version>_e2e.raw.csv`，并保存
`<version>_ragas.raw.csv` 的逐条 Judge 输出。Judge 超时或出现 NaN 时质量
门禁仍会失败，但生产回答、实际意图、实际知识点和失败评分不会丢失。

`skill_harness.py` 与 `pipeline_harness.py` 同时支持 `expected`、`contains` 和 `answer_contains` 断言，以及 `expected_paths` 点路径断言。Skill 的前两者检查顶层 Skill 结果，Pipeline 的前两者检查 `expected_node`，两者的 `answer_contains` 都检查其公开答案。点路径可以检查嵌套的 Skill 响应，例如 `response.response_type`；Pipeline case 则针对完整执行 state，例如 `curriculum_solve.response.response_type`。`absent_paths` 要求路径真正不存在（存在但为 `null` 仍会失败）；`not_contains` 扫描完整公开响应序列化（Skill 为 `response`，Pipeline 为 `response_render`），以阻止内部报告、验证证据、隐藏答案或内部失败文案泄露。本地对话控制回归覆盖 `几何`、`代数`、`一次函数`、`换个问题`、`难一点` 和 `简单一点`。离线 Harness 固定关闭外部模型以保证可重复，生产配置则要求数学题经过对应 Agent；专门的假模型测试负责验证调用契约。

`exercise_quality_harness.py` 对 `exercise_cases.jsonl` 中每个年级、主题、难度和题型组合生成 100 个确定性种子，要求数学验证失败数与公开答案泄漏数均为 0，并以最差配置口径要求题目及提示唯一率不低于 90%。

`bad_case_harness.py` 会先执行 `chaos_cases.yaml`，要求所有故障只返回批准的公开响应类型、明确的下一步和 Trace ID，同时禁止“复杂推理服务”“8 秒”“bad case”“知识库没有召回”、内部 Critic/异常及供应商详情进入学生回答。

检索模式使用透明的关键词 `Context Precision/Recall` 与知识点匹配率；E2E 模式增加意图、错因、步骤、直接答案违规和幻觉检出指标；RAGAS 负责 `Context Precision`、`Context Recall`、`Faithfulness`、`Answer Relevance`。

C-Eval/GSM8K 的准确率、MTEB/BEIR 的检索分数、RAGAS 的 RAG 质量指标和领域错因诊断指标属于四种不同口径，报告与质量门禁均不得相互替代或合并。

历史口径为 Recall 提升 15%、幻觉率 `35% -> 10%`、Answer Relevance `0.50 -> 0.78`，当前工作区尚未全量复现。10 条 smoke A/B 的三种策略 Context Recall 均为 `0.9667`，不构成提升证据。

## 正式评测结果（2026-09-02）

统一汇总见 `evaluation/reports/formal-evaluation-20260902.json`。以下任务、指标和
适用范围相互独立，不合并成单一“总分”。

- C-Eval 中学数学：公开有标签 `val` 全部 19 条，生产 API accuracy `1.00000`，空预测 0。官方隐藏 `test` 无公开标签，因此这不是 C-Eval 官方 test 榜单成绩。
- GSM8K：官方 `main/test` 全部 1,319 条，生产 API accuracy `0.80591`；238 条空预测、18 条非空错误，非空回答 conditional accuracy `0.98335`。空预测暴露的是完整题目被 Critic/响应合同拒绝的可用性缺陷，不能从分母剔除。
- SciFact：BGE-M3、`max_seq_length=512`、CUDA，完整 5,183 篇 corpus / 300 个 judged queries / 339 条 qrels；NDCG@10 `0.64150`、MAP@10 `0.59201`、Recall@10 `0.77511`。
- CovidRetrieval：BGE-M3、`max_seq_length=512`、CUDA，完整 100,001 篇 corpus / 949 个 judged queries / 959 条 qrels；NDCG@10 `0.76522`、MAP@10 `0.72543`、Recall@10 `0.88830`。
- PRM800K：官方 `phase2_test` 固定 seed 的 300 条人工 found-error 标签子集；首错 exact accuracy `0.69333`、within-one accuracy `0.81000`、错误检出率 `0.98000`、解析率 `1.00000`。
- 同一 300 条 PRM800K 回答完成 RAGAS：Answer Relevance `0.60843`、Faithfulness `0.01279`、Context Precision/Recall 均为 `0`。该首错审查路由不执行知识库检索，contexts 为空；后 3 项只能证明当前回答没有检索依据，不能冒充 RAG 检索质量。

GSM8K 空预测审查发现，自包含题即使数学逻辑通过，仍会因无关检索上下文的引用检查
失败而被拒绝。当前修复允许完整自包含题以独立数学逻辑校验通过，不绕过逻辑 Critic；
对应生产 Pipeline 与 MTEB 报告器测试已通过。修复后的 GSM8K 全量 API 指标必须在
重新运行后另开 version 报告，不能覆盖上述已完成基线。

本次修改后完整 pytest 共收集并通过 1,039 个用例；唯一警告是
`langchain-community` 的上游弃用提示。

## 抽样与专项回归（不作为正式全量成绩）

- RAGAS 初中数学知识问答专项集：5 条，Answer Relevance `0.9378`、Faithfulness `0.9260`、Context Recall `1.0`。
- 早期 C-Eval/GSM8K 生产 API smoke：各 2 条，accuracy 分别为 `0.50` 与 `1.00`。
- BGE-M3 sampled MTEB smoke：每任务 50 query、512 corpus；SciFact NDCG@10 `0.79461`，CovidRetrieval NDCG@10 `0.92508`。
- 2026-08-29 工程门禁基线：1004 个 pytest、1000/1000 bad-case、6/6 chaos 和桌面/移动端 Playwright 回归通过。
