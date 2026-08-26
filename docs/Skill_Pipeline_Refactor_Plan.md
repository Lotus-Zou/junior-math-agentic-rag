# 初中数学错题系统 Skill Pipeline 重构计划

## 1. 重构目标

把当前“LangGraph 节点直接调用业务函数、Tool 与 MCP 手工重复绑定”的结构，重构为同一套业务 Skill 在四种执行表面复用：

1. 固定 Workflow Pipeline。
2. LangGraph Agent Tool-Calling。
3. MCP 标准工具服务。
4. 离线 Eval/Test Harness。

重构必须保持现有行为基线：30 项自动化测试通过、1000 条课程基准全部进入确定性路径、P95 小于 100 ms、开放题 8 秒熔断、所有失败进入 bad-case 池。

## 2. 当前结构问题

- `agentic_rag/skills.py` 同时承担服务实现、LangChain Tool 包装和白名单校验，职责混合。
- Skill 没有版本、机器可读 manifest、Pydantic 输入输出契约、超时、重试、幂等性和副作用声明。
- `graph.py`、`react_agent.py` 与 `mcp_server.py` 分别手工注册函数，容易出现参数和行为漂移。
- `fast_path.py` 与 LangGraph 是两套入口，Trace、指标和错误语义需要 API 层额外拼接。
- Pipeline 固化在 Python 边连接中，无法安全做 A/B、灰度或按题型切换策略。
- Eval 只能从最终响应观察结果，无法对单个 Skill 做契约、延迟、召回和故障注入测试。

## 3. 目标目录

```text
agentic_rag/
  skill_runtime/
    __init__.py
    contracts.py          # SkillContext、SkillResult、Artifact、Citation
    manifest.py           # SkillManifest 与 YAML 校验
    registry.py           # skill_id@version 注册、发现、依赖检查
    router.py             # 意图、能力、风险与成本路由
    executor.py           # 超时、重试、并发、幂等、熔断
    pipeline.py           # Pipeline YAML 加载、DAG 校验、执行
    policies.py           # 工具白名单、数据权限、答案策略
    errors.py             # 可重试/不可重试/降级错误模型
    telemetry.py          # 每 Skill Trace、Token、延迟、缓存和结果摘要
    adapters/
      langgraph.py        # Pipeline 编译为 StateGraph
      langchain_tools.py  # Registry Skill 转 Tool-Calling 工具
      mcp.py              # Registry Skill 自动暴露为 MCP tool
  domain/
    models.py             # Question、StudentAttempt、KnowledgePoint 等领域对象
    schemas.py            # 统一 API/Skill Pydantic Schema
  skills/
    input_guard/
      SKILL.md
      skill.yaml
      handler.py
      tests.yaml
    question_parse/
    query_rewrite/
    knowledge_classify/
    curriculum_solve/
    retrieval_dense/
    retrieval_bm25/
    retrieval_graph/
    rrf_fusion/
    rerank_filter/
    answer_generate/
    answer_critic/
    similar_exercise/
    answer_check/
    memory_recall/
    memory_commit/
    response_render/
  pipelines/
    correction.yaml       # 错题诊断与订正
    knowledge_qa.yaml     # 定义、公式和画法解释
    exercise.yaml         # 相似题/指定章节出题
    answer_check.yaml     # 学生提交步骤后的核对
    multimodal.yaml       # 后续图片题入口，默认 feature flag 关闭
    evaluation.yaml       # 离线逐 Skill 与端到端评测
  graph.py                # 只负责从 Pipeline Adapter 构建 LangGraph
  skills.py               # 兼容层，迁移完成后删除

schemas/
  skill-manifest.schema.json
  pipeline.schema.json
  trace-event.schema.json

evaluation/
  skill_cases/
    input_guard.jsonl
    query_rewrite.jsonl
    retrieval.jsonl
    critic.jsonl
    conversation.jsonl
  pipeline_cases/
    correction.jsonl
    knowledge_qa.jsonl
    exercise.jsonl
    answer_check.jsonl
  chaos_cases.yaml
  skill_harness.py
  pipeline_harness.py

tests/
  skill_runtime/
  skills/
  pipelines/
  contracts/

docs/
  Skill_Pipeline_Architecture.md
  Skill_Authoring_Guide.md
  Skill_Pipeline_Runbook.md
  Skill_Pipeline_Refactor_Plan.md
```

## 4. Skill 包规范

每个业务 Skill 必须同时包含：

- `SKILL.md`：面向开发者与 Agent 的用途、适用边界、禁止事项和示例。
- `skill.yaml`：机器可读 manifest。
- `handler.py`：纯业务实现，不依赖 FastAPI、MCP 或 LangGraph。
- `tests.yaml`：正常、边界、对抗、超时和降级用例。

建议 manifest：

```yaml
id: math.query_rewrite
version: 1.0.0
description: 将学生口语问题改写并分解为数学检索 Query
input_schema: agentic_rag.domain.schemas.QueryRewriteInput
output_schema: agentic_rag.domain.schemas.QueryRewriteOutput
timeout_ms: 1500
max_attempts: 1
idempotent: true
side_effects: []
required_capabilities: [llm.optional]
fallback: math.query_rewrite.rules@1
policies: [junior_math_scope, no_condition_fabrication]
evaluator: query_rewrite_v1
```

规则：

- Skill 输入输出只能使用领域 Schema，禁止传递任意 `dict`。
- Skill 不得直接写 Trace、缓存或数据库；由 Executor 统一处理。
- Skill 不得自行递归调用 Agent；依赖通过 Registry 显式声明。
- 有外部副作用的 Skill 必须声明权限、幂等键和补偿策略。
- 模型 Skill 必须提供确定性或无模型降级，或者明确返回 `UNSUPPORTED`。

## 5. Pipeline 规范

Pipeline YAML 只描述控制流，不放 Prompt 和业务代码：

```yaml
id: math.correction
version: 1.0.0
sla_ms: 8000
entry: input_guard
nodes:
  input_guard:
    skill: math.input_guard@1
    next: parse
  parse:
    skill: math.question_parse@1
    next: route
  route:
    type: router
    branches:
      deterministic: curriculum_solve
      rag: rewrite
      clarify: response_render
  curriculum_solve:
    skill: math.curriculum_solve@1
    next: answer_critic
  rewrite:
    skill: math.query_rewrite@1
    next: retrieval_parallel
  retrieval_parallel:
    type: parallel
    skills: [math.retrieve_dense@1, math.retrieve_bm25@1, math.retrieve_graph@1]
    next: rrf_fusion
  rrf_fusion:
    skill: math.rrf_fusion@1
    next: rerank
  rerank:
    skill: math.rerank_filter@1
    next: answer_generate
  answer_generate:
    skill: math.answer_generate@1
    next: answer_critic
  answer_critic:
    skill: math.answer_critic@1
    on:
      pass: response_render
      retryable_fail: rewrite
      fail: response_render
  response_render:
    skill: math.response_render@1
    next: END
```

编译前必须检查：无环或仅允许有界重试环、每个节点存在、Schema 可连接、总超时不超过 Pipeline SLA、并行节点无冲突副作用、所有失败分支都有终点。

## 6. 首批 Skill 清单

| Skill ID | 来源迁移 | 关键输出 |
|---|---|---|
| `math.input_guard` | `guardrails.py`、`app.py` | 规范输入、安全判定 |
| `math.question_parse` | `nodes.parse_question_node` | 题干、错误作答、意图 |
| `math.query_rewrite` | `nodes.rewrite_query_node` | 改写 Query、子 Query、缺失条件 |
| `math.knowledge_classify` | `math_taxonomy.py` | 年级、章节、知识点、题型 |
| `math.curriculum_solve` | `deterministic_tutor.py`、`math_validation.py` | 本地解答、证明工件 |
| `math.retrieve_dense` | `math_retriever.py` | 稠密候选与分数 |
| `math.retrieve_bm25` | `math_retriever.py` | 关键词候选与分数 |
| `math.retrieve_graph` | `knowledge_graph.py` | 依赖知识点与关联候选 |
| `math.rrf_fusion` | `math_retriever.py` | 去重融合候选 |
| `math.rerank_filter` | `nodes.rerank_documents_node` | Top-K 与拒绝原因 |
| `math.answer_generate` | `nodes.generate_response_node` | 带引用草稿 |
| `math.answer_critic` | `nodes.validate_answer_node` | 事实/逻辑/引用报告 |
| `math.similar_exercise` | `fast_path.py`、`skills.py` | 练习、提示、隐藏答案状态 |
| `math.answer_check` | `fast_path.py` | 学生步骤核对与首错定位 |
| `math.memory_recall` | `memory.py` | 脱敏薄弱点 |
| `math.memory_commit` | `memory.py` | 带租户和 TTL 的记忆事件 |
| `math.response_render` | API 响应拼装 | 统一 AnswerEnvelope |

## 7. 统一执行契约

`SkillContext` 至少包含：

- `request_id`、`trace_id`、`session_id`、`tenant_id`。
- `language`、`deadline_at`、`remaining_budget_ms`。
- `question`、`student_attempt`、`conversation`。
- `policy_set`、`feature_flags`、`model_profile`。

`SkillResult[T]` 至少包含：

- `status`: `OK | CLARIFY | UNSUPPORTED | RETRYABLE_ERROR | FATAL_ERROR`。
- `value`: 强类型业务结果。
- `artifacts`: 检索片段、方程证明、引用、Critic 报告。
- `metrics`: 延迟、Token、缓存、模型、工具次数。
- `provenance`: skill id/version、代码版本、数据版本。
- `safe_error`: 可直接展示给学生的错误，不包含上游异常细节。

## 8. LangGraph、Tool-Calling 与 MCP

- `graph.py` 不再手工枚举节点，只调用 `LangGraphPipelineAdapter.compile("math.correction@1")`。
- `react_agent.py` 从 Registry 按 capability 和 policy 获取可用 Tool，禁止维护第二份列表。
- `mcp_server.py` 通过 `MCPRegistryAdapter` 暴露 manifest 中 `expose.mcp: true` 的 Skill。
- Pydantic Schema 自动转换为 MCP/Tool JSON Schema，参数变化由契约测试阻止。
- MCP、Workflow 和 Agent 调用相同 Executor，因此超时、Trace、权限与错误语义一致。

## 9. Trace 与评测

每次 Skill 调用写一条结构化事件：

```json
{
  "trace_id": "...",
  "pipeline": "math.correction@1.0.0",
  "skill": "math.rerank_filter@1.0.0",
  "status": "OK",
  "latency_ms": 38.2,
  "cache": "miss",
  "input_hash": "...",
  "artifact_ids": ["chunk-12"],
  "policy_decisions": ["junior_math_scope:pass"]
}
```

质量门禁分四层：

1. Skill 单测：数学正确性、Schema、异常和语言。
2. Contract Test：Manifest、版本、MCP/Tool Schema 一致性。
3. Pipeline Test：分支、重试、超时、降级、会话连续性。
4. Product Eval：1000 条基准、真实 bad case、RAGAS、P95、Token 和成本。

故障注入至少覆盖模型超时、向量库不可用、BM25 空召回、Graph 数据损坏、Critic 拒绝、Redis 不可用、SQLite 锁冲突和 MCP 参数错误。

## 10. 实施阶段

### 阶段 0：冻结基线

- 保存当前测试、1000 条门禁报告、关键 API 响应和延迟。
- 为现有 `skills.py` 四个工具生成 Schema 快照。
- 门禁：不得出现行为变更。

### 阶段 1：契约与 Runtime 骨架

- 新增 `contracts.py`、`manifest.py`、`registry.py`、`errors.py`。
- 先实现内存 Registry，不接 LangGraph。
- 门禁：Manifest 非法、版本冲突、Schema 不兼容均有测试。

### 阶段 2：Executor 与 Telemetry

- 实现统一超时、重试、熔断、缓存、策略和 Trace。
- 将现有快速路径先包装为 `math.curriculum_solve`。
- 门禁：确定性题 P95 不回退，快速路径 Trace 与现有兼容。

### 阶段 3：检索 Skill 拆分

- 把 Dense、BM25、Graph、RRF、Rerank 从单体 Retriever 拆为独立 Skill。
- 并行执行三路召回，RRF 只消费标准候选 Schema。
- 门禁：Recall@K 不下降，任何一路故障仍可降级。

### 阶段 4：生成与 Critic

- 迁移 query rewrite、answer generate、answer critic。
- Critic 输入必须包含题设、证据、草稿和本地证明工件。
- 门禁：生成与 Critic 不共享实例状态；旧题答案污染回归继续通过。

### 阶段 5：配置化 Pipeline

- 新增 correction、knowledge_qa、exercise、answer_check YAML。
- 用 Adapter 编译 LangGraph，逐条替换现有硬编码边。
- 门禁：新旧 Pipeline 双跑，对响应语义和 Trace 做差异报告。

### 阶段 6：自动 Tool/MCP 暴露

- Registry 生成 LangChain Tool 与 MCP Tool。
- 删除 `skills.py` 和 `mcp_server.py` 中的重复手工签名。
- 门禁：Tool/MCP Schema 快照完全一致。

### 阶段 7：产品阻断项

- 增加 tenant/session 身份、记忆 TTL、导出与删除。
- 引入图片 OCR、公式识别、几何图关系 Skill。
- 引入脱敏真实学生错题集与数据版本管理。
- 门禁：完成安全评审、隐私评审、备份恢复和真实流量灰度。

## 11. 迁移原则

- 使用 Strangler 方式逐 Skill 替换，不一次性重写 LangGraph。
- 每迁移一个 Skill，先写失败测试，再实现，再运行完整门禁。
- 旧接口保留兼容层，只有新旧双跑稳定后才删除。
- Prompt、模型和检索策略版本必须进入 Trace，禁止隐式修改。
- Pipeline 配置变更视为代码变更，必须评审和回归。
- 不把 Codex 全局安装的 Superpowers/Anthropic 技能直接作为线上业务依赖；它们用于开发流程，线上业务 Skill 必须随仓库版本化。

## 12. 推荐下一轮执行方式

下一轮启用新安装的 Superpowers 后，按以下技能顺序执行：

1. `brainstorming`：审查本计划的边界和关键取舍。
2. `writing-plans`：把阶段 0-2 拆成可提交的小任务。
3. `test-driven-development`：先建立 Manifest/Registry 失败测试。
4. `systematic-debugging`：处理迁移中的行为差异。
5. `verification-before-completion`：每阶段运行完整门禁并留证。
6. `requesting-code-review`：阶段合并前做独立评审。

Anthropic 技能建议用于 `mcp-builder`、`webapp-testing`、`frontend-design` 和文档交付；不得让同名或外部 Skill 绕过项目自己的 Registry、Policy 与 Eval。
