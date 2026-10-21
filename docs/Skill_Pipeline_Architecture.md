# Skill Pipeline 架构

系统以仓库内版本化 Skill 作为唯一业务能力定义。Workflow、ReAct Tool、MCP 与离线评测都通过同一 `SkillRegistry` 和 `SkillExecutor` 调用，避免参数、策略和错误语义漂移。

```text
HTTP / LangGraph / ReAct / MCP / Eval
                  |
             Adapters
                  |
        Registry + Executor
       / Policy / Telemetry \
    skill.yaml   Pydantic Schema
                  |
             handler.py
```

## 目录与职责

- `agentic_rag/domain/`：题目、学生作答以及每个 Skill 的强类型输入输出。
- `agentic_rag/skill_runtime/`：发现、版本、策略、预算、执行、Trace 和 Pipeline。
- `agentic_rag/skills/`：版本化业务 Skill。Handler 不依赖 FastAPI、MCP 或 LangGraph。
- `agentic_rag/pipelines/`：控制流配置，不保存 Prompt 与业务代码。
- `schemas/`：Manifest、Pipeline 和 Trace 的外部 JSON Schema。
- `evaluation/`：单 Skill、Pipeline、产品基准和故障注入资产。

## 执行语义

Registry 拒绝重复版本和缺失依赖。Executor 依次执行输入校验、策略授权、超时、有限重试、输出校验和 provenance。Telemetry 只记录输入 SHA-256。Pipeline 在运行前检查节点、可达性、无界环、并行副作用与 SLA。MCP 和 LangChain 共享一个包装输入 Schema。

## 调用路径

API 通过 `math.curriculum_solve@1` 调用教学主路径。LangGraph 保留兼容调用路径；
`build_skill_pipeline_graph()` 提供 Pipeline 编译入口。新能力通过 Skill Manifest、
结构化输入输出与独立测试接入，逐步迁移时保持公开响应合同。

## 部署关注点

- 根据模型的多模态能力配置附件解析和相应预算。
- 为租户身份、学生数据保留、导出/删除与备份设置访问控制和审计。
- 根据索引存储方式选择独立检索服务或版本化索引分发。
- 用目标环境的实际模型、请求分布和硬件验证延迟、容量与恢复行为。

