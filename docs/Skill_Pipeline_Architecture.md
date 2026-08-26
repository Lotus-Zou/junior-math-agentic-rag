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
- `agentic_rag/skills/`：17 个业务 Skill。Handler 不依赖 FastAPI、MCP 或 LangGraph。
- `agentic_rag/pipelines/`：控制流配置，不保存 Prompt 与业务代码。
- `schemas/`：Manifest、Pipeline 和 Trace 的外部 JSON Schema。
- `evaluation/`：单 Skill、Pipeline、产品基准和故障注入资产。

## 执行语义

Registry 拒绝重复版本和缺失依赖。Executor 依次执行输入校验、策略授权、超时、有限重试、输出校验和 provenance。Telemetry 只记录输入 SHA-256。Pipeline 在运行前检查节点、可达性、无界环、并行副作用与 SLA。MCP 和 LangChain 共享一个包装输入 Schema。

## 迁移状态

确定性主路径已由 API 通过 `math.curriculum_solve@1` 调用。复杂开放题仍保留旧 LangGraph 作为兼容路径；`build_skill_pipeline_graph()` 提供新路径编译入口。只有差异报告和真实流量灰度通过后，才能删除 `skills.py` 与旧图节点。

## 尚未达到生产成熟的阻断项

- 图片 OCR、公式识别与几何图关系抽取仍仅有关闭的 `multimodal` 入口。
- 租户身份、学生隐私请求、记忆 TTL、导出/删除与备份恢复尚未完成安全评审。
- Dense/BM25/Graph 的物理索引仍由旧 Retriever 提供，下一阶段需完全拆开故障域。
- 真实脱敏流量、RAGAS 全量 Judge、容量与灾难恢复尚未完成。

