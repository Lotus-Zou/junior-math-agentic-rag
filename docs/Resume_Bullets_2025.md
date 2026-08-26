# 2027 简历项目 Bullet

> 使用边界：下述 `Recall +15%`、幻觉率 `35% -> 10%`、Answer Relevance `0.50 -> 0.78` 与“导师采纳/校企落地”来自用户提供的项目历史，当前工作区未全量复现；证据状态见 `evaluation/benchmark_history.json`。

- 面向初中数学错题订正场景，基于 2025 年 Agentic RAG 技术搭建错题智能问答系统；在模型外层构建约束、编排、独立校验和自动化评测运行脚手架，解决数学公式检索困难、推导幻觉和 Agent 执行失控问题。

- 搭建公式感知知识库入库管线，实现 280/380/700 token 自适应分块与公式保护；构建数学稠密向量 + BM25 双检索，引入轻量 GraphRAG 维护知识点依赖；通过查询改写、多 Query 分解、RRF 融合与独立 LLM-Rerank 提升带来源片段的检索质量，历史对比纯向量检索 Recall 提升 15%。

- 基于 LangGraph 实现 Workflow + Agent 混合编排与三级记忆；通过 Pydantic Schema、工具白名单、最大迭代步数和超时熔断约束运行风险；将错题解析、多路检索、相似习题和答案校验封装为 Tool-Calling/MCP Skill；引入与生成模型隔离的 Critic，结合教材忠实度与 SymPy 数学逻辑校验，历史幻觉率由 35% 降至 10%。

- 实现独立 Eval 测试脚手架，构建 1000 条覆盖代数、几何、函数和统计概率的分层标注集；集成 RAGAS 与业务指标，支持检索策略 A/B、Prompt 回归和质量门禁；用 JSONL Trace 保存 Agent 执行轨迹与 bad case，历史 Answer Relevance 从 0.50 提升至 0.78。

- 使用 FastAPI、Redis、本地 TTL 缓存、Prometheus、Dockerfile 与 Docker Compose 完成服务封装；历史方案经组内分享后被导师采纳并用于校企合作教育项目。

## 当前仓库可验证口径

- 单元测试：10/10 通过；1000 条数据集覆盖 normal、hallucination_risk、colloquial 三类。
