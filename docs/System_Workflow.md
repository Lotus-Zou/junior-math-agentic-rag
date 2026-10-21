# 系统工作流

核心图定义于 `agentic_rag/graph.py`，状态定义于 `agentic_rag/state.py`。

1. `retrieve_memory`：召回长期错因/知识缺口，并生成受预算约束的本轮工作记忆。
2. `parse_question`：结构化提取题干、学生错误作答、意图和显式错因。
3. `rewrite_query`：规范化口语问题并分解 2-4 条子 Query；题设缺失时转入 `clarify`。
4. `classify_knowledge`：识别年级、章节、题型和知识点。
5. `react_agent`：知识查询分支允许在 8 步预算内自主调用白名单 Skill；求解/错因分支进入固定 Workflow。
6. `retrieve_documents`：执行章节召回、多 Query 稠密向量 + BM25 + GraphRAG，并通过 RRF 融合。
7. `llm_rerank`：独立重排模型过滤公式条件不匹配、年级错误和低支撑力片段。
8. `generate_response`：基于编号教材片段生成引导式订正草稿。
9. `validate_answer`：独立 Critic 结合教材忠实度与 SymPy 数学逻辑检查输出缺陷报告。
10. 验证失败时携带 `validation_issues` 回到查询改写；达到次数或运行预算后拒绝输出未验证结论。
11. `finalize`：压缩会话、沉淀长期薄弱点，并持久化 JSONL Trace；失败 case 进入单独记录。
12. API 层再施加总超时、Redis/本地缓存、Prometheus 指标和用户反馈闭环。

关键状态包括结构化题目、改写 Query、子 Query、意图、章节、知识点、候选片段、重排结果、Critic 报告、三级记忆、预算、Trace、纠错次数和会话历史。所有分支条件均由 LangGraph 显式管理。
