# 架构设计

## 设计目标

围绕数学错题订正提供低耦合的服务边界：网关治理集中请求，Agent 服务处理多步骤
推理，Worker 承担异步评测与报告。各层通过结构化协议传递数据与 Trace，避免把
模型供应商接口、浏览器连接和后台任务绑定在同一个生命周期中。

## 服务边界

| 服务 | 职责 | 入口 |
| --- | --- | --- |
| Gin Gateway | HTTP/SSE、身份、限流、并发准入、幂等与熔断 | `gateway/main.go`、`gateway/routes.go` |
| Agent API | 请求规范化、上下文、缓存、答案合同与领域接口 | `app.py` |
| gRPC | `Ask` / `AskStream` 内网调用 | `proto/math_agent.proto`、`grpc_server.py` |
| Skill Runtime | 路由、Schema 校验、步骤执行、预算与遥测 | `agentic_rag/skill_runtime/` |
| Worker | 错题归档、反馈评测、学习报告与任务状态 | `worker.py`、`agentic_rag/task_queue.py` |

## 一次问答的生命周期

1. 网关校验身份、解析幂等键并进行流量准入，为请求传递 Trace Context。
2. Agent 服务加载会话与练习状态，按版本化键查询答案缓存。
3. LangGraph 路由到适合的 YAML Pipeline；独立 Skill 完成题干改写和知识点识别。
4. BM25、Dense 与 Graph 检索候选，经 RRF 与重排生成带来源的上下文。
5. 生成答案，使用数学工具与 Critic 检查；必要时在预算内修复，再按答案合同发布。
6. SSE 发送进度、模型增量、结果及结束事件；缓存或确定性结果可直接返回最终答案。
7. 错题归档、评测和报告通过 Streams 交由 Worker 处理。

## 并发与幂等

- 网关与 Agent 分别设置并发上限，排队受容量和时间预算约束，过载返回明确状态。
- 工作线程未结束时不会因调用方超时提前释放执行槽，防止实际在途任务突破上限。
- 幂等记录采用请求指纹、Redis `SET NX`、TTL 与 owner 校验。相同请求可重放已完成
  结果；处理中或请求内容不同的复用分别处理。
- 熔断与降级保留可追踪状态，模型输出遵守 `response_type` 与验证字段的响应合同。

## 数据模型与隔离

MySQL 保存用户、会话、消息、错题、反馈、学习报告与异步任务。复合索引及访问条件
共同约束用户/会话归属；连接池、事务和迁移脚本见 `agentic_rag/domain_store.py`、
`agentic_rag/memory.py` 与 `sql/`。

L1 进程内缓存和 Redis L2 答案缓存共享版本语义，键包含用户、会话、历史及练习状态。
检索缓存单独随知识版本失效。更新模型、提示词或知识库时同步递增相关版本。

## 异步投递与恢复

Redis Streams 使用消费组、任务租约和 Pending 回收实现至少一次投递。有限重试与
死信避免无限循环；owner 校验防止过期消费者覆盖新状态，归档通过任务标识去重。
数据库提交与队列确认分属不同组件，因此业务侧承担幂等，而不是依赖全局事务。

## 安全与观测

输入与检索内容分别建立信任边界，工具采用白名单、结构化参数及预算限制。学生端
仅获得公开答案字段；内部 Critic 诊断、异常详情与供应商信息保存在私有 Trace。

W3C Trace Context 关联网关、gRPC、Skill 和 Worker，Prometheus 提供延迟、队列与
执行指标。部署时保护运维端点、密钥、学生数据和备份，使用 HTTPS 与正式登录。

## 部署取舍

Compose 提供单机开发拓扑；Kustomize 提供基础部署、单 GPU 与容量任务配置。
部署到实际环境时配置持久化、数据库/缓存高可用、模型资源、探针和索引初始化。
模型推理与缓存路径分别测量，CPU 扩容策略与模型队列扩容策略分开设置。

扩展 Skill 见 [编写指南](Skill_Authoring_Guide.md)，部署步骤见
[Kubernetes 手册](../deploy/k8s/README.md)。
