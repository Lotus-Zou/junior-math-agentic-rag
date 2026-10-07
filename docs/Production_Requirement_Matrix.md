# 优化定稿版落地矩阵

| 要求 | 当前实现 | 证据 |
|---|---|---|
| Go 高并发接入层 | 已实现 | `gateway/main.go`：令牌桶限流、有界并发、幂等、熔断 |
| Python AI 推理层 | 已实现 | FastAPI + LangGraph 现有 `/ask` 链路 |
| HTTP/SSE 实时问答 | 已实现 | `POST /ask`、`POST /ask/stream`；SSE 在答案完成后发送已校验结果 |
| Hybrid Agentic RAG | 已实现 | Query Router、重写、Dense/BM25/GraphRAG、RRF、Critic |
| Redis 缓存与任务队列 | 已实现 | `agentic_rag/cache.py`、`agentic_rag/task_queue.py` |
| 错题归档/评测/报告异步解耦 | 已实现接口 | `POST /tasks` + `worker.py`；业务消费者可继续扩展 |
| MySQL 领域建模 | 已实现迁移 | `sql/schema.sql`；应用生产连接层需绑定实际 MySQL 驱动 |
| gRPC 服务边界 | 契约已固化 | `proto/math_agent.proto`；当前 Compose 默认使用 HTTP，避免未生成 stub 的伪实现 |
| Docker Compose | 已实现 | gateway、app、worker、Redis、MySQL |
| Kubernetes 自愈与弹性 | 部署骨架已实现 | `deploy/k8s/*.yaml`；镜像、Secret、Ingress 需按集群替换 |
| Trace、Prometheus、bad case | 已实现 | `/metrics`、JSONL Trace、bad case registry |

## 当前边界

当前项目可以作为完整的高并发架构骨架运行，但不能把“支撑多所初中规模化流量”“P95 数值”“正式校企落地”等内容当作本地测试结论。上线前还需要真实压测、MySQL 高可用、Redis 哨兵/集群、镜像仓库、Kubernetes Secret、Ingress/TLS 和真实多副本容量验证。
