# C 端高并发架构落地

系统采用 `Go Gateway → Python AI Service → Redis/Chroma` 的分层部署。网关只负责接入和流量治理，Python 服务负责 LangGraph Agentic RAG，后置任务通过 `/tasks` 入队。

## 运行

```powershell
docker compose up --build
```

用户端入口为 `http://localhost:8080`，开发调试仍可直接访问 `http://localhost:8000`。网关提供 `/health`、`/ready` 和 `POST /ask`，支持 `Idempotency-Key`、有界并发、令牌桶限流、45 秒上游超时和 5 次失败/10 秒熔断。

实时问答只返回模型结果；错题归档、知识点评测和学习报告调用 `POST /tasks`，避免阻塞 SSE/问答链路。MySQL 负责用户、错题、会话、反馈和任务元数据，Redis 负责缓存与队列，`sql/schema.sql` 为初始化迁移脚本。

生产环境应把 `MYSQL_*`、`REDIS_URL` 和模型密钥放入 Secret，使用 Kubernetes HPA 按网关并发数和 AI 服务 CPU 扩缩容，并将 `/metrics` 接入 Prometheus。网关内存幂等表仅用于单实例开发；多副本部署时把幂等键迁移到 Redis。
