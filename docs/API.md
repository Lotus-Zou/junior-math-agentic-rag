# 接口说明

浏览器通过 Gin 网关调用服务。开发网关默认监听 `8080`，Python 原生接口默认监听
`8000`，gRPC 协议定义位于 `proto/math_agent.proto`。

## 身份与请求上下文

浏览器先调用 `GET /auth/session` 获取匿名设备身份，网关设置 HttpOnly Cookie。
后续请求携带 Cookie 与 `X-Session-ID`；网关生成签名转发给内部服务。
服务间调用可使用 `X-User-ID`、`X-Session-ID`、`X-User-Signature`，签名为共享密钥
对用户 ID 的 HMAC-SHA256。共享密钥不得放入前端或移动端。

`Idempotency-Key` 标识可重试请求，`X-Request-ID` 与标准 `traceparent` 关联调用链。
同一幂等键应对应同一用户、会话和请求内容。

## 学生端接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/auth/session` | 建立或恢复浏览器设备会话 |
| POST | `/ask` | 一次问答或追问 |
| POST | `/ask/stream` | SSE 流式问答 |
| POST | `/attachments/parse` | 题目附件解析 |
| POST | `/feedback` | 答案反馈 |
| GET | `/mistakes` | 当前用户与会话的错题列表 |
| GET | `/reports/learning` | 当前学习情况汇总 |
| POST | `/tasks` | 提交异步评测、报告或归档任务 |
| GET | `/tasks/{task_id}` | 查询归属当前用户/会话的任务 |
| GET | `/memory` | 查询当前学习记忆 |
| DELETE | `/memory/{memory_id}` | 删除当前用户的指定记忆 |

问答请求示例，`user_id` 使用会话接口返回值：

```json
{
  "request_id": "math-question-001",
  "user_id": "<session-user-id>",
  "session_id": "algebra-session",
  "query": "解方程 2(x-3)=8，说明每一步的依据。",
  "language": "zh",
  "conversation_history": []
}
```

答案包含 `answer`、`response_type`、`validation_passed`、`sources` 等字段。
调用方按响应类型区分完整答案、澄清、拒绝与一般回答，而不是仅检查 HTTP 状态码。
具体模型与请求字段见 `app.py` 中的 Pydantic 定义及
`agentic_rag/domain/schemas.py`。

## SSE 与异常

SSE 使用命名事件传递进度、模型增量、最终 `result`、错误和 `done`。消费者同时
检查最终结果及结束事件，断开连接时保持幂等键语义，不将不完整流直接作为成功结果。

| 状态 | 调用方处理 |
| --- | --- |
| 401 | 重新建立身份或检查签名 |
| 404 | 检查资源归属或任务标识 |
| 409 | 检查幂等冲突或请求处理中状态 |
| 429 | 按 `Retry-After` 与退避策略重试 |
| 503 / 504 | 保留请求标识，按业务预算处理上游不可用或超时 |

## 运维接口

`/health` 为存活检查，`/ready` 检查服务依赖，`/metrics` 输出指标。
`/runtime`、`/queue` 等诊断接口按服务层提供；生产 ingress 应限制运维接口，指标与
Trace 仅向受信任的监控系统开放。完整 gRPC 消息与流式定义以 `.proto` 文件为准。
