# 运行指南

## Docker Compose

准备 Docker Engine、Compose 插件与可访问的 OpenAI 兼容模型服务。
Dense 检索默认使用 `BAAI/bge-m3`，首次建立索引会加载模型，需预留下载和运行空间。

```bash
cp .env.example .env
```

填写下列配置后启动；`.env` 只保留在本机。

| 配置 | 用途 |
| --- | --- |
| `OPENAI_API_BASE`、`OPENAI_API_KEY` | 模型服务地址与访问凭证 |
| `MODEL_WIRE_API` | 按供应商选择 `chat_completions` 或 `responses` |
| `LLM_MODEL_NAME`、`CRITIC_MODEL_NAME`、`RERANK_MODEL_NAME` | 服务实际提供的模型名称 |
| `AUTH_SHARED_SECRET` | 网关与 Agent 共用的随机密钥，可用 `openssl rand -hex 32` 生成 |
| `MATH_EMBEDDING_MODEL_PATH` | 本地 embedding 模型路径或模型标识 |

容器访问宿主机模型服务时，填写容器可达地址，而不是容器内部的 `localhost`。
请按供应商说明配置出口、模型访问和费用限额。

```bash
docker compose up -d mysql redis
docker compose run --rm --no-deps app python ingest.py
docker compose up --build -d
docker compose ps
```

打开 <http://localhost:8080/>。网关使用 gRPC 调用 Python 服务，MySQL 初始化时执行
`sql/schema.sql` 和 `sql/002_domain_model.sql`。已有数据库升级可运行
`tools/migrate_mysql.py`。知识索引更新时停止读取旧索引的服务，完成重建后再启动。

```bash
docker compose logs --tail=100 gateway app grpc worker
docker compose down
```

## 原生 Linux 隔离运行

提供不占用默认服务端口的本机管理器。准备 Linux x86_64、Python、Node 22、Go、
Ollama、`dpkg-deb` 及模型运行所需硬件，随后执行：

```bash
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python tools/download_reproduction_resources.py --mysql --native-deps
python tools/reproduction_stack.py dependencies
python tools/download_reproduction_resources.py --models
python tools/reproduction_stack.py run -- python ingest.py
python tools/reproduction_stack.py start
python tools/reproduction_stack.py status
```

页面位于 <http://127.0.0.1:28080/>，MySQL/Redis/Ollama 使用独立回环端口。
`REPRO_GO` 与 `REPRO_REDIS_BINARY` 可覆盖本机可执行文件路径。运行状态和下载资源
存入忽略的 `.runtime/`，其中包含密钥，不应加入版本控制。

```bash
python tools/reproduction_stack.py run -- python tools/reproduce_experiments.py smoke
python tools/reproduction_stack.py stop
```

## 部署配置

- Compose 的数据库账户与单副本依赖用于本地开发，正式部署替换为受管理的凭证与数据服务。
- 浏览器演示使用匿名设备会话；公开服务配置正式账户、HTTPS、Secure Cookie 与访问控制。
- 保护 `/runtime`、`/queue`、指标和 Trace 数据，配置学生隐私、保留期限与备份恢复策略。
- 重建知识库或修改生成策略时更新 `KNOWLEDGE_INDEX_VERSION`、`ANSWER_CACHE_VERSION`。
- Kubernetes 节点、GPU runtime、镜像摘要与容量验收参阅 [部署手册](../deploy/k8s/README.md)。
