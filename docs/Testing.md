# 测试与评测

## 工程回归

[GitHub Actions](../.github/workflows/engineering.yml) 包含三个独立任务：

| 任务 | 检查 |
| --- | --- |
| Python | 构建离线合同测试索引、审计数据集、pytest、上传 JUnit |
| Go | 使用真实 Redis 测试幂等与并发行为，执行 race 检查并构建网关镜像 |
| 前端 | 安装锁定依赖、测试、TypeScript 检查、生产及独立展示构建 |

归档的 [Python JUnit](validation/python-engineering.xml) 来自工程 CI，包含 1178 项
测试及 24 项子测试，共 1202 项通过。测试覆盖响应合同、用户/会话隔离、校验修复、
队列租约、幂等、副作用去重、缓存和失败恢复。后续代码以实时 CI 状态为准。

离线工程测试使用词法合同测试索引，Dense 模型检索由独立评测命令执行。

```bash
python -m pip install -r requirements.txt
CHROMA_PATH=.runtime/contract-test-index ENABLE_DENSE_RETRIEVAL=false python tools/bootstrap_local_index.py
CHROMA_PATH=.runtime/contract-test-index ENABLE_DENSE_RETRIEVAL=false python -m pytest -q
cd gateway
go test -race -count=1 ./...
cd ../frontend
npm ci
npm test
npm run typecheck
npm run build
```

Redis 集成测试通过 `QUEUE_TEST_REDIS_URL` / `GATEWAY_TEST_REDIS_URL` 指定专用
测试库，避免与开发业务数据共用。参阅 CI 中的服务配置。

## 在线展示验收

[Pages 工作流](../.github/workflows/pages.yml) 在部署前运行前端测试、独立展示构建和
Playwright 桌面（1440 × 1000）/手机（390 × 844）浏览器验收，留存截图与 JSON 结果。
检查示例问答、追问、反馈、错题归档、报告下载、刷新恢复、会话隔离、主题和语言切换。
同时检查控制台、资源加载、横向溢出与意外后端请求；任一失败均阻止部署。

本地构建后，以 showcase 模式在 28181 端口启动 Vite preview，再从仓库根目录运行：

```bash
python -m pip install 'playwright>=1.58,<2'
python -m playwright install chromium
python tools/verify_showcase_browser.py
```

展示版只提供标识清楚的自编示例，不调用模型或伪造检索、校验及模型耗时。
该验收只证明静态作品的交互流程，不计入数学模型质量或真实服务容量指标。

## 数据集

`evaluation/math_benchmark_1000.csv` 包含 1000 条底层题干去重的程序化标注题，覆盖
七至九年级、10 个知识点，含普通题、错答诊断及口语化提问。标签、教材来源摘要和
生成器版本通过 [数据审计](validation/dataset-audit.json) 的 SHA-256 关联。

```bash
python -m evaluation.dataset_audit --output evaluation/reports/dataset-audit.json
python evaluation/evaluation.py --mode validate
```

程序化回归集、公开数学题集与知识问答 RAGAS 数据分别使用，不合并评分。

## 质量指标

| 口径 | 指标与用途 |
| --- | --- |
| 检索 | Recall@K、Precision、知识点匹配及配对分析 |
| RAGAS | Answer Relevance、Faithfulness、Context Precision/Recall |
| 答案 | 公开题参考答案匹配、业务完成率、解析与错因指标 |
| 接口 | 业务吞吐、P95/P99、拒绝率、SSE 完整性与首增量延迟 |

工程测试通过数不作为数学正确率；内部校验字段不代替外部参考答案。完整评测命令
见 [评测工具说明](../evaluation/README.md)。

## 集成与容量

```bash
python tools/reproduction_stack.py run -- python tools/reproduce_experiments.py smoke
python tools/reproduction_stack.py run -- python tools/verify_reproduction_traces.py
python tools/reproduction_stack.py run -- python tools/evening_capacity_test.py \
  --url http://127.0.0.1:28080/ask \
  --steady-seconds 900 --burst-seconds 180 \
  --steady-concurrency 50 --burst-concurrency 300 \
  --min-steady-rps 1 --max-p95-ms 20000 \
  --output evaluation/reports/capacity-local.json
```

容量报告记录题单和源码摘要、并发、时长、缓存命中、重放、非缓存模型尝试数及
各项 SLO。确定性/缓存路径与模型推理路径分别解释，记录原始门禁结论。测量期间
固定配置并单独执行；完整 GPU/集群验证使用部署手册中的目标环境流程。
