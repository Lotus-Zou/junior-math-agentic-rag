# Skill Pipeline 运行手册

## 本地检查

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Program Files\Anaconda\python.exe' -m pytest -q tests\skill_runtime tests\contracts
& 'D:\Program Files\Anaconda\python.exe' evaluation\skill_harness.py --report
& 'D:\Program Files\Anaconda\python.exe' evaluation\pipeline_harness.py
python evaluation\bad_case_harness.py --report
python evaluation\evaluation.py --mode validate
```

当前机器的系统 Python 3.14 与旧 `typeguard` pytest 插件不兼容，应关闭第三方 pytest 自动加载或使用项目虚拟环境。

## 启动

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
python mcp_server.py
```

生产环境保持 `ENABLE_TUTOR_AGENT=true`、`ENABLE_EXERCISE_AGENT=true` 和 `FORCE_LLM_EVERY_TURN=true`。前两项保证确定性题与练习题调用对应 Agent，后一项禁止最终答案缓存绕过模型。测试通过 `tests/conftest.py` 关闭外部调用，只在专门 Agent 测试中使用假模型。

MCP 使用 stdio，日志不能写到 stdout。`/health` 用于存活检查，`/ready` 展示依赖状态。

## 发布门禁

- 旧测试与新增契约测试全绿。
- 1000 条确定性数学内核覆盖保持 100%，P95 小于 100 ms；生产 Workflow 另验 Tutor Agent 与 Critic 调用次数。
- 外部 Agent 以 45 秒为常规目标、180 秒为准确性优先硬上限；模型失败时转入有教材来源的 grounded fallback，并记录 bad case。
- MCP 与 LangChain 输入 Schema 一致。
- 新旧 Pipeline 差异仅包含已评审字段。
- 未完成的隐私、真实数据与灾备阻断项不得标记为生产完成。

## 故障处理

- Manifest 失败：检查重复版本、导入路径和依赖引用。
- Pipeline 失败：检查不可达节点、YAML 中需加引号的 `"on"`、环与 SLA。
- Skill 超时：先查该 Skill Trace；当前单模型 45 秒、Tutor 100 秒、全局 180 秒，调整时必须同步 Pipeline、前端和回归测试。
- 检索失败：保留成功通道并禁止无证据生成。
- Critic 拒绝：保存草稿、证据和缺陷报告进入 bad-case 池。
