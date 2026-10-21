# Skill Pipeline 运行手册

## 本地检查

在项目虚拟环境中执行：

```bash
python -m pytest -q tests/skill_runtime tests/contracts
python evaluation/skill_harness.py --report
python evaluation/pipeline_harness.py
python evaluation/bad_case_harness.py --report
python evaluation/evaluation.py --mode validate
```

## 启动与配置

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
python mcp_server.py
```

配置 `ENABLE_TUTOR_AGENT`、`ENABLE_EXERCISE_AGENT` 选择教学与练习 Agent。
`FORCE_LLM_EVERY_TURN=true` 用于逐轮验证模型链路，并绕过最终答案缓存；缓存路径
与模型路径分开测试。时间预算以 `config.py` 及 Pipeline YAML 中的设置为准。

MCP 使用 stdio，日志写入 stderr；`/health` 用于存活检查，`/ready` 检查依赖。
完整运行环境见 [运行指南](Quickstart.md)。

## 发布检查

- Skill Manifest、输入输出 Schema、适配器与回归测试保持一致。
- YAML Pipeline 无不可达节点、无界循环和未授权副作用。
- 同时验证普通答案、澄清、拒绝、受控修复及超时返回的响应合同。
- 在目标模型与硬件上确认步骤预算、延迟、队列和恢复行为。
- 对新增数据访问、工具权限、学生数据与备份配置进行审查。

## 故障定位

| 情况 | 检查 |
| --- | --- |
| Manifest 加载失败 | 重复版本、Handler 导入、依赖 Skill 引用 |
| Pipeline 校验失败 | 节点引用、YAML 布尔关键字、可达性、循环与预算 |
| Skill 超时 | 对应 Trace、供应商耗时、线程槽占用及重试预算 |
| 检索降级 | 通道状态、索引版本、知识来源与重排输入 |
| Critic 拒绝 | 草稿、检索证据、缺陷报告及修复调用记录 |
