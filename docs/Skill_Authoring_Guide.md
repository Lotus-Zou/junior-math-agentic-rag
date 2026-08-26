# Skill 编写指南

每个 Skill 目录必须包含 `SKILL.md`、`skill.yaml`、`handler.py` 和 `tests.yaml`。可复用脚本放入本 Skill 的 `scripts/`，大型说明放 `references/`。

## Manifest

- `id` 使用点分小写标识，版本使用完整 SemVer。
- 输入输出引用 `agentic_rag.domain.schemas` 中的 Pydantic 模型。
- 显式声明超时、尝试次数、幂等性、副作用、能力、依赖与策略。
- Tool/MCP 暴露由 `expose` 控制，禁止在适配器中手写第二份参数。

## Handler

签名为 `handle(input_model, skill_context) -> output_model`。Handler 只实现业务能力，不写 Trace、缓存或数据库，不递归启动 Agent，也不捕获异常后伪造正常结果。数学答案不得补造条件，引用必须映射到真实检索工件。

## 测试

至少覆盖正常、边界、对抗、超时和降级。修改输出语义时提升版本，并增加旧版/新版差异用例。

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'D:\Program Files\Anaconda\python.exe' -m pytest -q tests\skill_runtime tests\contracts
& 'D:\Program Files\Anaconda\python.exe' evaluation\skill_harness.py --report
```

