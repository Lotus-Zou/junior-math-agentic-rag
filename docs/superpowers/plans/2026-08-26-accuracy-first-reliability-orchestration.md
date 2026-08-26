# Accuracy-First Reliability Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace user-visible model, retrieval, and Critic failures with verified partial results or precise clarification while preserving diagnostics.

**Architecture:** Add a deterministic completeness analyzer and a central failure-to-response policy. LangGraph nodes and FastAPI report internal failure kinds to this policy, which returns only an approved response type and never promotes an unvalidated draft.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, Pydantic 2, Trace/bad-case infrastructure, pytest, Playwright

**Spec:** `docs/superpowers/specs/2026-08-26-accuracy-first-adaptive-tutor-design.md`

## Global Constraints

- Default external reasoning budget is configurable and defaults to 30 seconds.
- Budget expiry never bypasses validation.
- User copy omits providers, retries, Trace, bad cases, internal services, and timeout seconds.
- Every internal failure records category, issues, timing, and final fallback type.

---

### Task 1: Deterministic completeness analysis

**Files:**
- Create: `agentic_rag/completeness.py`
- Modify: `agentic_rag/domain/schemas.py`
- Test: `tests/test_completeness.py`

**Interfaces:**
- Produces: `CompletenessResult(status, missing, follow_up)` and `analyze_completeness(query, language, has_image=False)`.
- Consumes: normalized request text; makes no external calls.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("query", ["这题怎么做", "不会", "怎么解"])
def test_deictic_request_needs_full_problem(query):
    result = analyze_completeness(query, "zh")
    assert result.status == "missing_conditions"
    assert result.missing == ["完整题干"]
    assert "完整题目" in result.follow_up


def test_missing_diagram_relations_are_named():
    result = analyze_completeness("如图，在△ABC中求∠A", "zh", has_image=False)
    assert result.status == "requires_image"
    assert "图中" in result.follow_up


@pytest.mark.parametrize("query", ["解方程 2x+3=11", "一次函数 y=-2x+3 的斜率是什么"])
def test_complete_problem_continues(query):
    assert analyze_completeness(query, "zh").status == "complete"
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_completeness.py
```

Expected: completeness module is absent.

- [ ] **Step 3: Implement conservative analysis**

Use exact short-deictic markers, diagram-reference detection, target detection, and math-signal detection. Mark incomplete only when the rule names a missing input. Unknown full text remains `complete` and proceeds to RAG.

```python
class CompletenessResult(StrictModel):
    status: Literal["complete", "missing_conditions", "requires_image", "out_of_scope"]
    missing: list[str] = Field(default_factory=list)
    follow_up: str = ""
```

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add agentic_rag/completeness.py agentic_rag/domain/schemas.py tests/test_completeness.py
git commit -m "feat: identify missing math problem conditions"
```

---

### Task 2: Central failure policy

**Files:**
- Create: `agentic_rag/reliability.py`
- Modify: `agentic_rag/response_contract.py`
- Test: `tests/test_reliability_policy.py`

**Interfaces:**
- Produces: `FailureKind` and `resolve_failure(query, language, history, summary, failure_kind, issues, verified_partial=None)`.
- Consumes: completeness analysis and response constructors.

- [ ] **Step 1: Write failing policy tests**

```python
FORBIDDEN = ("复杂推理服务", "超时", "bad case", "模型", "重试", "Trace")


@pytest.mark.parametrize("kind", ["timeout", "runtime_error", "retrieval_empty", "critic_rejected"])
def test_internal_failure_becomes_actionable_clarification(kind):
    result = resolve_failure("证明这个几何结论", "zh", [], "", kind, ["internal detail"])
    assert result["response_type"] == "clarification_required"
    assert all(token not in result["answer"] for token in FORBIDDEN)
    assert result["metrics"]["internal_failure_kind"] == kind


def test_verified_partial_is_preserved_but_not_promoted():
    result = resolve_failure(
        "求解并证明", "zh", [], "", "critic_rejected", ["证明未验证"],
        verified_partial="已确定 x = 4；证明部分还缺少图形条件。",
    )
    assert result["response_type"] == "clarification_required"
    assert "已确定 x = 4" in result["answer"]
    assert "证明部分" in result["answer"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_reliability_policy.py
```

Expected: reliability module is absent.

- [ ] **Step 3: Implement a single policy table**

```python
FailureKind = Literal["timeout", "runtime_error", "retrieval_empty", "critic_rejected", "expired_exercise", "cache_error"]
```

Use completeness results when they identify a concrete gap. Otherwise ask for the complete stem, diagram relations, attempted step, or target knowledge point based on query features. Keep `failure_kind` only in metrics/Trace; never interpolate internal issues into the answer.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: every failure kind returns approved, safe copy.

- [ ] **Step 5: Commit**

```powershell
git add agentic_rag/reliability.py agentic_rag/response_contract.py tests/test_reliability_policy.py
git commit -m "feat: centralize safe reasoning fallbacks"
```

---

### Task 3: LangGraph reliability routing

**Files:**
- Modify: `agentic_rag/state.py`
- Modify: `agentic_rag/nodes.py`
- Modify: `agentic_rag/graph.py`
- Modify: `config.py`
- Test: `tests/test_reliability_graph.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: graph state with `response_type`, `clarification`, `internal_failure_kind`, and validated response text.

- [ ] **Step 1: Write failing node tests**

```python
def base_state(**overrides):
    state = {
        "query": "", "response_language": "zh", "conversation_history": [],
        "conversation_summary": "", "trace_events": [], "step_count": 0,
        "deadline_at": time.time() + 30, "metrics": {}, "validation_issues": [],
    }
    return state | overrides


def test_no_evidence_is_a_neutral_clarification():
    result = no_evidence_response_node(base_state(query="证明两三角形全等"))
    assert result["response_type"] == "clarification_required"
    assert "知识库没有召回" not in result["response"]


def test_rejected_draft_is_never_returned():
    result = validation_failure_response_node(base_state(
        query="证明两三角形全等",
        draft_response="未经验证的结论：两个三角形全等",
        validation_issues=["缺少夹角条件"],
    ))
    assert result["response_type"] == "clarification_required"
    assert "未经验证的结论" not in result["response"]
    assert "夹角条件" in result["response"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_reliability_graph.py
```

Expected: current nodes contain implementation copy and no response type.

- [ ] **Step 3: Add completeness node**

Place `analyze_completeness_node` after parsing and before retrieval. Only named missing inputs route to `clarify`; complete inputs continue. Add these fields to `AgentState`:

```python
response_type: str
clarification: dict[str, Any] | None
internal_failure_kind: str
verified_partial: str
```

- [ ] **Step 4: Replace node-specific failure copy**

Make `no_evidence_response_node`, `validation_failure_response_node`, and `clarification_response_node` call the central policy/constructors. Preserve validation issues and failure metrics in state and Trace.

- [ ] **Step 5: Configure bounded correction**

```python
RUN_TIMEOUT_SECONDS = float(os.getenv("RUN_TIMEOUT_SECONDS", "30"))
MAX_CORRECTION_ATTEMPTS = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "1"))
```

One correction cycle is allowed. Individual model-call timeouts remain independently configurable.

- [ ] **Step 6: Verify GREEN**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_reliability_graph.py tests\test_math_pipeline.py
```

Expected: all selected tests pass; rejected drafts are absent from responses.

- [ ] **Step 7: Commit**

```powershell
git add agentic_rag/state.py agentic_rag/nodes.py agentic_rag/graph.py config.py tests/test_reliability_graph.py tests/test_math_pipeline.py
git commit -m "refactor: route graph failures through reliability policy"
```

---

### Task 4: API and frontend failure handling

**Files:**
- Modify: `app.py`
- Modify: `static/app.js`
- Test: `tests/test_api_failures.py`
- Modify: `evaluation/webapp_smoke.cjs`

**Interfaces:**
- Consumes: central reliability policy.
- Produces: HTTP 200 teaching responses for in-scope processing failures and neutral client network copy.

- [ ] **Step 1: Write failing fault-injection tests**

```python
class SlowGraph:
    def invoke(self, *_args, **_kwargs):
        time.sleep(0.05)
        return {}


class BrokenGraph:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("provider secret and stack detail")


@pytest.mark.parametrize("graph", [SlowGraph(), BrokenGraph()])
def test_runtime_failure_returns_safe_clarification(monkeypatch, graph):
    monkeypatch.setattr(app_module, "_run_curriculum_skill", lambda _request: None)
    monkeypatch.setattr(app_module, "get_graph", lambda: graph)
    monkeypatch.setattr(app_module, "RUN_TIMEOUT_SECONDS", 0.01)
    response = TestClient(app_module.app).post("/ask", json={"query": "证明这两个三角形全等", "language": "zh"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["response_type"] == "clarification_required"
    assert "复杂推理服务" not in payload["answer"]
    assert "provider" not in payload["answer"]
    assert payload["trace_id"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_api_failures.py
```

Expected: timeout returns old copy and runtime exception returns HTTP 500.

- [ ] **Step 3: Convert exceptions through reliability policy**

Call `_record_failure`, then return `resolve_failure(...)`. Security guard failures return `supported_refusal`; malformed schemas stay HTTP 422. Do not send exception strings to the response.

- [ ] **Step 4: Align frontend budget and copy**

Set browser abort to 35 seconds, exceeding the server's 30-second default. Replace abort copy with a neutral connectivity message that preserves the user's typed problem for retry and does not claim validation.

- [ ] **Step 5: Verify API and browser**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_api_failures.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' --check static\app.js
$env:NODE_PATH='C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' evaluation\webapp_smoke.cjs
```

Expected: tests and browser smoke pass without forbidden copy.

- [ ] **Step 6: Commit**

```powershell
git add app.py static/app.js tests/test_api_failures.py evaluation/webapp_smoke.cjs
git commit -m "fix: hide internal reasoning failures from students"
```

---

### Task 5: Chaos and bad-case gate

**Files:**
- Modify: `evaluation/chaos_cases.yaml`
- Modify: `evaluation/bad_case_harness.py`
- Modify: `evaluation/bad_case_registry.json`
- Modify: `evaluation/README.md`
- Test: `tests/test_failure_copy.py`

**Interfaces:**
- Produces: executable failure cases and forbidden-copy gate.

- [ ] **Step 1: Write failing copy test**

```python
FORBIDDEN_USER_COPY = (
    "复杂推理服务", "未在 8 秒内完成", "系统已自动记录为 bad case",
    "知识库没有召回", "Critic 服务异常", "RuntimeError",
)


def build_all_failure_responses():
    return [
        resolve_failure("证明这个结论", "zh", [], "", kind, ["internal"])
        for kind in ("timeout", "runtime_error", "retrieval_empty", "critic_rejected", "expired_exercise", "cache_error")
    ]


def test_public_fallbacks_exclude_internal_copy():
    responses = build_all_failure_responses()
    assert responses
    for response in responses:
        assert all(token not in response["answer"] for token in FORBIDDEN_USER_COPY)
        assert response["response_type"] in {"clarification_required", "supported_refusal"}
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_failure_copy.py
```

Expected: current timeout, no-evidence, and Critic copy violates the gate.

- [ ] **Step 3: Add executable chaos cases**

Cover model timeout, provider exception, empty retrieval, Critic rejection, expired exercise state, and cache exception. Assert approved response type, actionable answer, trace ID, and absence of forbidden text.

- [ ] **Step 4: Run complete gate**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' evaluation\bad_case_harness.py --report
& '.\.venv\Scripts\python.exe' evaluation\exercise_quality_harness.py --report
& '.\.venv\Scripts\python.exe' evaluation\skill_harness.py --report
& '.\.venv\Scripts\python.exe' evaluation\pipeline_harness.py
```

Expected: every command exits 0 with zero failures.

- [ ] **Step 5: Live verification**

Restart only the confirmed AgentiRAG listener. Test `几何`, a deterministic equation, an incomplete diagram problem, and an injected reasoning failure. Confirm approved response types and no forbidden text.

- [ ] **Step 6: Register and commit**

Mark the user-visible timeout wording bad case resolved only after the chaos gate passes, then:

```powershell
git add evaluation/chaos_cases.yaml evaluation/bad_case_harness.py evaluation/bad_case_registry.json evaluation/README.md tests/test_failure_copy.py
git commit -m "test: gate accuracy-first failure handling"
```
