# Reliable Response Contract and Local Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every request a stable response type and route short study commands such as “几何” locally.

**Architecture:** Add a repository-owned response contract and a strict local command parser. Keep `math.curriculum_solve@1` as the API entry, while normalizing fast-path, cache, and graph results at the API boundary.

**Tech Stack:** Python 3.11, Pydantic 2, FastAPI, Skill Runtime, pytest, vanilla JavaScript

**Spec:** `docs/superpowers/specs/2026-08-26-accuracy-first-adaptive-tutor-design.md`

## Global Constraints

- Public response types are only `verified_answer`, `guided_exercise`, `clarification_required`, and `supported_refusal`.
- A mathematical conclusion requires deterministic or independent Critic validation.
- Internal timeout, retrieval, model, and Critic states stay in Trace data.
- Preserve unrelated worktree changes.

---

### Task 1: Response contract

**Files:**
- Create: `agentic_rag/response_contract.py`
- Modify: `agentic_rag/domain/schemas.py`
- Test: `tests/test_response_contract.py`

**Interfaces:**
- Produces: `ResponseType`, `normalize_response(payload, response_type)`, `clarification_response(...)`, `supported_refusal_response(...)`.
- Consumes: current fast-path and graph response dictionaries.

- [ ] **Step 1: Write failing tests**

```python
def test_verified_answer_requires_validation():
    with pytest.raises(ValueError, match="verified_answer"):
        normalize_response({"answer": "x = 4", "validation_passed": False}, "verified_answer")


def test_clarification_names_missing_input_without_internal_copy():
    result = clarification_response("如图，求角A", [], "", ["图中已知角和点的位置关系"], "zh")
    assert result["response_type"] == "clarification_required"
    assert "图中已知角和点的位置关系" in result["answer"]
    assert "复杂推理服务" not in result["answer"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_response_contract.py
```

Expected: import fails because `agentic_rag.response_contract` is absent.

- [ ] **Step 3: Implement the contract**

```python
ResponseType = Literal["verified_answer", "guided_exercise", "clarification_required", "supported_refusal"]


def normalize_response(payload: dict[str, Any], response_type: ResponseType) -> dict[str, Any]:
    result = {
        "answer": "", "trace_id": str(uuid.uuid4()), "intent": "",
        "knowledge_points": [], "sources": [], "validation_passed": False,
        "critic_report": {}, "conversation_history": [], "conversation_summary": "",
        "exercise_state": None, "clarification": None, "metrics": {}, "cached": False,
        **payload, "response_type": response_type,
    }
    if response_type == "verified_answer" and not result["validation_passed"]:
        raise ValueError("verified_answer requires validation_passed=True")
    if response_type == "guided_exercise" and (
        not result["validation_passed"] or not result["critic_report"].get("exercise_answer_hidden")
    ):
        raise ValueError("guided_exercise requires a verified hidden answer")
    return result
```

`clarification_response` appends the current student and tutor turns, sets `clarification={"missing": missing}`, and uses no sources.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: all response contract tests pass.

- [ ] **Step 5: Commit**

```powershell
git add agentic_rag/response_contract.py agentic_rag/domain/schemas.py tests/test_response_contract.py
git commit -m "feat: define reliable tutor response contract"
```

---

### Task 2: Local command parser and topic routing

**Files:**
- Create: `agentic_rag/local_intents.py`
- Modify: `agentic_rag/fast_path.py`
- Modify: `agentic_rag/skill_handlers.py`
- Test: `tests/test_local_intents.py`
- Modify: `tests/test_product_hardening.py`

**Interfaces:**
- Produces: `LocalCommand(action, topic, difficulty_delta)` and `parse_local_command(query, language)`.
- Consumes: response normalization from Task 1 and current verified exercise builders.

- [ ] **Step 1: Write failing parser tests**

```python
@pytest.mark.parametrize("query,action,topic,delta", [
    ("几何", "practice", "geometry", 0),
    ("代数", "practice", "algebra", 0),
    ("一次函数", "practice", "linear_function", 0),
    ("再来一道", "next_exercise", None, 0),
    ("难一点", "adjust_difficulty", None, 1),
    ("简单一点", "adjust_difficulty", None, -1),
    ("换个问题", "new_question", None, 0),
])
def test_short_commands(query, action, topic, delta):
    command = parse_local_command(query, "zh")
    assert (command.action, command.topic, command.difficulty_delta) == (action, topic, delta)


def test_complete_problem_is_not_a_topic_command():
    assert parse_local_command("在三角形ABC中，A=40度，求B和C", "zh") is None
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_local_intents.py
```

Expected: missing `agentic_rag.local_intents`.

- [ ] **Step 3: Implement exact normalized matching**

```python
@dataclass(frozen=True)
class LocalCommand:
    action: Literal["practice", "next_exercise", "adjust_difficulty", "new_question", "reset"]
    topic: Literal["geometry", "algebra", "linear_function"] | None = None
    difficulty_delta: int = 0
```

Normalize with NFKC, lowercase English, collapsed whitespace, and stripped trailing punctuation. Match single-topic commands as complete strings. Include English aliases `geometry`, `algebra`, `linear function`, `another exercise`, `harder`, `easier`, and `new question`.

- [ ] **Step 4: Add failing fast-path tests**

```python
def test_geometry_topic_is_a_local_guided_exercise():
    result = build_fast_response("几何", [], language="zh")
    assert result["response_type"] == "guided_exercise"
    assert result["intent"] == "geometry_exercise"
    assert result["metrics"]["tool_calls"] == 0
    assert result["critic_report"]["exercise_answer_hidden"] is True


def test_every_advertised_topic_has_a_result():
    for query in ("代数", "几何", "一次函数"):
        assert build_fast_response(query, [], language="zh")["response_type"] == "guided_exercise"
```

- [ ] **Step 5: Verify fast-path RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_product_hardening.py -k "topic or advertised"
```

Expected: `几何` currently returns `None`.

- [ ] **Step 6: Route commands before question handlers**

Call the command handler before geometry-history, curriculum, and equation handlers. Reuse verified geometry and equation templates, add one locally verified linear-function exercise for this transition, and assign response types explicitly. `换个问题` returns `clarification_required`, clears current history/summary, and asks for the next input.

- [ ] **Step 7: Verify GREEN**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_local_intents.py tests\test_product_hardening.py tests\test_math_pipeline.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```powershell
git add agentic_rag/local_intents.py agentic_rag/fast_path.py agentic_rag/skill_handlers.py tests/test_local_intents.py tests/test_product_hardening.py tests/test_math_pipeline.py
git commit -m "fix: route short study commands locally"
```

---

### Task 3: API and frontend enforcement

**Files:**
- Modify: `app.py`
- Modify: `static/app.js`
- Modify: `evaluation/webapp_smoke.cjs`
- Modify: `evaluation/bad_case_registry.json`
- Test: `tests/test_api_contract.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: required `response_type` on `/ask` and response-type-aware rendering.

- [ ] **Step 1: Write failing API tests**

```python
client = TestClient(app)


def test_geometry_never_reaches_complex_reasoning():
    payload = client.post("/ask", json={"query": "几何", "language": "zh"}).json()
    assert payload["response_type"] == "guided_exercise"
    assert payload["metrics"]["tool_calls"] == 0
    assert "复杂推理服务" not in payload["answer"]


def test_new_question_clears_previous_state():
    payload = client.post("/ask", json={
        "query": "换个问题", "language": "zh", "conversation_summary": "旧摘要",
        "conversation_history": [{"role": "student", "content": "解方程 2x+3=11"}],
    }).json()
    assert payload["conversation_summary"] == ""
    assert "2x+3=11" not in str(payload["conversation_history"])
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_api_contract.py
```

Expected: `response_type` or geometry routing assertion fails.

- [ ] **Step 3: Normalize successful API responses**

Normalize cache hits, curriculum Skill results, and graph results. Infer type only from validation, hidden-exercise metadata, clarification state, or refusal state; never from non-empty answer text.

- [ ] **Step 4: Render response labels**

```javascript
const responseLabels = {
  verified_answer: t("validated"),
  guided_exercise: t("localExercise"),
  clarification_required: t("needsMoreInfo"),
  supported_refusal: t("scopeLimited")
};
```

Keep state assignment by property presence so empty fields clear state. Extend browser smoke to submit `换个问题`, then `几何`.

- [ ] **Step 5: Verify API and browser**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_api_contract.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' --check static\app.js
$env:NODE_PATH='C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' evaluation\webapp_smoke.cjs
```

Expected: all commands exit 0 and browser sees `几何练习` without internal failure copy.

- [ ] **Step 6: Register and commit**

Add `ROUTE-006` for the `几何` short-command timeout regression.

```powershell
git add app.py static/app.js evaluation/webapp_smoke.cjs evaluation/bad_case_registry.json tests/test_api_contract.py
git commit -m "feat: enforce response contract at API boundary"
```

---

### Task 4: Phase gate

**Files:**
- Modify: `evaluation/skill_cases/conversation.jsonl`
- Modify: `evaluation/pipeline_cases/exercise.jsonl`
- Modify: `evaluation/skill_harness.py`
- Modify: `evaluation/pipeline_harness.py`
- Modify: `evaluation/README.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: permanent executable regression cases.

- [ ] **Step 1: Add cases**

Add literal cases for `几何`, `代数`, `一次函数`, `换个问题`, `难一点`, and `简单一点`. Extend both harnesses with dotted-path assertions so nested `response.response_type` and rendered `response_type` are executable checks:

```python
def value_at_path(value, dotted_path):
    current = value
    for part in dotted_path.split("."):
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
    return current


for path, expected in case.get("expected_paths", {}).items():
    if value_at_path(actual, path) != expected:
        reasons.append(f"{path}={value_at_path(actual, path)!r}")
```

Pipeline cases use the same helper against `state`; Skill cases assert `response.response_type`. Keep the existing flat `expected` and `answer_contains` behavior for backward compatibility.

- [ ] **Step 2: Run full phase gate**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' evaluation\skill_harness.py --report
& '.\.venv\Scripts\python.exe' evaluation\pipeline_harness.py
```

Expected: all commands exit 0 with zero failures.

- [ ] **Step 3: Verify live bad case**

Restart only the confirmed AgentiRAG listener. Submit `换个问题`, then `几何`; assert `guided_exercise`, zero tool calls, and no `复杂推理服务` or `超时` text.

- [ ] **Step 4: Commit**

```powershell
git add evaluation/skill_cases/conversation.jsonl evaluation/pipeline_cases/exercise.jsonl evaluation/skill_harness.py evaluation/pipeline_harness.py evaluation/README.md
git commit -m "test: gate local conversation routing"
```
