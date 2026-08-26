# Adaptive Verified Exercise Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate varied, difficulty-aware exercises whose hidden answers are deterministically verified before display.

**Architecture:** Add a typed exercise domain with parameterized templates, validation, progression, deduplication, and a server-side TTL store. The browser receives only an opaque exercise ID and non-sensitive metadata.

**Tech Stack:** Python 3.11, Pydantic 2, SymPy, `random.Random`, FastAPI, pytest, vanilla JavaScript

**Spec:** `docs/superpowers/specs/2026-08-26-accuracy-first-adaptive-tutor-design.md`

## Global Constraints

- Browser-visible state never contains solution text, answer signatures, or answer parameters.
- Every candidate passes condition-sufficiency and answer validation.
- Difficulty is an integer from 1 through 5 and respects grade boundaries.
- Recent problem and answer fingerprints prevent short-term repetition.

---

### Task 1: Exercise models and secure storage

**Files:**
- Create: `agentic_rag/exercises/__init__.py`
- Create: `agentic_rag/exercises/models.py`
- Create: `agentic_rag/exercises/store.py`
- Modify: `app.py`
- Test: `tests/exercises/test_models_and_store.py`

**Interfaces:**
- Produces: `ExerciseRequest`, `GeneratedExercise`, `PublicExerciseState`, `ExerciseSessionState`, `ExerciseStore`.
- `ExerciseStore.start(exercise, mastery) -> PublicExerciseState`; `ExerciseStore.get_exercise(exercise_id) -> GeneratedExercise | None`; `ExerciseStore.get_session(session_id) -> ExerciseSessionState | None`.

- [ ] **Step 1: Write failing secrecy and expiry tests**

```python
class FakeClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value


def make_exercise(**overrides):
    values = {
        "exercise_id": "ex-1", "topic": "geometry", "grade": 8, "difficulty": 2,
        "template_id": "geo.isosceles.base_angles.v1", "problem": "顶角为40度，求底角",
        "hint": "使用内角和", "solution": "底角为70度", "answer_signature": "70|70",
        "knowledge_points": ["等腰三角形"], "parameters": {"vertex": 40}, "fingerprint": "fp-1",
    }
    return GeneratedExercise(**(values | overrides))


def test_public_state_never_serializes_hidden_solution():
    exercise = make_exercise(solution="两个底角都是70度", answer_signature="70|70")
    public = ExerciseStore(ttl_seconds=60).start(exercise, mastery={})
    text = public.model_dump_json()
    assert "70|70" not in text
    assert "solution" not in text


def test_store_expires_exercise():
    clock = FakeClock(100.0)
    store = ExerciseStore(ttl_seconds=30, clock=clock)
    store.start(make_exercise(exercise_id="ex-1"), mastery={})
    clock.value = 131.0
    assert store.get_exercise("ex-1") is None
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises\test_models_and_store.py
```

Expected: `agentic_rag.exercises` is missing.

- [ ] **Step 3: Implement strict models**

```python
class ExerciseRequest(StrictModel):
    topic: Literal["geometry", "algebra", "linear_function"]
    grade: int = Field(default=8, ge=7, le=9)
    difficulty: int = Field(default=2, ge=1, le=5)
    exercise_type: Literal["calculation", "proof", "application", "mixed"] = "calculation"
    recent_fingerprints: list[str] = Field(default_factory=list, max_length=20)
    recent_answer_signatures: list[str] = Field(default_factory=list, max_length=10)
    seed: int | None = None


class PublicExerciseState(StrictModel):
    exercise_id: str
    session_id: str
    topic: str
    grade: int
    difficulty: int
    template_id: str
    fingerprint: str
    knowledge_points: list[str]


class ExerciseSessionState(StrictModel):
    session_id: str
    current_exercise_id: str
    recent_fingerprints: list[str] = Field(default_factory=list, max_length=20)
    recent_prompt_fingerprints: list[str] = Field(default_factory=list, max_length=10)
    recent_answer_signatures: list[str] = Field(default_factory=list, max_length=5)
    mastery: dict[str, float] = Field(default_factory=dict)
```

`GeneratedExercise` additionally holds `problem`, `hint`, `solution`, `answer_signature`, and private parameters. The store owns exercise records and session records, updates recent-history arrays and mastery after answer checks, and never serializes those private arrays to the browser. Protect both TTL dictionaries with `threading.Lock` and prune expired entries on reads and writes.

- [ ] **Step 4: Accept public state in requests**

Add `exercise_state: PublicExerciseState | None = None` to `AskRequest`; include it in the cache key. Pydantic must reject answer-bearing extra fields.

- [ ] **Step 5: Verify GREEN**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises\test_models_and_store.py tests\test_product_hardening.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add agentic_rag/exercises app.py tests/exercises/test_models_and_store.py tests/test_product_hardening.py
git commit -m "feat: add secure exercise session state"
```

---

### Task 2: Parameterized templates and validation

**Files:**
- Create: `agentic_rag/exercises/templates.py`
- Create: `agentic_rag/exercises/validation.py`
- Test: `tests/exercises/test_templates.py`

**Interfaces:**
- Produces: `TemplateDefinition`, `TEMPLATE_REGISTRY`, `generate_from_template(...)`, `validate_generated_exercise(...)`.
- Consumes: models from Task 1.

- [ ] **Step 1: Write failing property tests**

```python
@pytest.mark.parametrize("seed", range(50))
def test_isosceles_template_is_valid(seed):
    item = generate_from_template("geo.isosceles.base_angles.v1", 2, 8, seed)
    assert validate_generated_exercise(item).passed
    assert item.parameters["vertex_angle"] + 2 * item.parameters["base_angle"] == 180


@pytest.mark.parametrize("seed", range(50))
def test_linear_equation_has_unique_integer_solution(seed):
    item = generate_from_template("alg.linear_equation.v1", 2, 7, seed)
    assert validate_generated_exercise(item).passed
    p = item.parameters
    assert p["a"] != 0 and p["a"] * p["x"] + p["b"] == p["c"]


@pytest.mark.parametrize("seed", range(50))
def test_function_points_match_equation(seed):
    item = generate_from_template("fn.slope_intercept.v1", 2, 8, seed)
    assert validate_generated_exercise(item).passed
    p = item.parameters
    assert p["y0"] == p["b"] and p["y1"] == p["k"] + p["b"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises\test_templates.py
```

Expected: template APIs are absent.

- [ ] **Step 3: Implement template registry**

Register these families with sampler, problem renderer, hint renderer, hidden solution renderer, answer signature, and validator:

```text
geo.isosceles.base_angles.v1       grade 7-8, difficulty 1-2
geo.triangle.angle_ratio.v1        grade 7-8, difficulty 2-3
geo.congruence.sas_proof.v1        grade 8,   difficulty 3-4
alg.linear_equation.v1             grade 7,   difficulty 1-3
alg.factorization.difference.v1    grade 8,   difficulty 2-3
fn.slope_intercept.v1              grade 8,   difficulty 1-2
fn.two_points.v1                   grade 8-9, difficulty 2-4
```

Use exact integer/rational arithmetic. Reject degenerate triangles, ambiguous congruence premises, zero coefficients, and non-integer solutions at introductory difficulty.

- [ ] **Step 4: Verify GREEN**

Run Step 2. Expected: all 350 seeded property cases pass.

- [ ] **Step 5: Commit**

```powershell
git add agentic_rag/exercises/templates.py agentic_rag/exercises/validation.py tests/exercises/test_templates.py
git commit -m "feat: generate verified math exercises"
```

---

### Task 3: Adaptive selection and deduplication

**Files:**
- Create: `agentic_rag/exercises/generator.py`
- Create: `agentic_rag/exercises/progression.py`
- Test: `tests/exercises/test_adaptive_generator.py`

**Interfaces:**
- Produces: `AdaptiveExerciseGenerator.generate(request)`, `next_difficulty(current, outcome, explicit_delta)`, and `parse_practice_preferences(query, current)`.
- Consumes: Tasks 1-2.

- [ ] **Step 1: Write failing adaptation tests**

```python
def test_recent_twenty_fingerprints_are_avoided():
    generator = AdaptiveExerciseGenerator(max_attempts=100)
    recent = [generator.generate(ExerciseRequest(topic="geometry", seed=i)).fingerprint for i in range(20)]
    item = generator.generate(ExerciseRequest(topic="geometry", recent_fingerprints=recent, seed=100))
    assert item.fingerprint not in recent


@pytest.mark.parametrize("current,outcome,delta,expected", [
    (2, "correct", 0, 3), (3, "correct_after_hint", 0, 3),
    (3, "incorrect", 0, 2), (5, "correct", 0, 5),
    (1, "incorrect", 0, 1), (2, "unknown", 1, 3),
])
def test_progression(current, outcome, delta, expected):
    assert next_difficulty(current, outcome, delta) == expected


def test_same_topic_produces_varied_verified_items():
    items = [AdaptiveExerciseGenerator().generate(ExerciseRequest(topic="geometry", seed=i)) for i in range(12)]
    assert len({item.fingerprint for item in items}) >= 10
    assert all(validate_generated_exercise(item).passed for item in items)


def test_composite_preference_request_is_parsed():
    request = parse_practice_preferences("八年级几何难一点", current=None)
    assert (request.grade, request.topic, request.difficulty) == (8, "geometry", 3)


def test_explicit_easy_function_request_overrides_history():
    current = ExerciseSessionState(session_id="s1", current_exercise_id="e1", mastery={"一次函数": 0.9})
    request = parse_practice_preferences("来一道简单的一次函数题", current=current)
    assert request.topic == "linear_function"
    assert request.difficulty == 1
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises\test_adaptive_generator.py
```

Expected: generator and progression modules are absent.

- [ ] **Step 3: Implement bounded selection**

Filter by topic, grade, difficulty, and type. Use a local `random.Random(seed)` and at most 100 candidates. Reject recent 20 normalized fingerprints, recent 10 prompt fingerprints, and previous 5 answer signatures. If answer-signature diversity exhausts candidates, relax only that diversity rule; never relax validation or exact fingerprint avoidance.

`parse_practice_preferences` recognizes composite phrases such as `七年级代数简单题`, `八年级几何难一点`, `九年级一次函数综合题`, and English equivalents. Explicit grade, topic, difficulty, and exercise type override session defaults; omitted values come from the private session and then product defaults.

- [ ] **Step 4: Implement progression**

Outcomes are `correct`, `correct_after_hint`, `incorrect`, and `unknown`. Apply explicit delta first, then outcome adjustment, clamp 1-5, and filter templates by grade.

Update the private session mastery score for each knowledge point using a bounded exponential update: `new = clamp(0, 1, 0.8 * old + 0.2 * observed)`, where observed is `1.0` for correct, `0.6` for correct after hint, and `0.0` for incorrect. Use mastery only to select difficulty when the user did not provide an explicit difficulty command.

- [ ] **Step 5: Verify GREEN**

Run Step 2. Expected: all adaptive tests pass without external calls.

- [ ] **Step 6: Commit**

```powershell
git add agentic_rag/exercises/generator.py agentic_rag/exercises/progression.py tests/exercises/test_adaptive_generator.py
git commit -m "feat: adapt difficulty and prevent exercise repetition"
```

---

### Task 4: Conversation and frontend integration

**Files:**
- Modify: `agentic_rag/fast_path.py`
- Modify: `agentic_rag/domain/schemas.py`
- Modify: `agentic_rag/skill_handlers.py`
- Modify: `app.py`
- Modify: `static/app.js`
- Test: `tests/exercises/test_exercise_conversation.py`
- Modify: `evaluation/webapp_smoke.cjs`

**Interfaces:**
- Consumes: generator, store, local commands, response contract.
- Produces: `guided_exercise` with public state and verified answer checking by opaque ID.

- [ ] **Step 1: Write failing conversation tests**

```python
def test_next_exercise_does_not_repeat():
    first = build_fast_response("几何", [], language="zh")
    second = build_fast_response("再来一道", first["conversation_history"], language="zh", exercise_state=first["exercise_state"])
    assert second["exercise_state"]["fingerprint"] != first["exercise_state"]["fingerprint"]


def test_harder_increases_level_without_answer_leak():
    first = build_fast_response("一次函数", [], language="zh")
    second = build_fast_response("难一点", first["conversation_history"], language="zh", exercise_state=first["exercise_state"])
    assert second["exercise_state"]["difficulty"] == min(first["exercise_state"]["difficulty"] + 1, 5)
    assert "solution" not in str(second["exercise_state"])
    assert "answer_signature" not in str(second["exercise_state"])


def test_expired_exercise_requests_a_fresh_problem():
    result = check_exercise_answer("missing-id", "70度")
    assert result["response_type"] == "clarification_required"
    assert "新题" in result["answer"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises\test_exercise_conversation.py
```

Expected: current fast path has no exercise-state interface.

- [ ] **Step 3: Integrate generation and answer checks**

Extend `CurriculumSolveInput` and `build_fast_response` with optional public exercise state. Topic, next, and difficulty commands create a request, generate/store an item, and render problem plus hint only. Answer checks load hidden state by ID; missing IDs return a precise clarification. Keep legacy history-based checks only for pre-migration sessions.

- [ ] **Step 4: Persist public browser state**

```javascript
state.exercise = null;
// request body
exercise_state: state.exercise
// response handling
if ("exercise_state" in result) state.exercise = result.exercise_state;
```

Clear it on new conversation and new-question commands. Never write hidden answers to DOM or local storage.

- [ ] **Step 5: Verify integration**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises tests\test_math_pipeline.py tests\test_product_hardening.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' --check static\app.js
$env:NODE_PATH='C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' evaluation\webapp_smoke.cjs
```

Expected: tests, syntax, and browser smoke pass.

- [ ] **Step 6: Commit**

```powershell
git add agentic_rag/fast_path.py agentic_rag/domain/schemas.py agentic_rag/skill_handlers.py app.py static/app.js tests/exercises/test_exercise_conversation.py evaluation/webapp_smoke.cjs
git commit -m "feat: integrate adaptive practice sessions"
```

---

### Task 5: Exercise quality gate

**Files:**
- Create: `evaluation/exercise_quality_harness.py`
- Create: `evaluation/exercise_cases.jsonl`
- Modify: `evaluation/README.md`
- Modify: `evaluation/bad_case_registry.json`
- Test: `tests/exercises/test_quality_harness.py`

**Interfaces:**
- Produces: `run_exercise_quality(seed_count)` and a nonzero CLI exit on quality failure.

- [ ] **Step 1: Write failing gate test**

```python
def test_quality_harness():
    report = run_exercise_quality(seed_count=100)
    assert report["invalid_count"] == 0
    assert report["answer_leak_count"] == 0
    assert report["unique_ratio"] >= 0.90
    assert report["covered_topics"] == ["algebra", "geometry", "linear_function"]
```

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q tests\exercises\test_quality_harness.py
```

Expected: quality harness is absent.

- [ ] **Step 3: Implement executable quality checks**

For every supported topic and valid difficulty, generate 100 seeded candidates, validate them, inspect public state for leakage, and compute uniqueness. Fail on any invalid item, leak, unsupported level, or uniqueness below 0.90.

- [ ] **Step 4: Run release gate**

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' evaluation\exercise_quality_harness.py --report
& '.\.venv\Scripts\python.exe' evaluation\skill_harness.py --report
& '.\.venv\Scripts\python.exe' evaluation\pipeline_harness.py
```

Expected: all commands exit 0; invalid and leak counts are zero.

- [ ] **Step 5: Register and commit**

Register fixed-template repetition and answer-leak regressions, then:

```powershell
git add evaluation/exercise_quality_harness.py evaluation/exercise_cases.jsonl evaluation/README.md evaluation/bad_case_registry.json tests/exercises/test_quality_harness.py
git commit -m "test: gate adaptive exercise quality"
```
