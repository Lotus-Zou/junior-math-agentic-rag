from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agentic_rag.domain.schemas import (
    AnswerCriticInput,
    AnswerDraftOutput,
    AnswerGenerateInput,
    AnswerRepairInput,
    CriticOutput,
    CurriculumSolveOutput,
    RerankInput,
    RetrievalCandidate,
    RetrievalOutput,
    RenderInput,
    TurnRouteInput,
)
from agentic_rag.fast_path import build_fast_response
from agentic_rag.skill_handlers import (
    audit_final_judge,
    answer_final_judge,
    answer_critic,
    answer_generate,
    rerank_filter,
    targeted_fallback,
    turn_router,
)
from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.errors import RetryableSkillError
from agentic_rag.skill_runtime.pipeline import PipelineExecutor
from agentic_rag.skill_runtime.router import SkillRouter

import app as api
from config import CHROMA_PATH, KNOWLEDGE_SOURCE_PATH

def _active_geometry_turn():
    exercise = build_fast_response("几何", [], language="zh")
    return TurnRouteInput(
        query="这一步为什么这样做？请解释原理",
        language="zh",
        conversation_history=exercise["conversation_history"],
        conversation_summary=exercise["conversation_summary"],
        exercise_state=exercise["exercise_state"],
    )


def test_turn_router_sends_conceptual_follow_up_to_contextual_rag():
    routed = turn_router(_active_geometry_turn(), None)

    assert routed.route == "rag"
    assert routed.intent == "conceptual_followup"
    assert "当前练习" in routed.routed_query
    assert "当前练习知识点" in routed.routed_query
    assert "为什么" in routed.routed_query


def test_turn_router_keeps_answer_submission_and_solution_reveal_deterministic():
    active = _active_geometry_turn()
    answer = turn_router(active.model_copy(update={"query": "∠B=70°，∠C=70°"}), None)
    reveal = turn_router(active.model_copy(update={"query": "我不会，请直接给出完整答案"}), None)

    assert (answer.route, answer.intent) == ("deterministic", "answer_submission")
    assert (reveal.route, reveal.intent) == ("deterministic", "solution_reveal")


def test_turn_router_sends_natural_difficulty_feedback_to_exercise_agent():
    active = _active_geometry_turn()
    routed = turn_router(active.model_copy(update={"query": "太简单了"}), None)

    assert (routed.route, routed.intent) == (
        "exercise_agent",
        "difficulty_adjustment",
    )


def test_turn_router_prioritizes_explicit_competition_request_over_active_exercise():
    active = _active_geometry_turn()
    routed = turn_router(
        active.model_copy(update={"query": "帮我出一道九年级的数学竞赛题"}),
        None,
    )

    assert (routed.route, routed.intent) == ("exercise_agent", "new_exercise")
    assert routed.reason == "explicit exercise request"


def test_open_knowledge_question_uses_rag_route():
    routed = turn_router(
        TurnRouteInput(query="为什么等式两边同时加上同一个数，等式仍然成立？"),
        None,
    )

    assert (routed.route, routed.intent) == ("rag", "knowledge_query")


def test_complete_english_word_problem_is_not_misread_as_knowledge_query():
    routed = turn_router(
        TurnRouteInput(
            query=(
                "Solve this mathematics problem. John picks 4 bananas on Wednesday. "
                "Then he picks 6 bananas on Thursday. On Friday, he picks triple the "
                "number he did on Wednesday. How many bananas does John have?"
            ),
            language="en",
        ),
        None,
    )

    assert (routed.route, routed.intent) == ("deterministic", "problem_solve")


def test_practice_inside_word_problem_does_not_request_a_new_exercise():
    routed = turn_router(
        TurnRouteInput(
            query=(
                "Solve this mathematics problem. Hallie had dance practice for 1 hour "
                "on Tuesday and 2 hours on Thursday. How many hours did she practice?"
            ),
            language="en",
        ),
        None,
    )

    assert (routed.route, routed.intent) == ("deterministic", "problem_solve")


def test_complete_word_problem_with_currency_uses_solve_agent():
    routed = turn_router(
        TurnRouteInput(
            query=(
                "Solve this mathematics problem. A dance studio costs $25 per session "
                "plus $1.50 per student. It has 10 students and is rented 3 days a week. "
                "How much does it earn in a four-week month?"
            ),
            language="en",
        ),
        None,
    )

    assert (routed.route, routed.intent) == ("deterministic", "problem_solve")


def test_unhandled_local_solver_routes_to_solve_agent():
    selected = SkillRouter().choose(
        {"curriculum_solve": CurriculumSolveOutput(handled=False, response=None)},
        {
            "deterministic": "curriculum_tutor",
            "solve_agent": "answer_generate",
        },
    )

    assert selected == "answer_generate"


def test_first_error_audit_does_not_use_generic_fast_path_baseline(monkeypatch):
    from agentic_rag import chains

    calls = []

    def fake_invoke(_instance, _messages):
        calls.append(_messages)
        return AIMessage(
            content=(
                "Step 2 is the first mathematically incorrect step because 2 + 2 is 4."
            )
        )

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fake_invoke)
    context = SkillContext(
        request_id="first-error",
        trace_id="first-error",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_generate": True, "force_llm_every_turn": True},
    )
    result = answer_generate(
        AnswerGenerateInput(
            query=(
                "Audit this solution. Step 1. x = 3. Step 2. 2 + 2 = 5. "
                "Write only 'First error: N' on the final line."
            ),
            intent="problem_solve",
            contexts=[],
            language="en",
        ),
        context,
    )

    assert len(calls) == 1
    assert "最早一个会使推导、结论或后续计算失效" in calls[0][0].content
    assert "不精确但无害的术语" in calls[0][0].content
    assert "不得仅因知识点超纲或无检索证据拒绝" in calls[0][0].content
    assert result.answer.endswith("First error: 2")


def test_turn_router_sends_first_error_audit_to_agentic_pipeline():
    routed = turn_router(
        TurnRouteInput(
            query=(
                "Audit the proposed steps and identify the first incorrect step. "
                "Write only 'First error: N' on the final line."
            ),
            language="en",
        ),
        None,
    )

    assert (routed.route, routed.intent) == ("audit_agent", "error_analysis")
    assert routed.reason == "complete self-contained first-error audit"


def test_first_error_audit_ignores_harmless_wording_before_real_algebra_error(
    monkeypatch,
):
    from agentic_rag import chains

    query = (
        "Problem: Find the maximum of 4(x+7)(2-x).\n"
        "Step 1. This is a product of three linear factors.\n"
        "Step 2. Expand the expression.\n"
        "Step 3. 4(x+7)(2-x) = -4x^2 + 4x + 56.\n"
        "Identify the first mathematically incorrect step and write only "
        "'First error: N' on the final line."
    )
    captured = []

    def fake_invoke(_instance, messages):
        captured.append(messages)
        return AIMessage(
            content=(
                "The wording in Step 1 is harmless; Step 3 changes the linear "
                "coefficient from -20 to 4.\n\nFirst error: 3"
            )
        )

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fake_invoke)
    context = SkillContext(
        request_id="first-error-consequential",
        trace_id="first-error-consequential",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_generate": True, "force_llm_every_turn": True},
    )

    routed = turn_router(TurnRouteInput(query=query, language="en"), None)
    result = answer_generate(
        AnswerGenerateInput(
            query=query,
            intent=routed.intent,
            contexts=[],
            language="en",
        ),
        context,
    )

    assert routed.route == "audit_agent"
    assert len(captured) == 1
    assert result.answer.endswith("First error: 3")


def test_audit_final_judge_publishes_independent_structured_decision(monkeypatch):
    from agentic_rag import chains

    class FakeJudgeChain:
        def invoke(self, payload):
            assert "Step 5." in payload["query"]
            assert payload["candidate"]
            return {
                "first_error_step": 5,
                "explanation": "Step 5 treats an approximation as an exact equality.",
            }

    monkeypatch.setattr(
        chains,
        "get_first_error_judge_chain",
        lambda: FakeJudgeChain(),
    )
    context = SkillContext(
        request_id="audit-final-judge",
        trace_id="audit-final-judge",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True},
    )
    result = audit_final_judge(
        AnswerRepairInput(
            query=(
                "Step 1. The setup is valid.\n"
                "Step 2. The identity is valid.\n"
                "Step 3. The substitution is valid.\n"
                "Step 4. The algebra is valid.\n"
                "Step 5. 2 = 1.26^3.\n"
                "Write only 'First error: N' on the final line."
            ),
            answer="The candidate chose another step.\n\nFirst error: 7",
            issues=["candidate selected the wrong step"],
            intent="error_analysis",
            language="en",
        ),
        context,
    )

    assert result.response_type == "verified_answer"
    assert result.validation_passed is True
    assert result.answer.endswith("First error: 5")
    assert result.metrics.model_attempts == 1


def test_answer_final_judge_recovers_complete_problem_after_two_rejections(monkeypatch):
    from agentic_rag import chains

    class FakeJudgeChain:
        def invoke(self, payload):
            assert "22 games" in payload["query"]
            assert payload["candidate"]
            return {
                "is_complete": True,
                "math_logic_valid": True,
                "corrected_answer": (
                    "Let w be wins and l be losses. "
                    "w + l = 22 and w = l + 8, so 2l = 14, l = 7, and w = 15.\n\n"
                    "Answer: 15"
                ),
                "issues": [],
            }

    monkeypatch.setattr(
        chains,
        "get_final_answer_judge_chain",
        lambda: FakeJudgeChain(),
    )
    context = SkillContext(
        request_id="answer-final-judge",
        trace_id="answer-final-judge",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True},
    )
    result = answer_final_judge(
        AnswerRepairInput(
            query=(
                "Solve this mathematics problem. A football team played 22 games. "
                "They won 8 more than they lost. How many did they win? "
                "Write the final line as 'Answer: value'."
            ),
            answer="The twice-repaired candidate was rejected.",
            issues=["critic rejected the repaired draft"],
            intent="problem_solve",
            language="en",
        ),
        context,
    )

    assert result.response_type == "verified_answer"
    assert result.validation_passed is True
    assert result.answer.endswith("Answer: 15")
    assert result.metrics.model_attempts == 1


def test_answer_final_judge_cannot_bypass_strict_threshold_check(monkeypatch):
    from agentic_rag import chains

    class FakeJudgeChain:
        def invoke(self, _payload):
            return {
                "is_complete": True,
                "math_logic_valid": True,
                "corrected_answer": (
                    "The annual net is 7.5 and 90 / 7.5 = 12.\n\nAnswer: 12"
                ),
                "issues": [],
            }

    monkeypatch.setattr(
        chains,
        "get_final_answer_judge_chain",
        lambda: FakeJudgeChain(),
    )
    context = SkillContext(
        request_id="answer-final-threshold",
        trace_id="answer-final-threshold",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True},
    )

    with pytest.raises(ValueError, match="deterministic math checks"):
        answer_final_judge(
            AnswerRepairInput(
                query=(
                    "A tree costs $90. Its annual net income is $7.50. "
                    "How many years before it starts earning money? "
                    "Write the final line as 'Answer: value'."
                ),
                answer="Answer: 12",
                issues=["break-even is not profit"],
                intent="problem_solve",
                language="en",
            ),
            context,
        )


def test_answer_final_judge_explains_a_specific_ambiguity_instead_of_generic_rejection(
    monkeypatch,
):
    from agentic_rag import chains

    class FakeJudgeChain:
        def invoke(self, _payload):
            return {
                "is_complete": False,
                "math_logic_valid": True,
                "corrected_answer": (
                    "Lylah's increase rate is not stated, so the later total is not unique. "
                    "If her increase rate is r, the total is 56000 + (40000 / 1.3)(1 + r)."
                ),
                "issues": ["Lylah's salary increase rate"],
            }

    monkeypatch.setattr(
        chains,
        "get_final_answer_judge_chain",
        lambda: FakeJudgeChain(),
    )
    context = SkillContext(
        request_id="answer-final-ambiguity",
        trace_id="answer-final-ambiguity",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True},
    )
    result = answer_final_judge(
        AnswerRepairInput(
            query=(
                "Adrien earned 30% more than Lylah. Adrien's salary later rose by 40%. "
                "Both salaries increased. What was their later total?"
            ),
            answer="Please provide the complete problem.",
            issues=["Lylah's increase is unspecified"],
            intent="problem_solve",
            language="en",
        ),
        context,
    )

    assert result.response_type == "clarification_required"
    assert result.validation_passed is True
    assert "not stated" in result.answer
    assert result.clarification.missing == ["Lylah's salary increase rate"]


def test_self_contained_first_error_audit_uses_math_logic_without_exact_evidence(
    monkeypatch,
):
    from agentic_rag import chains

    class FakeCriticChain:
        def invoke(self, _payload):
            return {
                "is_valid": False,
                "factual_faithfulness": False,
                "math_logic_valid": True,
                "issues": ["no exact textbook excerpt"],
            }

    monkeypatch.setattr(chains, "get_answer_validation_chain", lambda: FakeCriticChain())
    context = SkillContext(
        request_id="audit-critic",
        trace_id="audit-critic",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True, "force_llm_every_turn": True},
    )
    result = answer_critic(
        AnswerCriticInput(
            query=(
                "Audit this complete solution and write only 'First error: N' on the final line."
            ),
            answer="The addition in step 2 is invalid.\n\nFirst error: 2",
            contexts=[],
        ),
        context,
    )

    assert result.passed is True
    assert result.critic["validation_policy"] == "self_contained_first_error_math_logic"


def test_turn_router_sends_complete_multiple_choice_problem_to_rag():
    routed = turn_router(
        TurnRouteInput(
            query=(
                "线段 AB=9，点 C 在线段 AB 上，求 MC。\n"
                "A. 3\nB. 3/2\nC. 9/2\nD. 15/2\n"
                "请在最后一行写答案字母。"
            )
        ),
        None,
    )

    assert (routed.route, routed.intent) == ("rag", "multiple_choice")


def test_out_of_scope_turn_is_identified_by_router_before_any_answer_branch():
    routed = turn_router(TurnRouteInput(query="帮我写一封求职邮件"), None)

    assert (routed.route, routed.intent) == ("general_agent", "general_chat")


def test_generic_non_math_question_does_not_enter_math_clarification():
    routed = turn_router(
        TurnRouteInput(query="请用两句话解释为什么养成复盘习惯有帮助"),
        None,
    )

    assert (routed.route, routed.intent) == ("general_agent", "general_chat")


def test_active_exercise_does_not_capture_unrelated_general_question():
    active = _active_geometry_turn()
    routed = turn_router(
        active.model_copy(update={"query": "请解释为什么养成复盘习惯有帮助"}),
        None,
    )

    assert (routed.route, routed.intent) == ("general_agent", "general_chat")


def test_time_request_uses_deterministic_utility_tool():
    routed = turn_router(TurnRouteInput(query="现在几点？"), None)

    assert (routed.route, routed.intent) == ("utility_tool", "utility_query")


def test_critic_value_not_skill_status_controls_pipeline_branch():
    rejected = CriticOutput(
        passed=False,
        factual_faithfulness=False,
        math_logic_valid=False,
        issues=["unsupported conclusion"],
    )
    accepted = rejected.model_copy(
        update={"passed": True, "factual_faithfulness": True, "math_logic_valid": True}
    )

    assert PipelineExecutor.branch_key(accepted) == "pass"
    assert PipelineExecutor.branch_key(rejected) == "fail"


def test_runtime_uses_shared_populated_knowledge_assets():
    assert Path(KNOWLEDGE_SOURCE_PATH).is_file()
    assert Path(CHROMA_PATH, "chroma.sqlite3").is_file()
    assert api._readiness_checks()["vector_index"] is True


def test_deterministic_rerank_prefers_active_exercise_knowledge_point():
    ranked = rerank_filter(
        RerankInput(
            query="当前练习知识点：三角形内角和。为什么要设为 k？",
            knowledge_points=["三角形内角和", "方程思想"],
            candidates=[
                RetrievalCandidate(chunk_id="pythagorean", content="勾股定理用于直角三角形。"),
                RetrievalCandidate(
                    chunk_id="angle-sum",
                    content="三角形内角和为 180°，比例角可设为 ak、bk、ck。",
                ),
            ],
        ),
        None,
    )

    assert ranked.candidates[0].chunk_id == "angle-sum"


def test_pipeline_projection_keeps_reranked_context_order():
    runner = PipelineExecutor(api._skill_executor)
    earlier = RetrievalOutput(
        candidates=[RetrievalCandidate(chunk_id="rrf-first", content="勾股定理")]
    )
    reranked = RetrievalOutput(
        candidates=[RetrievalCandidate(chunk_id="rerank-first", content="三角形内角和")]
    )
    state = {"rrf_fusion": earlier, "rerank_filter": reranked}

    projected = runner._project_input(
        "math.answer_generate@1",
        state,
        {"query": "为什么要使用三角形内角和？", "language": "zh"},
        None,
    )

    assert projected["contexts"][0]["chunk_id"] == "rerank-first"


def test_response_projection_keeps_public_model_metrics_only():
    runner = PipelineExecutor(api._skill_executor)
    state = {
        "last_result": {
            "metrics": {"latency_ms": 2.5, "attempt_limit": 1}
        },
        "rerank_filter": RetrievalOutput(
            candidates=[
                RetrievalCandidate(
                    chunk_id="linear-function",
                    content="一次函数 y=kx+b。",
                    source="教材.md",
                )
            ],
            model_attempts=1,
            model_successes=1,
        ),
        "answer_generate": AnswerDraftOutput(
            answer="斜率 k = -2，截距 b = 3。[1]",
            model_attempts=1,
            model_successes=1,
        ),
        "answer_critic": CriticOutput(
            passed=True,
            factual_faithfulness=True,
            math_logic_valid=True,
            model_attempts=1,
            model_successes=1,
        ),
    }

    projected = runner._project_input(
        "math.response_render@1",
        state,
        {
            "query": "一次函数 y=-2x+3 的斜率和截距是什么？",
            "language": "zh",
            "trace_id": "metrics-projection",
        },
        None,
    )

    assert projected["metrics"]["model_attempts"] == 3
    assert projected["metrics"]["model_successes"] == 3
    assert projected["metrics"]["model_failures"] == 0
    assert projected["metrics"]["tool_calls"] == 3
    assert "attempt_limit" not in projected["metrics"]
    assert projected["sources"][0]["excerpt"] == "一次函数 y=kx+b。"
    RenderInput.model_validate(projected)


def test_grounded_equality_fallback_explains_why_operation_preserves_equality():
    context = SkillContext(
        request_id="equality",
        trace_id="equality",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
    )
    draft = answer_generate(
        AnswerGenerateInput(
            query="为什么等式两边同时加上同一个数，等式仍然成立？",
            contexts=[
                RetrievalCandidate(
                    chunk_id="equation-property",
                    content="等式两边同时加上同一个数，等式仍成立。",
                )
            ],
        ),
        context,
    )

    assert "(a + c) - (b + c) = a - b = 0" in draft.answer


def test_force_llm_rag_generation_calls_model_even_with_deterministic_baseline(
    monkeypatch,
):
    from agentic_rag import chains

    baseline = build_fast_response(
        "一次函数 y = -2x + 3 的斜率和截距分别是什么？图像大致怎么画？",
        [],
        language="zh",
    )
    calls = []

    def fake_invoke(instance, _messages):
        calls.append(instance)
        return AIMessage(content=baseline["answer"])

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fake_invoke)
    context = SkillContext(
        request_id="forced-generation",
        trace_id="forced-generation",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_generate": True, "force_llm_every_turn": True},
    )
    result = answer_generate(
        AnswerGenerateInput(
            query="一次函数 y = -2x + 3 的斜率和截距分别是什么？图像大致怎么画？",
            contexts=[
                RetrievalCandidate(
                    chunk_id="linear-function",
                    content="一次函数 y=kx+b 中，k 是斜率，b 是纵截距。",
                )
            ],
        ),
        context,
    )

    assert calls == [chains.generator_llm]
    assert result.model_attempts == 1
    assert result.model_successes == 1
    assert result.model_failures == 0
    assert "斜率 k = -2" in result.answer


def test_failed_forced_generation_is_not_reported_as_successful_model_call(
    monkeypatch,
):
    from agentic_rag import chains

    def fail(_instance, _messages):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fail)
    context = SkillContext(
        request_id="failed-generation",
        trace_id="failed-generation",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_generate": True, "force_llm_every_turn": True},
    )
    result = answer_generate(
        AnswerGenerateInput(
            query="解方程 2x+3=11",
            contexts=[
                RetrievalCandidate(
                    chunk_id="equation",
                    content="解一元一次方程时，可在等式两边进行相同运算。",
                )
            ],
        ),
        context,
    )

    assert result.model_attempts == 1
    assert result.model_successes == 0
    assert result.model_failures == 1
    assert "x = 4" in result.answer


def test_failed_self_contained_generation_without_context_is_retryable(monkeypatch):
    from agentic_rag import chains

    def fail(_instance, _messages):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fail)
    context = SkillContext(
        request_id="failed-self-contained-generation",
        trace_id="failed-self-contained-generation",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_generate": True, "force_llm_every_turn": True},
    )

    try:
        answer_generate(
            AnswerGenerateInput(
                query=(
                    "Solve this mathematics problem. A shop sold 12 pens in the "
                    "morning and 8 in the afternoon. How many pens were sold?"
                ),
                intent="problem_solve",
                contexts=[],
                language="en",
            ),
            context,
        )
    except RetryableSkillError as exc:
        assert "complete problem has been preserved" in exc.safe_message
    else:
        raise AssertionError("model outage must be classified as retryable")


def test_force_llm_critic_runs_for_deterministic_draft(monkeypatch):
    from agentic_rag import chains

    calls = []

    class FakeCriticChain:
        def invoke(self, payload):
            calls.append(payload)
            return {
                "is_valid": True,
                "factual_faithfulness": True,
                "math_logic_valid": True,
                "issues": [],
            }

    monkeypatch.setattr(
        chains,
        "get_answer_validation_chain",
        lambda: FakeCriticChain(),
    )
    baseline = build_fast_response("解方程 2x+3=11", [], language="zh")
    context = SkillContext(
        request_id="forced-critic",
        trace_id="forced-critic",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True, "force_llm_every_turn": True},
    )
    result = answer_critic(
        AnswerCriticInput(
            query="解方程 2x+3=11",
            answer=baseline["answer"],
            contexts=["解方程时，等式两边进行相同运算。"],
        ),
        context,
    )

    assert len(calls) == 1
    assert result.passed is True
    assert result.model_attempts == 1
    assert result.model_successes == 1
    assert result.model_failures == 0


def test_self_contained_multiple_choice_uses_independent_math_logic_when_exact_evidence_is_absent(
    monkeypatch,
):
    from agentic_rag import chains

    class FakeCriticChain:
        def invoke(self, _payload):
            return {
                "is_valid": False,
                "factual_faithfulness": False,
                "math_logic_valid": True,
                "issues": ["知识库没有逐字覆盖此题"],
            }

    monkeypatch.setattr(
        chains,
        "get_answer_validation_chain",
        lambda: FakeCriticChain(),
    )
    context = SkillContext(
        request_id="choice-critic",
        trace_id="choice-critic",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True, "force_llm_every_turn": True},
    )
    result = answer_critic(
        AnswerCriticInput(
            query="三角形内角和是多少？\nA. 90°\nB. 180°\nC. 270°\nD. 360°",
            answer="逐项核对可知三角形内角和为 180°。\n答案：B",
            contexts=[],
        ),
        context,
    )

    assert result.passed is True
    assert result.math_logic_valid is True
    assert result.factual_faithfulness is False
    assert result.critic["validation_policy"] == "self_contained_multiple_choice_math_logic"


def test_self_contained_arithmetic_uses_math_logic_when_retrieval_is_empty(
    monkeypatch,
):
    from agentic_rag import chains

    class FakeCriticChain:
        def invoke(self, _payload):
            return {
                "is_valid": False,
                "factual_faithfulness": False,
                "math_logic_valid": True,
                "issues": ["no external evidence"],
            }

    monkeypatch.setattr(
        chains,
        "get_answer_validation_chain",
        lambda: FakeCriticChain(),
    )
    context = SkillContext(
        request_id="arithmetic-critic",
        trace_id="arithmetic-critic",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True, "force_llm_every_turn": True},
    )
    result = answer_critic(
        AnswerCriticInput(
            query=(
                "Solve this mathematics problem. A team has 5 players and 2 coaches. "
                "How many people are there? Show the calculation."
            ),
            answer="5 + 2 = 7.\nAnswer: 7",
            contexts=[],
        ),
        context,
    )

    assert result.passed is True
    assert result.critic["validation_policy"] == "self_contained_problem_math_logic"


def test_profit_threshold_requires_period_after_exact_break_even(monkeypatch):
    from agentic_rag import chains

    class FakeCriticChain:
        def invoke(self, _payload):
            return {
                "is_valid": True,
                "factual_faithfulness": False,
                "math_logic_valid": True,
                "issues": [],
            }

    monkeypatch.setattr(
        chains,
        "get_answer_validation_chain",
        lambda: FakeCriticChain(),
    )
    context = SkillContext(
        request_id="profit-threshold",
        trace_id="profit-threshold",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True, "force_llm_every_turn": True},
    )
    query = (
        "Solve this mathematics problem. A tree costs $90. Its annual net income "
        "is $7.50. How many years will it take before it starts earning money?"
    )
    break_even = answer_critic(
        AnswerCriticInput(
            query=query,
            answer="$90 ÷ $7.50 = 12 years.\nAnswer: 12",
            contexts=[],
        ),
        context,
    )
    profitable = answer_critic(
        AnswerCriticInput(
            query=query,
            answer=(
                "$90 ÷ $7.50 = 12 years to break even. Profit starts in the "
                "next year.\nAnswer: 13"
            ),
            contexts=[],
        ),
        context,
    )

    assert break_even.passed is False
    assert any("盈亏平衡" in issue for issue in break_even.issues)
    assert profitable.passed is True


def test_self_contained_arithmetic_is_not_rejected_by_irrelevant_context(
    monkeypatch,
):
    from agentic_rag import chains

    class FakeCriticChain:
        def invoke(self, _payload):
            return {
                "is_valid": False,
                "factual_faithfulness": False,
                "math_logic_valid": True,
                "issues": ["retrieved context does not support this word problem"],
            }

    monkeypatch.setattr(
        chains,
        "get_answer_validation_chain",
        lambda: FakeCriticChain(),
    )
    context = SkillContext(
        request_id="arithmetic-irrelevant-context",
        trace_id="arithmetic-irrelevant-context",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
        feature_flags={"llm_critic": True, "force_llm_every_turn": True},
    )
    result = answer_critic(
        AnswerCriticInput(
            query=(
                "Solve this mathematics problem. A team has 5 players and 2 coaches. "
                "How many people are there? Show the calculation."
            ),
            answer="5 + 2 = 7.\nAnswer: 7",
            contexts=["A triangle has an interior-angle sum of 180 degrees."],
        ),
        context,
    )

    assert result.passed is True
    assert result.math_logic_valid is True
    assert result.factual_faithfulness is False
    assert result.deterministic["passed"] is False
    assert result.critic["validation_policy"] == "self_contained_problem_math_logic"


def test_production_entry_executes_versioned_turn_router_pipeline():
    before = len(api._skill_executor.telemetry.events)
    run = api._run_curriculum_skill(api.AskRequest(query="几何", language="zh"))
    events = api._skill_executor.telemetry.events[before:]

    assert run is not None
    assert run.response["response_type"] == "guided_exercise"
    assert any(event.skill == "math.turn_router@1.0.0" for event in events)
    assert any(event.skill == "math.exercise_generate@1.0.0" for event in events)
    assert all(event.pipeline == "math.correction@1.0.0" for event in events)


def test_production_pipeline_replaces_active_exercise_with_competition_request():
    first = api._run_curriculum_skill(api.AskRequest(query="几何", language="zh"))
    assert first is not None
    second = api._run_curriculum_skill(
        api.AskRequest(
            query="帮我出一道九年级的数学竞赛题",
            language="zh",
            conversation_history=first.response["conversation_history"],
            conversation_summary=first.response["conversation_summary"],
            exercise_state=first.response["exercise_state"],
        )
    )

    assert second is not None
    assert second.response["response_type"] == "guided_exercise"
    assert second.response["intent"] == "algebra_exercise"
    assert second.response["exercise_state"]["grade"] == 9
    assert second.response["exercise_state"]["difficulty"] == 5
    assert second.response["exercise_state"]["exercise_type"] == "mixed"
    assert (
        second.response["exercise_state"]["exercise_id"]
        != first.response["exercise_state"]["exercise_id"]
    )


def test_production_pipeline_starts_with_turn_router_for_general_turn(monkeypatch):
    from agentic_rag import chains

    def fake_invoke(_instance, _messages):
        return AIMessage(content="这是一封简洁的求职邮件。")

    monkeypatch.setattr(type(chains.generator_llm), "invoke", fake_invoke)
    before = len(api._skill_executor.telemetry.events)
    run = api._run_curriculum_skill(api.AskRequest(query="帮我写一封求职邮件", language="zh"))
    events = api._skill_executor.telemetry.events[before:]

    assert run is not None
    assert run.response["response_type"] == "general_answer"
    assert run.response["intent"] == "general_chat"
    assert events[0].skill == "math.turn_router@1.0.0"
    assert any(event.skill == "assistant.general_agent@1.0.0" for event in events)
    assert not any(event.skill in {"math.answer_generate@1.0.0", "math.answer_critic@1.0.0"} for event in events)


def test_production_pipeline_answers_time_without_math_rag():
    before = len(api._skill_executor.telemetry.events)
    run = api._run_curriculum_skill(api.AskRequest(query="现在几点？", language="zh"))
    events = api._skill_executor.telemetry.events[before:]

    assert run is not None
    assert run.response["response_type"] == "general_answer"
    assert run.response["intent"] == "utility_time"
    assert "Asia/Shanghai" in run.response["answer"]
    assert any(event.skill == "assistant.utility_tool@1.0.0" for event in events)
    assert not any(event.skill.startswith("math.retrieve") for event in events)


def test_cache_hit_is_still_classified_by_versioned_turn_router(monkeypatch):
    request = api.AskRequest(query="一次函数的斜率是什么？", language="zh")
    before = len(api._skill_executor.telemetry.events)

    assert api._route_cached_turn(request) is True
    events = api._skill_executor.telemetry.events[before:]

    assert len(events) == 1
    assert events[0].skill == "math.turn_router@1.0.0"
    assert events[0].pipeline == "math.correction.cache_hit@1.0.0"


def test_force_llm_mode_bypasses_final_answer_cache(monkeypatch):
    monkeypatch.setattr(api, "FORCE_LLM_EVERY_TURN", True)

    assert api._bypass_answer_cache(
        api.AskRequest(query="解方程 2x+3=11", language="zh")
    ) is True


def test_twice_rejected_answer_becomes_targeted_continuation_not_generic_rejection():
    context = SkillContext(
        request_id="fallback",
        trace_id="fallback",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
    )
    result = targeted_fallback(
        AnswerRepairInput(
            query="这一步为什么要用三角形内角和？",
            answer="",
            contexts=[
                RetrievalCandidate(
                    chunk_id="angle-sum",
                    content="三角形三个内角的和等于 180°。",
                    source="教材.md",
                    metadata={"chapter": "几何"},
                )
            ],
            issues=["missing application to the current step"],
        ),
        context,
    )

    assert result.validation_passed is True
    assert result.response_type == "clarification_required"
    assert "当前可以确认" in result.answer
    assert "草稿未通过" not in result.answer


def test_twice_rejected_complete_knowledge_query_returns_grounded_answer():
    context = SkillContext(
        request_id="knowledge-fallback",
        trace_id="knowledge-fallback",
        deadline_at=api.datetime.now(api.timezone.utc) + api.timedelta(seconds=5),
    )
    result = targeted_fallback(
        AnswerRepairInput(
            query="为什么不等式两边同除以负数时必须改变方向？",
            answer="unsupported draft",
            intent="knowledge_query",
            contexts=[
                RetrievalCandidate(
                    chunk_id="inequality-order",
                    content="同乘或同除以负数会使大小顺序反转，所以不等号必须改变方向。",
                    source="教材.md",
                    metadata={"chapter": "代数"},
                )
            ],
            issues=["critic rejected draft"],
        ),
        context,
    )

    assert result.response_type == "verified_answer"
    assert result.intent == "knowledge_query"
    assert result.clarification is None
    assert "不等号必须改变方向" in result.answer
    assert "请补" not in result.answer


def test_pipeline_solution_reveal_uses_authoritative_turn_router_intent():
    exercise = api._run_curriculum_skill(api.AskRequest(query="几何", language="zh"))
    assert exercise is not None

    response = api._run_curriculum_skill(
        api.AskRequest(
            query="我不会，请直接给出完整答案和步骤",
            language="zh",
            conversation_history=exercise.response["conversation_history"],
            conversation_summary=exercise.response["conversation_summary"],
            exercise_state=exercise.response["exercise_state"],
        )
    )

    assert response is not None
    assert response.response["response_type"] == "verified_answer"
    assert response.response["intent"] == "adaptive_solution_reveal"
    assert "完整解答" in response.response["answer"]
    assert "答案继续隐藏" not in response.response["answer"]
