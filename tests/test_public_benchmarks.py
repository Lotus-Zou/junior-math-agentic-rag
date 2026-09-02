from evaluation.public_benchmarks import (
    BenchmarkCase,
    ceval_case,
    evaluate_cases,
    extract_prediction,
    gsm8k_case,
    normalize_number,
    VersionRunLock,
)
from agentic_rag.domain.schemas import TurnRouteOutput
from agentic_rag.skill_runtime.router import SkillRouter


def test_ceval_adapter_and_choice_extraction():
    case = ceval_case(
        {
            "id": 7,
            "question": "下列结论正确的是",
            "A": "一",
            "B": "二",
            "C": "三",
            "D": "四",
            "answer": "C",
        }
    )

    assert case.expected == "C"
    assert "A. 一" in case.question
    assert extract_prediction(case.benchmark, "推导过程\n答案：C") == "C"


def test_gsm8k_adapter_uses_official_final_answer_marker():
    case = gsm8k_case(
        {"question": "What is 9 times 2?", "answer": "9 * 2 = 18\n#### 18"},
        0,
    )

    assert case.expected == "18"
    assert case.question.startswith("Solve this mathematics problem.")
    assert extract_prediction(case.benchmark, "Calculation\nAnswer: 18") == "18"


def test_numeric_normalization_handles_commas_decimals_and_negative_values():
    assert normalize_number("#### 1,200.00") == "1200"
    assert normalize_number("Answer: -3.50") == "-3.5"
    assert normalize_number("Answer: 120/3") == "40"


def test_gsm8k_extraction_requires_a_numeric_final_answer_line():
    assert extract_prediction("gsm8k", "Hint 1: add 2 and 3") == ""
    assert extract_prediction("gsm8k", "Answer: cannot be determined") == ""
    assert extract_prediction("gsm8k", "Work\nAnswer: 120/3") == "40"


def test_public_benchmark_summary_keeps_dataset_scores_separate(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "答案：A", "validation_passed": True, "metrics": {}}

    monkeypatch.setattr("httpx.Client.post", lambda *args, **kwargs: Response())
    cases = [
        BenchmarkCase("c1", "ceval_middle_school_mathematics", "q", "A", "zh"),
        BenchmarkCase("c2", "gsm8k", "q", "2", "en"),
    ]

    _, summary = evaluate_cases(cases, api_url="http://test", timeout_seconds=1)

    assert "accuracy" not in summary
    assert summary["results"]["ceval_middle_school_mathematics"]["accuracy"] == 1.0
    assert summary["results"]["gsm8k"]["accuracy"] == 0.0
    assert summary["results"]["gsm8k"]["blank_predictions"] == 1
    assert summary["conditional_accuracy"] == 1.0


def test_public_benchmark_checkpoint_resumes_without_repeating_calls(
    monkeypatch, tmp_path
):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "答案：A", "validation_passed": True, "metrics": {}}

    def post(*args, **kwargs):
        calls.append(kwargs["json"]["query"])
        return Response()

    monkeypatch.setattr("httpx.Client.post", post)
    cases = [
        BenchmarkCase("c1", "ceval_middle_school_mathematics", "q1", "A", "zh"),
        BenchmarkCase("c2", "ceval_middle_school_mathematics", "q2", "A", "zh"),
    ]
    checkpoint = tmp_path / "checkpoint.jsonl"

    first, _ = evaluate_cases(
        cases,
        api_url="http://test",
        timeout_seconds=1,
        checkpoint_path=checkpoint,
        workers=2,
    )
    second, summary = evaluate_cases(
        cases,
        api_url="http://test",
        timeout_seconds=1,
        checkpoint_path=checkpoint,
        workers=2,
    )

    assert len(first) == len(second) == 2
    assert calls == ["q1", "q2"] or calls == ["q2", "q1"]
    assert summary["resumed_cases"] == 2
    assert summary["retried_cases"] == 0
    assert summary["new_cases"] == 0


def test_public_benchmark_checkpoint_retries_empty_predictions(
    monkeypatch, tmp_path
):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "Answer: 18", "validation_passed": True, "metrics": {}}

    def post(*args, **kwargs):
        calls.append(kwargs["json"]["query"])
        return Response()

    monkeypatch.setattr("httpx.Client.post", post)
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                '{"case_id":"c1","benchmark":"gsm8k","predicted":"","error":""}',
                '{"case_id":"c2","benchmark":"gsm8k","predicted":"18","error":""}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cases = [
        BenchmarkCase("c1", "gsm8k", "retry me", "18", "en"),
        BenchmarkCase("c2", "gsm8k", "keep me", "18", "en"),
    ]

    frame, summary = evaluate_cases(
        cases,
        api_url="http://test",
        timeout_seconds=1,
        checkpoint_path=checkpoint,
        workers=1,
    )

    assert calls == ["retry me"]
    assert len(frame) == 2
    assert summary["resumed_cases"] == 1
    assert summary["retried_cases"] == 1
    assert summary["new_cases"] == 1


def test_public_benchmark_report_only_keeps_empty_predictions_without_calls(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        "httpx.Client.post",
        lambda *args, **kwargs: calls.append(kwargs["json"]["query"]),
    )
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                '{"case_id":"c1","benchmark":"gsm8k","predicted":"","correct":0.0,"error":"","latency_ms":10,"validation_passed":true}',
                '{"case_id":"c2","benchmark":"gsm8k","predicted":"18","correct":1.0,"error":"","latency_ms":20,"validation_passed":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cases = [
        BenchmarkCase("c1", "gsm8k", "do not retry", "18", "en"),
        BenchmarkCase("c2", "gsm8k", "keep me", "18", "en"),
    ]

    frame, summary = evaluate_cases(
        cases,
        api_url="http://test",
        timeout_seconds=1,
        checkpoint_path=checkpoint,
        retry_invalid=False,
    )

    assert calls == []
    assert len(frame) == 2
    assert summary["blank_predictions"] == 1
    assert summary["conditional_accuracy"] == 1.0
    assert summary["resumed_cases"] == 2
    assert summary["retried_cases"] == 0
    assert summary["new_cases"] == 0


def test_skill_router_can_branch_on_turn_intent_after_retrieval_failure():
    state = {
        "turn_router": TurnRouteOutput(
            intent="problem_solve",
            route="rag",
            routed_query="Solve the complete problem",
            reason="complete problem",
        )
    }

    selected = SkillRouter().choose(
        state,
        {"problem_solve": "answer_generate", "default": "no_evidence_response"},
    )

    assert selected == "answer_generate"


def test_version_run_lock_rejects_a_second_writer(tmp_path):
    lock_path = tmp_path / "same-version.lock"

    with VersionRunLock(lock_path):
        try:
            with VersionRunLock(lock_path):
                raise AssertionError("second lock unexpectedly succeeded")
        except RuntimeError as exc:
            assert "已在运行" in str(exc)

    assert not lock_path.exists()
