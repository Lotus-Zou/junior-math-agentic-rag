import json
from pathlib import Path

import yaml

from agentic_rag.reliability import resolve_failure


FORBIDDEN_USER_COPY = (
    "复杂推理服务",
    "未在 8 秒内完成",
    "系统已自动记录为 bad case",
    "知识库没有召回",
    "Critic 服务异常",
    "RuntimeError",
    "provider secret",
)


def build_all_failure_responses():
    return [
        resolve_failure("证明这个结论", "zh", [], "", kind, ["internal"])
        for kind in (
            "timeout",
            "runtime_error",
            "retrieval_empty",
            "critic_rejected",
            "expired_exercise",
            "cache_error",
        )
    ]


def test_public_fallbacks_exclude_internal_copy():
    responses = build_all_failure_responses()

    assert responses
    for response in responses:
        assert all(token.lower() not in response["answer"].lower() for token in FORBIDDEN_USER_COPY)
        assert response["response_type"] in {"clarification_required", "supported_refusal"}
        assert response["answer"].strip()
        assert response["trace_id"]


def test_every_declared_chaos_case_has_an_executable_failure_policy():
    path = Path(__file__).parents[1] / "evaluation" / "chaos_cases.yaml"
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]

    assert {case["failure_kind"] for case in cases} == {
        "timeout",
        "runtime_error",
        "retrieval_empty",
        "critic_rejected",
        "expired_exercise",
        "cache_error",
    }
    for case in cases:
        response = resolve_failure(
            case["query"],
            case.get("language", "zh"),
            [],
            "",
            case["failure_kind"],
            [case["injection"]],
        )
        public = json.dumps(response, ensure_ascii=False)
        assert response["response_type"] == case["expected_response_type"]
        assert case["answer_contains"] in response["answer"]
        assert all(token.lower() not in response["answer"].lower() for token in FORBIDDEN_USER_COPY)
        assert response["trace_id"]
        assert case["injection"] not in public

