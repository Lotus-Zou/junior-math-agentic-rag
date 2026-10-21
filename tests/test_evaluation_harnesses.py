from typing import Any

from pydantic import BaseModel

from evaluation.pipeline_harness import check_case as check_pipeline_case
from evaluation.pipeline_harness import path_value as pipeline_path_value
from evaluation.skill_harness import check_case as check_skill_case
from evaluation.skill_harness import path_value as skill_path_value


class NullableResponse(BaseModel):
    answer: str = ""
    clarification: str | None = None


def test_skill_path_value_distinguishes_missing_from_present_null():
    assert skill_path_value({"response": {"clarification": None}}, "response.clarification") == (True, None)
    assert skill_path_value({"response": {}}, "response.clarification") == (False, None)
    assert skill_path_value(NullableResponse(), "clarification") == (True, None)
    assert skill_path_value(NullableResponse(), "missing") == (False, None)


def test_pipeline_path_value_distinguishes_missing_from_present_null():
    assert pipeline_path_value({"response": {"clarification": None}}, "response.clarification") == (True, None)
    assert pipeline_path_value({"response": {}}, "response.clarification") == (False, None)
    assert pipeline_path_value(NullableResponse(), "clarification") == (True, None)
    assert pipeline_path_value(NullableResponse(), "missing") == (False, None)


def test_skill_not_contains_scans_complete_public_response_surface():
    case = {
        "id": "skill-public-surface",
        "skill": "test.skill",
        "input": {},
        "expected_status": "OK",
        "not_contains": ["复杂推理服务", "超时"],
    }
    actual = {"handled": True, "response": {"answer": "正常", "conversation_summary": "复杂推理服务不可用"}}
    reasons = check_skill_case(case, "OK", actual)
    assert "public response contains '复杂推理服务'" in reasons


def test_pipeline_not_contains_scans_complete_rendered_public_surface():
    case = {
        "id": "pipeline-public-surface",
        "pipeline": "test.yaml",
        "input": {},
        "expected_node": "response_render",
        "not_contains": ["复杂推理服务", "超时"],
    }
    state = {"response_render": {"answer": "正常", "clarification": "复杂推理服务不可用"}}
    reasons = check_pipeline_case(case, state)
    assert "public response contains '复杂推理服务'" in reasons


def test_absent_paths_reject_present_null_for_skill_and_pipeline():
    skill_case = {
        "id": "skill-present-null",
        "skill": "test.skill",
        "input": {},
        "expected_status": "OK",
        "absent_paths": ["response.clarification"],
    }
    pipeline_case = {
        "id": "pipeline-present-null",
        "pipeline": "test.yaml",
        "input": {},
        "expected_node": "response_render",
        "absent_paths": ["response_render.clarification"],
    }
    assert "response.clarification should be absent" in check_skill_case(skill_case, "OK", {"response": {"clarification": None}})
    assert "response_render.clarification should be absent" in check_pipeline_case(pipeline_case, {"response_render": NullableResponse()})


def test_skill_legacy_assertions_and_unknown_keys_are_executable():
    case = {
        "id": "skill-legacy",
        "skill": "test.skill",
        "input": {},
        "expected_status": "OK",
        "expected": {"handled": True},
        "contains": {"handled": "True"},
        "answer_contains": "保留答案",
    }
    actual = {"handled": True, "response": {"answer": "保留答案"}}
    assert check_skill_case(case, "OK", actual) == []
    assert any(reason.startswith("answer missing") for reason in check_skill_case({**case, "answer_contains": "不存在"}, "OK", actual))
    assert "unsupported assertion keys: unexpected_assertion" in check_skill_case({**case, "unexpected_assertion": Any}, "OK", actual)


def test_pipeline_legacy_assertions_and_unknown_keys_are_executable():
    case = {
        "id": "pipeline-legacy",
        "pipeline": "test.yaml",
        "input": {},
        "expected_node": "response_render",
        "expected": {"validation_passed": True},
        "contains": {"language": "zh"},
        "answer_contains": "保留答案",
    }
    state = {"response_render": {"answer": "保留答案", "language": "zh", "validation_passed": True}}
    assert check_pipeline_case(case, state) == []
    assert "validation_passed=True" in check_pipeline_case({**case, "expected": {"validation_passed": False}}, state)
    assert "unsupported assertion keys: unexpected_assertion" in check_pipeline_case({**case, "unexpected_assertion": Any}, state)
