from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

import agentic_rag.fast_path as fast_path
import agentic_rag.response_contract as response_contract
import app as api


def _verified_cache_record(answer: str, query: str) -> tuple[dict, api.AskRequest]:
    request = api.AskRequest(query=query, language="zh")
    contract = {
        "validation_evidence": {"kind": "deterministic", "passed": True},
    }
    public = api._public_response(
        {
            "response_type": "verified_answer",
            "answer": answer,
            "validation_passed": True,
            "metrics": {"tool_calls": 0},
        },
        request,
        contract=contract,
    )
    contract = api._bind_contract_to_public_response(contract, public)
    return api._cache_record(public, contract), request


def _guided_cache_record() -> tuple[dict, api.AskRequest]:
    request = api.AskRequest(query="几何", language="zh")
    run = api._run_curriculum_skill(request)
    assert run is not None
    public = api._public_response(run.response, request, contract=run.contract)
    return api._cache_record(public, run.contract), request


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("problem", "在等腰三角形 ABC 中，顶角 ∠A = 60°。求两个底角。"),
        ("hint", "把 60° 平分即可。"),
        ("hidden_answer", "∠B = ∠C = 60°。"),
    ],
)
def test_geometry_validator_rejects_independently_tampered_rendered_fields(
    field, replacement
):
    template = fast_path.ZH_GEOMETRY_EXERCISES[0]
    rendered = fast_path._render_geometry_template(template, "zh")
    tampered = replace(rendered, **{field: replacement})

    assert fast_path._validate_geometry_template(template, "zh", tampered) is None


def test_geometry_structured_render_and_validation_positive_control():
    template = fast_path.ZH_GEOMETRY_EXERCISES[0]
    rendered = fast_path._render_geometry_template(template, "zh")
    private_state = fast_path._validate_geometry_template(template, "zh", rendered)

    assert "∠A = 40°" in rendered.problem
    assert "180°" in rendered.hint
    assert "70°" in rendered.hidden_answer
    assert private_state is not None
    assert private_state.template_id == template.template_id
    assert private_state.public_fingerprint == rendered.public_fingerprint


@pytest.mark.parametrize(
    "student_answer",
    [
        "∠B 不是 70°，∠C = 70°。",
        "∠B 不等于 60°，所以 ∠B = 70°，∠C = 70°。",
        "∠B = 70°，∠C = 70°，但 ∠B = 60°。",
        "∠B ≠ 60°，所以 ∠B = 70°，∠C = 70°。",
        "B、C、70°。",
    ],
)
def test_geometry_answer_checker_rejects_negation_and_contradiction(student_answer):
    exercise = fast_path.build_fast_response("出一个几何体我做做", [], language="zh")
    assert exercise is not None and "∠A = 40°" in exercise["answer"]

    checked = fast_path.build_fast_response(
        student_answer, exercise["conversation_history"], language="zh"
    )

    assert checked is not None
    assert checked["response_type"] == "guided_exercise"
    assert "检查通过" not in checked["answer"]


def test_geometry_answer_checker_accepts_semantically_correct_assignments():
    exercise = fast_path.build_fast_response("出一个几何体我做做", [], language="zh")
    assert exercise is not None

    checked = fast_path.build_fast_response(
        "由内角和与两底角相等可得 ∠B = 70°，∠C = 70°。",
        exercise["conversation_history"],
        language="zh",
    )

    assert checked is not None
    assert checked["response_type"] == "verified_answer"
    assert "检查通过" in checked["answer"]


def test_geometry_answer_checker_rejects_english_negation():
    template = fast_path.EN_GEOMETRY_EXERCISES[0]

    assert not fast_path._geometry_answer_is_correct(
        template, "Angles B and C are not 70 degrees.", "en"
    )
    assert fast_path._geometry_answer_is_correct(
        template, "B = 70 degrees and C = 70 degrees.", "en"
    )


def test_canonical_public_response_digest_is_stable_and_content_sensitive():
    first = {"answer": "x = 4", "metrics": {"tool_calls": 0}, "sources": []}
    reordered = {"sources": [], "metrics": {"tool_calls": 0}, "answer": "x = 4"}
    changed = {**first, "answer": "x = 5"}

    assert response_contract.public_response_digest(first) == (
        response_contract.public_response_digest(reordered)
    )
    assert response_contract.public_response_digest(first) != (
        response_contract.public_response_digest(changed)
    )


def test_verified_cache_rejects_public_answer_content_swap():
    record_a, request_a = _verified_cache_record("x = 4", "解方程 2x+3=11")
    record_b, _ = _verified_cache_record("x = 5", "解方程 3x+1=16")
    swapped = deepcopy(record_a)
    swapped["public"] = deepcopy(record_b["public"])

    with pytest.raises(api.ContractViolation):
        api._cached_response(swapped, request_a)


def test_verified_cache_writer_rejects_public_from_another_contract():
    record_a, _ = _verified_cache_record("x = 4", "解方程 2x+3=11")
    record_b, _ = _verified_cache_record("x = 5", "解方程 3x+1=16")

    with pytest.raises(api.ContractViolation):
        api._cache_record(record_b["public"], record_a["contract"])


def test_verified_cache_exact_writer_record_round_trips():
    record, request = _verified_cache_record("x = 4", "解方程 2x+3=11")

    restored = api._cached_response(record, request)

    assert restored["answer"] == "x = 4"
    assert record["contract"]["public_response_sha256"] == (
        response_contract.public_response_digest(record["public"])
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("template_id", "topic-geometry-other"),
        ("topic", "algebra"),
        ("fingerprint", "0" * 64),
    ],
)
def test_guided_cache_rejects_public_private_exercise_mismatch(field, replacement):
    record, request = _guided_cache_record()
    tampered = deepcopy(record)
    tampered["public"]["exercise_state"][field] = replacement
    tampered["contract"]["public_response_sha256"] = (
        response_contract.public_response_digest(tampered["public"])
    )

    with pytest.raises(api.ContractViolation):
        api._cached_response(tampered, request)


def test_guided_cache_exact_public_private_binding_round_trips():
    record, request = _guided_cache_record()
    public_state = record["public"]["exercise_state"]
    private_state = record["contract"]["private_exercise_state"]

    assert public_state["template_id"] == private_state["template_id"]
    assert public_state["topic"] == private_state["topic"]
    assert public_state["fingerprint"] == private_state["public_fingerprint"]
    assert api._cached_response(record, request)["response_type"] == "guided_exercise"
