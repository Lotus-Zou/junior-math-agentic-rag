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


def test_geometry_production_context_rejects_tampered_displayed_hint():
    exercise = fast_path.build_fast_response("出一个几何体我做做", [], language="zh")
    template = fast_path.ZH_GEOMETRY_EXERCISES[0]
    rendered = fast_path._render_geometry_template(template, "zh")
    tampered_history = deepcopy(exercise["conversation_history"])
    tutor_turn = next(
        turn
        for turn in tampered_history
        if turn["role"] == "tutor" and rendered.problem in turn["content"]
    )
    tutor_turn["content"] = tutor_turn["content"].replace(
        rendered.hint, "错误提示：把顶角直接除以 2。"
    )

    assert fast_path._geometry_context(tampered_history, "zh") is None
    checked = fast_path.build_fast_response(
        "∠B = ∠C = 70°", tampered_history, language="zh"
    )
    assert checked is None or checked.get("intent") != "geometry_answer_check"


def test_geometry_context_requires_latest_tutor_problem_and_hint_from_same_template():
    first = fast_path._render_geometry_template(
        fast_path.ZH_GEOMETRY_EXERCISES[0], "zh"
    )
    second = fast_path._render_geometry_template(
        fast_path.ZH_GEOMETRY_EXERCISES[1], "zh"
    )
    valid = fast_path.build_fast_response("出一个几何体我做做", [], language="zh")
    mixed_message = (
        f"**几何练习**\n{first.problem}\n\n"
        f"**提示**\n{second.hint} [1]\n\n请写出推导过程。"
    )
    history = [
        *valid["conversation_history"],
        {"role": "tutor", "content": mixed_message},
    ]

    assert fast_path._geometry_context(history, "zh") is None


def test_geometry_context_ignores_student_owned_canonical_exercise_text():
    rendered = fast_path._render_geometry_template(
        fast_path.ZH_GEOMETRY_EXERCISES[0], "zh"
    )
    student_message = (
        f"**几何练习**\n{rendered.problem}\n\n"
        f"**提示**\n{rendered.hint} [1]"
    )

    assert fast_path._geometry_context(
        [{"role": "student", "content": student_message}], "zh"
    ) is None


@pytest.mark.parametrize(
    "student_answer",
    [
        "∠B = 70° 并非正确，∠C = 70°。",
        "B = 70 is false; C = 70.",
        "B = 70 is incorrect; C = 70.",
        "B = 70 is not correct; C = 70.",
    ],
)
def test_isosceles_claim_parser_rejects_scoped_negative_assignments(student_answer):
    template = fast_path.ZH_GEOMETRY_EXERCISES[0]
    language = "en" if " is " in student_answer else "zh"

    assert not fast_path._geometry_answer_is_correct(
        template, student_answer, language
    )


@pytest.mark.parametrize(
    ("template_index", "student_answer", "language"),
    [
        (
            0,
            "It is false that B = 70 degrees and C = 70 degrees.",
            "en",
        ),
        (0, "错误的是：B = 70°，C = 70°。", "zh"),
        (
            1,
            "It is false that the angles are 40 degrees, 60 degrees, and "
            "80 degrees, and the triangle is acute.",
            "en",
        ),
        (
            1,
            "错误的是：三个角分别为40°、60°、80°，所以是锐角三角形。",
            "zh",
        ),
        (
            2,
            "It is false that triangle ABC is congruent to triangle DEF by SAS.",
            "en",
        ),
        (2, "错误的是：△ABC ≌ △DEF，依据边角边。", "zh"),
    ],
)
def test_geometry_claim_parser_rejects_scoped_negative_prefix(
    template_index, student_answer, language
):
    assert not fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[template_index],
        student_answer,
        language,
    )


@pytest.mark.parametrize(
    ("template_index", "student_answer", "language"),
    [
        (
            0,
            "It is correct that B = 70 degrees and C = 70 degrees.",
            "en",
        ),
        (0, "正确的是：B = 70°，C = 70°。", "zh"),
        (
            1,
            "It is correct that the angles are 40 degrees, 60 degrees, and "
            "80 degrees, and the triangle is acute.",
            "en",
        ),
        (
            1,
            "正确的是：三个角分别为40°、60°、80°，所以是锐角三角形。",
            "zh",
        ),
        (
            2,
            "It is correct that triangle ABC is congruent to triangle DEF by SAS.",
            "en",
        ),
        (2, "正确的是：△ABC ≌ △DEF，依据边角边。", "zh"),
    ],
)
def test_geometry_claim_parser_accepts_positive_prefix_control(
    template_index, student_answer, language
):
    assert fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[template_index],
        student_answer,
        language,
    )


@pytest.mark.parametrize(
    ("student_answer", "language"),
    [
        ("两底角均为 70°。", "zh"),
        ("B = C = 70°。", "zh"),
        ("No angle is obtuse. B = C = 70 degrees.", "en"),
        ("No angle is obtuse. Both base angles are 70 degrees.", "en"),
    ],
)
def test_isosceles_claim_parser_accepts_equivalent_positive_answers(
    student_answer, language
):
    assert fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[0], student_answer, language
    )


@pytest.mark.parametrize(
    ("student_answer", "language"),
    [
        ("40°、60°、80°、锐角", "zh"),
        ("三个角分别为 80°、60°、40°，所以是锐角三角形。", "zh"),
        ("三个角分别为 40°、60°、80°，所以是直角三角形。", "zh"),
        ("40°, 60°, 80°, acute", "en"),
    ],
)
def test_angle_ratio_parser_rejects_tokens_wrong_order_and_wrong_classification(
    student_answer, language
):
    assert not fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[1], student_answer, language
    )


@pytest.mark.parametrize(
    ("student_answer", "language"),
    [
        (
            "设三个角为 2k、3k、4k，9k = 180°，k = 20°。"
            "三个角分别为 40°、60°、80°，所以是锐角三角形。"
            "最后验算三角形内角和仍为 180°。",
            "zh",
        ),
        (
            "The angles are 40 degrees, 60 degrees, and 80 degrees, "
            "and the triangle is acute. Their sum is 180 degrees.",
            "en",
        ),
    ],
)
def test_angle_ratio_parser_accepts_explicit_conclusion_with_trailing_derivation(
    student_answer, language
):
    assert fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[1], student_answer, language
    )


@pytest.mark.parametrize(
    ("student_answer", "language"),
    [
        ("由边角边可得 △DEF ≌ △ABC。", "zh"),
        ("△DEF 与 △ABC 全等，依据是 SAS。", "zh"),
        (
            "Triangle DEF is congruent to triangle ABC by side-angle-side.",
            "en",
        ),
    ],
)
def test_sas_parser_accepts_symmetric_congruence_and_equivalent_criterion(
    student_answer, language
):
    assert fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[2], student_answer, language
    )


@pytest.mark.parametrize(
    ("student_answer", "language"),
    [
        ("△ABC ≌ △DEF 并非正确，但依据写成 SAS。", "zh"),
        ("Triangle ABC is congruent to triangle DEF is false, by SAS.", "en"),
    ],
)
def test_sas_parser_rejects_negative_congruence_claim(student_answer, language):
    assert not fast_path._geometry_answer_is_correct(
        fast_path.ZH_GEOMETRY_EXERCISES[2], student_answer, language
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
