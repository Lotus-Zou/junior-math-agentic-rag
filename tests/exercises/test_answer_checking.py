from __future__ import annotations

import pytest

from agentic_rag.exercises.checking import check_exercise_answer
from agentic_rag.exercises.store import ExerciseStore
from agentic_rag.exercises.templates import generate_from_template


CASES = (
    ("geo.isosceles.base_angles.v1", 2, 8),
    ("geo.triangle.angle_ratio.v1", 3, 8),
    ("geo.congruence.sas_proof.v1", 3, 8),
    ("alg.linear_equation.v1", 2, 7),
    ("alg.factorization.difference.v1", 2, 8),
    ("fn.slope_intercept.v1", 2, 8),
    ("fn.two_points.v1", 3, 9),
)


def _correct_answer(item) -> str:
    p = item.parameters
    if item.template_id == "geo.isosceles.base_angles.v1":
        return f"∠B = {p['base_angle']}°，∠C = {p['base_angle']}°"
    if item.template_id == "geo.triangle.angle_ratio.v1":
        label = {"acute": "锐角", "right": "直角", "obtuse": "钝角"}[
            p["classification"]
        ]
        return "、".join(f"{value}°" for value in p["angles"]) + f"，是{label}三角形"
    if item.template_id == "geo.congruence.sas_proof.v1":
        return (
            f"△{p['first_triangle']} ≌ △{p['second_triangle']}，"
            "判定依据是 SAS（边角边）"
        )
    if item.template_id == "alg.linear_equation.v1":
        return f"x = {p['x']}"
    if item.template_id == "alg.factorization.difference.v1":
        return item.answer_signature
    if item.template_id == "fn.slope_intercept.v1":
        return (
            f"k = {p['k']}，b = {p['b']}，"
            f"两点是 (0, {p['y0']})、(1, {p['y1']})"
        )
    if item.template_id == "fn.two_points.v1":
        return f"k = {p['k_numerator']}/{p['k_denominator']}，b = {p['b']}"
    raise AssertionError(item.template_id)


@pytest.mark.parametrize(("template_id", "difficulty", "grade"), CASES)
@pytest.mark.parametrize("seed", range(10))
def test_correct_answers_pass_for_every_template_family(
    template_id, difficulty, grade, seed
):
    item = generate_from_template(template_id, difficulty, grade, seed)
    store = ExerciseStore(ttl_seconds=60)
    store.start(item, mastery={})

    result = check_exercise_answer(store, item.exercise_id, _correct_answer(item))

    assert result.passed, (item, result)
    assert result.status == "correct"


@pytest.mark.parametrize(("template_id", "difficulty", "grade"), CASES)
def test_unrelated_or_incorrect_answer_fails_for_every_template_family(
    template_id, difficulty, grade
):
    item = generate_from_template(template_id, difficulty, grade, seed=11)
    store = ExerciseStore(ttl_seconds=60)
    store.start(item, mastery={})

    result = check_exercise_answer(store, item.exercise_id, "x = 999，答案肯定正确")

    assert not result.passed
    assert result.status == "incorrect"


def test_negated_sas_conclusion_is_rejected():
    item = generate_from_template("geo.congruence.sas_proof.v1", 3, 8, seed=3)
    p = item.parameters
    store = ExerciseStore(ttl_seconds=60)
    store.start(item, mastery={})

    result = check_exercise_answer(
        store,
        item.exercise_id,
        f"△{p['first_triangle']} 与 △{p['second_triangle']} 不全等，不是 SAS",
    )

    assert not result.passed


def test_angle_ratio_reasoning_may_include_the_180_degree_sum():
    item = generate_from_template("geo.triangle.angle_ratio.v1", 3, 8, seed=3)
    p = item.parameters
    label = {"acute": "锐角", "right": "直角", "obtuse": "钝角"}[
        p["classification"]
    ]
    store = ExerciseStore(ttl_seconds=60)
    store.start(item, mastery={})
    answer = (
        f"三角形内角和是 180°，所以三个角为 "
        + "、".join(f"{value}°" for value in p["angles"])
        + f"，它是{label}三角形"
    )

    assert check_exercise_answer(store, item.exercise_id, answer).passed
