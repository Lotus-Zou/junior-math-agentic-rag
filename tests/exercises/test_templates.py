from __future__ import annotations

from fractions import Fraction

import pytest

from agentic_rag.exercises.templates import TEMPLATE_REGISTRY, generate_from_template
from agentic_rag.exercises.validation import validate_generated_exercise


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


@pytest.mark.parametrize("template_id", sorted(TEMPLATE_REGISTRY))
@pytest.mark.parametrize("seed", range(50))
def test_every_registered_template_passes_independent_validation(template_id, seed):
    definition = TEMPLATE_REGISTRY[template_id]
    grade = definition.grades[0]
    difficulty = definition.difficulties[0]
    item = generate_from_template(template_id, difficulty, grade, seed)

    result = validate_generated_exercise(item)

    assert result.passed, result.issues
    assert item.template_id == template_id
    assert item.grade == grade
    assert item.difficulty == difficulty


def test_same_seed_is_content_deterministic_but_uses_opaque_ids():
    first = generate_from_template("fn.two_points.v1", 3, 9, 2026)
    second = generate_from_template("fn.two_points.v1", 3, 9, 2026)

    assert first.exercise_id != second.exercise_id
    assert first.model_dump(exclude={"exercise_id"}) == second.model_dump(
        exclude={"exercise_id"}
    )


@pytest.mark.parametrize(
    ("template_id", "difficulty", "grade"),
    [
        ("geo.isosceles.base_angles.v1", 5, 8),
        ("geo.congruence.sas_proof.v1", 3, 7),
        ("alg.linear_equation.v1", 2, 9),
        ("fn.two_points.v1", 1, 8),
    ],
)
def test_generation_rejects_unsupported_grade_or_difficulty(
    template_id, difficulty, grade
):
    with pytest.raises(ValueError, match="does not support"):
        generate_from_template(template_id, difficulty, grade, seed=1)


def test_validation_rejects_tampered_linear_equation_answer():
    item = generate_from_template("alg.linear_equation.v1", 2, 7, seed=4)
    tampered = item.model_copy(update={"answer_signature": "x=999"})

    result = validate_generated_exercise(tampered)

    assert not result.passed
    assert "answer_signature" in " ".join(result.issues)


def test_validation_rejects_tampered_hidden_solution_text():
    item = generate_from_template("alg.linear_equation.v1", 2, 7, seed=4)
    tampered = item.model_copy(update={"solution": "x = 999"})

    result = validate_generated_exercise(tampered)

    assert not result.passed
    assert "solution" in " ".join(result.issues)


def test_validation_rejects_degenerate_or_ambiguous_geometry():
    item = generate_from_template("geo.triangle.angle_ratio.v1", 3, 8, seed=4)
    tampered = item.model_copy(
        update={"parameters": {**item.parameters, "angles": [0, 90, 90]}}
    )

    assert not validate_generated_exercise(tampered).passed


def test_two_point_template_uses_exact_rational_slope():
    item = generate_from_template("fn.two_points.v1", 4, 9, seed=9)
    p = item.parameters
    slope = Fraction(p["y2"] - p["y1"], p["x2"] - p["x1"])

    assert slope == Fraction(p["k_numerator"], p["k_denominator"])
    assert validate_generated_exercise(item).passed
