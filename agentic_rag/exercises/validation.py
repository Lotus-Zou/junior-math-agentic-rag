"""Independent deterministic checks for generated mathematics exercises."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import re
from typing import Any

from agentic_rag.exercises.models import GeneratedExercise, StrictExerciseModel


class ExerciseValidationResult(StrictExerciseModel):
    passed: bool
    issues: list[str]


_EXPECTED_METADATA = {
    "geo.isosceles.base_angles.v1": ("geometry", {7, 8}, {1, 2}, "calculation"),
    "geo.triangle.angle_ratio.v1": ("geometry", {7, 8}, {2, 3}, "calculation"),
    "geo.congruence.sas_proof.v1": ("geometry", {8}, {3, 4}, "proof"),
    "alg.linear_equation.v1": ("algebra", {7}, {1, 2, 3}, "calculation"),
    "alg.factorization.difference.v1": ("algebra", {8}, {2, 3}, "calculation"),
    "fn.slope_intercept.v1": ("linear_function", {8}, {1, 2}, "calculation"),
    "fn.two_points.v1": ("linear_function", {8, 9}, {2, 3, 4}, "application"),
}


def compute_exercise_fingerprint(
    *,
    template_id: str,
    topic: str,
    grade: int,
    difficulty: int,
    exercise_type: str,
    problem: str,
    hint: str,
    parameters: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "difficulty": difficulty,
            "exercise_type": exercise_type,
            "grade": grade,
            "hint": hint,
            "parameters": parameters,
            "problem": problem,
            "template_id": template_id,
            "topic": topic,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_isosceles(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    vertex = p["vertex_angle"]
    base = p["base_angle"]
    triangle = p["triangle"]
    vertex_label = p["vertex_label"]
    base_labels = p["base_labels"]
    equal_sides = p["equal_sides"]
    issues = []
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (vertex, base)):
        return ["isosceles parameters must be integers"]
    if not 0 < vertex < 180 or not 0 < base < 180 or vertex + 2 * base != 180:
        issues.append("isosceles parameters do not form the stated triangle")
    if (
        not isinstance(triangle, str)
        or len(triangle) != 3
        or len(set(triangle)) != 3
        or vertex_label != triangle[0]
        or base_labels != [triangle[1], triangle[2]]
        or equal_sides != [triangle[0] + triangle[1], triangle[0] + triangle[2]]
    ):
        issues.append("isosceles labels do not describe the stated equal sides")
    expected_signature = f"{base_labels[0]}={base};{base_labels[1]}={base}"
    if item.answer_signature != expected_signature:
        issues.append("answer_signature does not match the base angles")
    if not all(token in item.solution for token in (*base_labels, str(base))):
        issues.append("solution does not state the verified base angles")
    if not all(token in item.problem for token in (triangle, *equal_sides, str(vertex))):
        issues.append("problem does not contain every generated isosceles condition")
    return issues


def _validate_angle_ratio(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    ratio = p["ratio"]
    angles = p["angles"]
    classification = p["classification"]
    if (
        not isinstance(ratio, list)
        or not isinstance(angles, list)
        or len(ratio) != 3
        or len(angles) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in ratio + angles)
    ):
        return ["angle-ratio parameters must contain three integer ratios and angles"]
    issues = []
    if any(value <= 0 for value in ratio) or any(value <= 0 for value in angles):
        issues.append("angle-ratio parameters must be positive")
    if sum(angles) != 180 or any(angles[index] * sum(ratio) != ratio[index] * 180 for index in range(3)):
        issues.append("angle-ratio parameters do not produce a valid triangle")
    expected_classification = (
        "acute" if max(angles) < 90 else "right" if max(angles) == 90 else "obtuse"
    )
    if classification != expected_classification:
        issues.append("triangle classification does not match its angles")
    expected_signature = f"angles={','.join(map(str, angles))};class={expected_classification}"
    if item.answer_signature != expected_signature:
        issues.append("answer_signature does not match the angle-ratio solution")
    if not all(str(value) in item.solution for value in angles):
        issues.append("solution does not state every verified triangle angle")
    if ":".join(map(str, ratio)) not in item.problem:
        issues.append("problem does not contain the generated angle ratio")
    return issues


def _validate_sas(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    first = p["first_triangle"]
    second = p["second_triangle"]
    first_sides = p["first_sides"]
    second_sides = p["second_sides"]
    first_vertex = p["first_vertex"]
    second_vertex = p["second_vertex"]
    if not all(isinstance(value, str) for value in (first, second, first_vertex, second_vertex)):
        return ["SAS labels must be strings"]
    if not isinstance(first_sides, list) or not isinstance(second_sides, list):
        return ["SAS side pairs must be lists"]
    issues = []
    if (
        len(first) != 3
        or len(second) != 3
        or len(set(first)) != 3
        or len(set(second)) != 3
        or len(first_sides) != 2
        or len(second_sides) != 2
        or any(len(side) != 2 for side in [*first_sides, *second_sides])
    ):
        issues.append("SAS labels do not describe two nondegenerate triangles")
    if any(first_vertex not in side for side in first_sides) or any(
        second_vertex not in side for side in second_sides
    ):
        issues.append("the supplied angle is not included between the equal sides")
    expected_signature = f"{first}~{second};criterion=SAS"
    if item.answer_signature != expected_signature:
        issues.append("answer_signature does not match the SAS conclusion")
    if not all(token in item.solution for token in (first, second, "SAS")):
        issues.append("solution does not state the verified SAS conclusion")
    for token in [*first_sides, *second_sides, first_vertex, second_vertex]:
        if token not in item.problem:
            issues.append("problem does not contain every generated SAS condition")
            break
    return issues


def _validate_linear_equation(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    a, b, c, x = (p[key] for key in ("a", "b", "c", "x"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (a, b, c, x)):
        return ["linear-equation parameters must be integers"]
    issues = []
    if a == 0 or a * x + b != c:
        issues.append("linear equation does not have the generated unique solution")
    if item.answer_signature != f"x={x}":
        issues.append("answer_signature does not match the equation solution")
    if re.search(rf"(?<![\w.])x\s*=\s*{re.escape(str(x))}(?![\d.])", item.solution) is None:
        issues.append("solution does not state the verified equation solution")
    if str(c) not in item.problem:
        issues.append("problem does not contain the generated equation total")
    return issues


def _validate_difference_of_squares(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    m, n, left_square, right_square = (
        p[key] for key in ("m", "n", "left_square", "right_square")
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (m, n, left_square, right_square)
    ):
        return ["factorization parameters must be integers"]
    issues = []
    if m <= 0 or n <= 0 or left_square != m * m or right_square != n * n:
        issues.append("factorization parameters are not a difference of squares")
    if item.answer_signature != f"({m}x-{n})({m}x+{n})":
        issues.append("answer_signature does not match the factorization")
    compact_solution = re.sub(r"\s+", "", item.solution)
    if f"({m}x-{n})({m}x+{n})" not in compact_solution:
        issues.append("solution does not state the verified factorization")
    return issues


def _validate_slope_intercept(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    k, b, y0, y1 = (p[key] for key in ("k", "b", "y0", "y1"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (k, b, y0, y1)):
        return ["slope-intercept parameters must be integers"]
    issues = []
    if k == 0 or y0 != b or y1 != k + b:
        issues.append("function points do not satisfy the generated equation")
    if item.answer_signature != f"k={k};b={b};y0={y0};y1={y1}":
        issues.append("answer_signature does not match the slope-intercept data")
    if not all(token in item.solution for token in (f"k = {k}", f"b = {b}", f"(0, {y0})", f"(1, {y1})")):
        issues.append("solution does not state the verified slope-intercept results")
    return issues


def _validate_two_points(item: GeneratedExercise) -> list[str]:
    p = item.parameters
    values = [p[key] for key in ("x1", "y1", "x2", "y2", "k_numerator", "k_denominator", "b")]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return ["two-point parameters must be integers"]
    x1, y1, x2, y2, numerator, denominator, intercept = values
    issues = []
    if x1 == x2 or denominator <= 0:
        issues.append("two-point function has an undefined slope")
        return issues
    slope = Fraction(y2 - y1, x2 - x1)
    expected = Fraction(numerator, denominator)
    if slope != expected or Fraction(y1) != expected * x1 + intercept:
        issues.append("two points do not satisfy the generated linear function")
    if item.answer_signature != f"k={expected.numerator}/{expected.denominator};b={intercept}":
        issues.append("answer_signature does not match the two-point solution")
    if str(expected) not in item.solution or f"b = {intercept}" not in item.solution:
        issues.append("solution does not state the verified two-point result")
    return issues


_VALIDATORS = {
    "geo.isosceles.base_angles.v1": _validate_isosceles,
    "geo.triangle.angle_ratio.v1": _validate_angle_ratio,
    "geo.congruence.sas_proof.v1": _validate_sas,
    "alg.linear_equation.v1": _validate_linear_equation,
    "alg.factorization.difference.v1": _validate_difference_of_squares,
    "fn.slope_intercept.v1": _validate_slope_intercept,
    "fn.two_points.v1": _validate_two_points,
}


def validate_generated_exercise(item: GeneratedExercise) -> ExerciseValidationResult:
    issues: list[str] = []
    try:
        expected_topic, grades, difficulties, exercise_type = _EXPECTED_METADATA[
            item.template_id
        ]
        if (
            item.topic != expected_topic
            or item.grade not in grades
            or item.difficulty not in difficulties
            or item.exercise_type != exercise_type
        ):
            issues.append("exercise metadata does not match its template")
        expected_fingerprint = compute_exercise_fingerprint(
            template_id=item.template_id,
            topic=item.topic,
            grade=item.grade,
            difficulty=item.difficulty,
            exercise_type=item.exercise_type,
            problem=item.problem,
            hint=item.hint,
            parameters=item.parameters,
        )
        if item.fingerprint != expected_fingerprint:
            issues.append("fingerprint does not match the generated public problem")
        issues.extend(_VALIDATORS[item.template_id](item))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        issues.append(f"invalid generated exercise structure: {type(exc).__name__}")
    return ExerciseValidationResult(passed=not issues, issues=issues)
