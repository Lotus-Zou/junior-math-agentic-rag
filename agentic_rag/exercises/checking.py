"""Deterministic student-answer checks against private generated exercise state."""

from __future__ import annotations

from fractions import Fraction
from collections import Counter
import re
from typing import Literal
import unicodedata

from agentic_rag.exercises.models import GeneratedExercise, StrictExerciseModel
from agentic_rag.exercises.store import ExerciseStore


class StudentAnswerCheck(StrictExerciseModel):
    status: Literal["correct", "incorrect", "expired"]
    passed: bool
    issues: list[str]


_NUMBER = r"[+-]?\d+(?:\s*/\s*[+-]?\d+)?"


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().lower()


def _fraction(value: str) -> Fraction:
    return Fraction(re.sub(r"\s+", "", value))


def _assignments(answer: str, symbol: str) -> list[Fraction]:
    pattern = re.compile(
        rf"(?<![\w])(?:∠\s*)?{re.escape(symbol)}\s*(?:=|为|是)\s*(?P<value>{_NUMBER})",
        flags=re.IGNORECASE,
    )
    return [_fraction(match.group("value")) for match in pattern.finditer(answer)]


def _all_assignments_equal(answer: str, symbol: str, expected: Fraction) -> bool:
    values = _assignments(answer, symbol)
    return bool(values) and all(value == expected for value in values)


def _check_isosceles(item: GeneratedExercise, answer: str) -> bool:
    base = Fraction(item.parameters["base_angle"])
    first_base, second_base = item.parameters["base_labels"]
    return _all_assignments_equal(answer, first_base, base) and _all_assignments_equal(
        answer, second_base, base
    )


def _check_angle_ratio(item: GeneratedExercise, answer: str) -> bool:
    expected_angles = sorted(item.parameters["angles"])
    degree_values = sorted(
        int(value)
        for value in re.findall(r"(?<![\d.])([1-9]\d{0,2})\s*(?:°|度)", answer)
    )
    classification = item.parameters["classification"]
    labels = {
        "acute": ("锐角", "acute"),
        "right": ("直角", "right"),
        "obtuse": ("钝角", "obtuse"),
    }[classification]
    contradictory = {
        "acute": ("直角", "钝角", "right", "obtuse"),
        "right": ("锐角", "钝角", "acute", "obtuse"),
        "obtuse": ("锐角", "直角", "acute", "right"),
    }[classification]
    observed = Counter(degree_values)
    expected = Counter(expected_angles)
    return (
        all(observed[value] >= count for value, count in expected.items())
        and any(label in answer for label in labels)
        and not any(label in answer for label in contradictory)
    )


def _check_sas(item: GeneratedExercise, answer: str) -> bool:
    p = item.parameters
    compact = re.sub(r"\s+", "", answer)
    first = p["first_triangle"].lower()
    second = p["second_triangle"].lower()
    negative = re.search(
        r"不全等|并非全等|不是(?:边角边|sas)|notcongruent|notsas|isn'tcongruent",
        compact,
        flags=re.IGNORECASE,
    )
    congruent = "全等" in answer or "≌" in answer or "congruent" in answer
    criterion = "边角边" in answer or re.search(r"\bsas\b", answer, re.IGNORECASE)
    return (
        negative is None
        and first in compact
        and second in compact
        and bool(congruent)
        and bool(criterion)
    )


def _check_linear_equation(item: GeneratedExercise, answer: str) -> bool:
    return _all_assignments_equal(answer, "x", Fraction(item.parameters["x"]))


def _check_difference(item: GeneratedExercise, answer: str) -> bool:
    expected = re.sub(r"\s+", "", item.answer_signature).lower()
    compact = re.sub(r"\s+", "", answer).replace("×", "").lower()
    if expected in compact:
        return True
    m, n = item.parameters["m"], item.parameters["n"]
    reversed_factors = f"({m}x+{n})({m}x-{n})"
    return reversed_factors in compact


def _check_slope_intercept(item: GeneratedExercise, answer: str) -> bool:
    p = item.parameters
    if not _all_assignments_equal(answer, "k", Fraction(p["k"])):
        return False
    if not _all_assignments_equal(answer, "b", Fraction(p["b"])):
        return False
    compact = re.sub(r"\s+", "", answer)
    return f"(0,{p['y0']})" in compact and f"(1,{p['y1']})" in compact


def _check_two_points(item: GeneratedExercise, answer: str) -> bool:
    p = item.parameters
    slope = Fraction(p["k_numerator"], p["k_denominator"])
    return _all_assignments_equal(answer, "k", slope) and _all_assignments_equal(
        answer, "b", Fraction(p["b"])
    )


_CHECKERS = {
    "geo.isosceles.base_angles.v1": _check_isosceles,
    "geo.triangle.angle_ratio.v1": _check_angle_ratio,
    "geo.congruence.sas_proof.v1": _check_sas,
    "alg.linear_equation.v1": _check_linear_equation,
    "alg.factorization.difference.v1": _check_difference,
    "fn.slope_intercept.v1": _check_slope_intercept,
    "fn.two_points.v1": _check_two_points,
}


def check_exercise_answer(
    store: ExerciseStore, exercise_id: str, student_answer: str
) -> StudentAnswerCheck:
    item = store.get_exercise(exercise_id)
    if item is None:
        return StudentAnswerCheck(
            status="expired",
            passed=False,
            issues=["exercise is missing or expired"],
        )
    answer = _normalized(student_answer)
    if not answer:
        return StudentAnswerCheck(
            status="incorrect",
            passed=False,
            issues=["student answer is empty"],
        )
    try:
        passed = bool(_CHECKERS[item.template_id](item, answer))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        passed = False
    return StudentAnswerCheck(
        status="correct" if passed else "incorrect",
        passed=passed,
        issues=[] if passed else ["student conclusion does not match the verified answer"],
    )
