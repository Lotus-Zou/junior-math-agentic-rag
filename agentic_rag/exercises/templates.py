"""Parameterized, seedable templates for verified junior mathematics practice."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import random
from typing import Any, Callable
import uuid

from agentic_rag.exercises.models import ExerciseTopic, ExerciseType, GeneratedExercise
from agentic_rag.exercises.validation import (
    compute_exercise_fingerprint,
    validate_generated_exercise,
)


@dataclass(frozen=True)
class RenderedExercise:
    problem: str
    hint: str
    solution: str
    answer_signature: str


@dataclass(frozen=True)
class TemplateDefinition:
    template_id: str
    topic: ExerciseTopic
    grades: tuple[int, ...]
    difficulties: tuple[int, ...]
    exercise_type: ExerciseType
    knowledge_points: tuple[str, ...]
    sampler: Callable[[random.Random, int], dict[str, Any]]
    renderer: Callable[[dict[str, Any], str], RenderedExercise]


def _signed_term(coefficient: int, variable: str = "x") -> str:
    if coefficient == 1:
        return variable
    if coefficient == -1:
        return f"-{variable}"
    return f"{coefficient}{variable}"


def _linear_expression(slope: Fraction, intercept: int) -> str:
    if slope.denominator == 1:
        leading = _signed_term(slope.numerator)
    else:
        leading = f"({slope.numerator}/{slope.denominator})x"
    if intercept > 0:
        return f"{leading} + {intercept}"
    if intercept < 0:
        return f"{leading} - {abs(intercept)}"
    return leading


def _sample_isosceles(rng: random.Random, difficulty: int) -> dict[str, Any]:
    choices = range(30, 121, 10) if difficulty == 1 else range(20, 151, 2)
    vertex = rng.choice(list(choices))
    if (180 - vertex) % 2:
        vertex += 1
    labels = rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 3)
    return {
        "triangle": "".join(labels),
        "vertex_label": labels[0],
        "base_labels": labels[1:],
        "equal_sides": [labels[0] + labels[1], labels[0] + labels[2]],
        "vertex_angle": vertex,
        "base_angle": (180 - vertex) // 2,
    }


def _render_isosceles(p: dict[str, Any], language: str) -> RenderedExercise:
    vertex, base = p["vertex_angle"], p["base_angle"]
    triangle = p["triangle"]
    vertex_label = p["vertex_label"]
    first_base, second_base = p["base_labels"]
    first_side, second_side = p["equal_sides"]
    signature = f"{first_base}={base};{second_base}={base}"
    if language == "en":
        return RenderedExercise(
            f"In isosceles triangle {triangle}, {first_side} = {second_side} and angle {vertex_label} = {vertex}°. Find angles {first_base} and {second_base}.",
            "Use the equal base angles and the 180° angle sum of a triangle.",
            f"{first_base} = {second_base} = (180° - {vertex}°) / 2 = {base}°.",
            signature,
        )
    return RenderedExercise(
        f"在等腰三角形 {triangle} 中，{first_side} = {second_side}，顶角 ∠{vertex_label} = {vertex}°，求 ∠{first_base} 和 ∠{second_base}。",
        "利用等腰三角形两底角相等和三角形内角和为 180°。",
        f"∠{first_base} = ∠{second_base} = (180° - {vertex}°) ÷ 2 = {base}°。",
        signature,
    )


def _sample_angle_ratio(rng: random.Random, difficulty: int) -> dict[str, Any]:
    totals = (9, 10, 12, 15, 18, 20) if difficulty == 2 else (12, 15, 18, 20, 30, 36)
    total = rng.choice(totals)
    while True:
        first = rng.randint(1, total - 2)
        second = rng.randint(1, total - first - 1)
        third = total - first - second
        common = math.gcd(first, math.gcd(second, third))
        ratio = (first // common, second // common, third // common)
        if sum(ratio) in {9, 10, 12, 15, 18, 20, 30, 36}:
            break
    unit = 180 // sum(ratio)
    angles = [unit * value for value in ratio]
    classification = (
        "acute" if max(angles) < 90 else "right" if max(angles) == 90 else "obtuse"
    )
    return {"ratio": list(ratio), "angles": angles, "classification": classification}


def _render_angle_ratio(p: dict[str, Any], language: str) -> RenderedExercise:
    ratio = p["ratio"]
    angles = p["angles"]
    ratio_text = ":".join(map(str, ratio))
    angle_text = ", ".join(f"{value}°" for value in angles)
    classification = p["classification"]
    signature = f"angles={','.join(map(str, angles))};class={classification}"
    if language == "en":
        return RenderedExercise(
            f"A triangle's angles are in the ratio {ratio_text}. Find all angles and classify the triangle.",
            "Write the angles as multiples of k, then use their sum of 180°.",
            f"The angles are {angle_text}; therefore the triangle is {classification}.",
            signature,
        )
    label = {"acute": "锐角", "right": "直角", "obtuse": "钝角"}[classification]
    return RenderedExercise(
        f"一个三角形三个内角的度数之比为 {ratio_text}，求各角度数并判断三角形类型。",
        "把三个角设为相应倍数的 k，再利用三角形内角和为 180°。",
        f"三个角依次为 {angle_text}，所以它是{label}三角形。",
        signature,
    )


def _sample_sas(rng: random.Random, _difficulty: int) -> dict[str, Any]:
    labels = rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 6)
    first, second = "".join(labels[:3]), "".join(labels[3:])
    first_vertex, second_vertex = labels[0], labels[3]
    first_sides = [labels[0] + labels[1], labels[0] + labels[2]]
    second_sides = [labels[3] + labels[4], labels[3] + labels[5]]
    return {
        "first_triangle": first,
        "second_triangle": second,
        "first_sides": list(first_sides),
        "second_sides": list(second_sides),
        "first_vertex": first_vertex,
        "second_vertex": second_vertex,
    }


def _render_sas(p: dict[str, Any], language: str) -> RenderedExercise:
    first, second = p["first_triangle"], p["second_triangle"]
    left, right = p["first_sides"], p["second_sides"]
    v1, v2 = p["first_vertex"], p["second_vertex"]
    conditions = f"{left[0]} = {right[0]}, {left[1]} = {right[1]}"
    signature = f"{first}~{second};criterion=SAS"
    if language == "en":
        return RenderedExercise(
            f"In triangles {first} and {second}, {conditions}, and angle {v1} = angle {v2}. Prove the triangles congruent.",
            "Check whether the equal angle is included between the two pairs of equal sides.",
            f"The equal angles are included, so triangle {first} is congruent to triangle {second} by SAS.",
            signature,
        )
    return RenderedExercise(
        f"在 △{first} 和 △{second} 中，已知 {conditions}，且 ∠{v1} = ∠{v2}，证明两个三角形全等。",
        "检查已知相等角是否分别夹在两组已知相等边之间。",
        f"∠{v1} 与 ∠{v2} 均为两组已知边的夹角，所以 △{first} ≌ △{second}（SAS）。",
        signature,
    )


def _sample_linear_equation(rng: random.Random, difficulty: int) -> dict[str, Any]:
    limit = {1: 8, 2: 15, 3: 30}[difficulty]
    x = rng.randint(-limit, limit)
    coefficients = [value for value in range(-9, 10) if value != 0]
    if difficulty == 1:
        coefficients = list(range(1, 7))
        x = rng.randint(1, limit)
    a = rng.choice(coefficients)
    b = rng.randint(-limit, limit)
    return {"a": a, "b": b, "c": a * x + b, "x": x}


def _render_linear_equation(p: dict[str, Any], language: str) -> RenderedExercise:
    a, b, c, x = p["a"], p["b"], p["c"], p["x"]
    left = _signed_term(a)
    if b > 0:
        left += f" + {b}"
    elif b < 0:
        left += f" - {abs(b)}"
    problem = f"{left} = {c}"
    if language == "en":
        return RenderedExercise(
            f"Solve the equation {problem}.",
            "Use equivalent transformations to isolate x, then substitute back to check.",
            f"The unique solution is x = {x}; substitution makes both sides equal to {c}.",
            f"x={x}",
        )
    return RenderedExercise(
        f"解方程：{problem}。",
        "利用等式性质逐步把 x 单独留在一边，最后代回原方程验算。",
        f"方程的唯一解是 x = {x}；代回后等式两边都等于 {c}。",
        f"x={x}",
    )


def _sample_difference(rng: random.Random, difficulty: int) -> dict[str, Any]:
    m = rng.randint(1 if difficulty == 2 else 2, 30)
    n = rng.randint(1, 60 if difficulty == 3 else 45)
    return {"m": m, "n": n, "left_square": m * m, "right_square": n * n}


def _render_difference(p: dict[str, Any], language: str) -> RenderedExercise:
    m, n = p["m"], p["n"]
    expression = f"{p['left_square']}x² - {p['right_square']}"
    signature = f"({m}x-{n})({m}x+{n})"
    if language == "en":
        return RenderedExercise(
            f"Factor {expression}.",
            "Recognize a difference of two squares: a² - b² = (a-b)(a+b).",
            f"{expression} = ({m}x - {n})({m}x + {n}).",
            signature,
        )
    return RenderedExercise(
        f"因式分解：{expression}。",
        "识别平方差公式 a² - b² = (a-b)(a+b)。",
        f"{expression} = ({m}x - {n})({m}x + {n})。",
        signature,
    )


def _sample_consecutive_squares(
    rng: random.Random, difficulty: int
) -> dict[str, Any]:
    upper = 1200 if difficulty == 5 else 400
    first = rng.randint(3, upper)
    numbers = [first, first + 1, first + 2]
    return {
        "first": first,
        "numbers": numbers,
        "square_sum": sum(value * value for value in numbers),
    }


def _render_consecutive_squares(
    p: dict[str, Any], language: str
) -> RenderedExercise:
    first = p["first"]
    numbers = p["numbers"]
    square_sum = p["square_sum"]
    signature = f"numbers={','.join(map(str, numbers))}"
    if language == "en":
        return RenderedExercise(
            f"The sum of the squares of three consecutive positive integers is {square_sum}. Find the three integers.",
            "Let the smallest integer be n, form a quadratic equation, and use the positive-integer condition.",
            (
                f"Let the integers be n, n+1, n+2. Then "
                f"n²+(n+1)²+(n+2)²={square_sum}, so "
                f"3n²+6n+5={square_sum}. The positive integer root is "
                f"n={first}; hence the integers are {numbers[0]}, {numbers[1]}, {numbers[2]}."
            ),
            signature,
        )
    return RenderedExercise(
        f"三个连续正整数的平方和为 {square_sum}，求这三个正整数。",
        "设最小的正整数为 n，列出一元二次方程，再结合正整数条件筛选根。",
        (
            f"设三个数为 n、n+1、n+2，则 "
            f"n²+(n+1)²+(n+2)²={square_sum}，整理得 "
            f"3n²+6n+5={square_sum}。取符合题意的正整数根 n={first}，"
            f"所以三个数依次为 {numbers[0]}、{numbers[1]}、{numbers[2]}。"
        ),
        signature,
    )


def _sample_slope_intercept(rng: random.Random, difficulty: int) -> dict[str, Any]:
    slopes = [value for value in range(-12, 13) if value != 0]
    if difficulty == 1:
        slopes = list(range(1, 21))
    k = rng.choice(slopes)
    b = rng.randint(-50 if difficulty == 1 else -40, 50 if difficulty == 1 else 40)
    return {"k": k, "b": b, "y0": b, "y1": k + b}


def _render_slope_intercept(p: dict[str, Any], language: str) -> RenderedExercise:
    k, b, y0, y1 = p["k"], p["b"], p["y0"], p["y1"]
    expression = _linear_expression(Fraction(k), b)
    signature = f"k={k};b={b};y0={y0};y1={y1}"
    if language == "en":
        return RenderedExercise(
            f"For y = {expression}, state the slope and y-intercept, then give the points at x = 0 and x = 1.",
            "Compare with y = kx + b and substitute the two x-values.",
            f"k = {k}, b = {b}; the points are (0, {y0}) and (1, {y1}).",
            signature,
        )
    return RenderedExercise(
        f"已知一次函数 y = {expression}，写出斜率和纵截距，并求 x = 0、x = 1 时的两个点。",
        "与 y = kx + b 对照，再分别代入两个 x 值。",
        f"k = {k}，b = {b}；两个点为 (0, {y0})、(1, {y1})。",
        signature,
    )


def _sample_two_points(rng: random.Random, difficulty: int) -> dict[str, Any]:
    denominator = 1 if difficulty < 4 else rng.choice((2, 3))
    numerator_choices = [value for value in range(-7, 8) if value != 0 and math.gcd(abs(value), denominator) == 1]
    numerator = rng.choice(numerator_choices)
    slope = Fraction(numerator, denominator)
    x1 = rng.randint(-5, 5)
    x2 = x1 + denominator * rng.choice((1, 2, 3))
    intercept = rng.randint(-8, 8)
    y1 = int(slope * x1 + intercept) if denominator == 1 else numerator * x1 // denominator + intercept
    if denominator != 1 and x1 % denominator:
        x1 -= x1 % denominator
        x2 = x1 + denominator * rng.choice((1, 2, 3))
    y1 = int(slope * x1 + intercept)
    y2 = int(slope * x2 + intercept)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "k_numerator": slope.numerator,
        "k_denominator": slope.denominator,
        "b": intercept,
    }


def _render_two_points(p: dict[str, Any], language: str) -> RenderedExercise:
    slope = Fraction(p["k_numerator"], p["k_denominator"])
    expression = _linear_expression(slope, p["b"])
    signature = f"k={slope.numerator}/{slope.denominator};b={p['b']}"
    points = f"({p['x1']}, {p['y1']}) and ({p['x2']}, {p['y2']})"
    if language == "en":
        return RenderedExercise(
            f"A line passes through {points}. Find its equation y = kx + b.",
            "First calculate the slope from the two points, then substitute one point to find b.",
            f"The slope is {slope}; substituting a point gives b = {p['b']}, so y = {expression}.",
            signature,
        )
    return RenderedExercise(
        f"一次函数图像经过点 ({p['x1']}, {p['y1']}) 和 ({p['x2']}, {p['y2']})，求解析式 y = kx + b。",
        "先用两点坐标求斜率，再代入其中一个点求 b。",
        f"斜率 k = {slope}，代入一点得 b = {p['b']}，所以 y = {expression}。",
        signature,
    )


TEMPLATE_REGISTRY: dict[str, TemplateDefinition] = {
    item.template_id: item
    for item in (
        TemplateDefinition("geo.isosceles.base_angles.v1", "geometry", (7, 8), (1, 2), "calculation", ("等腰三角形", "三角形内角和"), _sample_isosceles, _render_isosceles),
        TemplateDefinition("geo.triangle.angle_ratio.v1", "geometry", (7, 8), (2, 3), "calculation", ("三角形内角和", "比例"), _sample_angle_ratio, _render_angle_ratio),
        TemplateDefinition("geo.congruence.sas_proof.v1", "geometry", (8,), (3, 4), "proof", ("全等三角形", "边角边"), _sample_sas, _render_sas),
        TemplateDefinition("alg.linear_equation.v1", "algebra", (7,), (1, 2, 3), "calculation", ("一元一次方程", "等式性质"), _sample_linear_equation, _render_linear_equation),
        TemplateDefinition("alg.factorization.difference.v1", "algebra", (8,), (2, 3), "calculation", ("因式分解", "平方差公式"), _sample_difference, _render_difference),
        TemplateDefinition("alg.consecutive_squares.v1", "algebra", (9,), (4, 5), "mixed", ("一元二次方程", "整数问题"), _sample_consecutive_squares, _render_consecutive_squares),
        TemplateDefinition("fn.slope_intercept.v1", "linear_function", (8,), (1, 2), "calculation", ("一次函数", "斜率与截距"), _sample_slope_intercept, _render_slope_intercept),
        TemplateDefinition("fn.two_points.v1", "linear_function", (8, 9), (2, 3, 4), "application", ("一次函数", "待定系数法"), _sample_two_points, _render_two_points),
    )
}


def generate_from_template(
    template_id: str,
    difficulty: int,
    grade: int,
    seed: int | None = None,
    *,
    language: str = "zh",
) -> GeneratedExercise:
    try:
        definition = TEMPLATE_REGISTRY[template_id]
    except KeyError as exc:
        raise ValueError(f"unknown exercise template: {template_id}") from exc
    if grade not in definition.grades or difficulty not in definition.difficulties:
        raise ValueError(
            f"template {template_id} does not support grade {grade}, difficulty {difficulty}"
        )
    if language not in {"zh", "en"}:
        raise ValueError(f"unsupported exercise language: {language}")
    parameters = definition.sampler(random.Random(seed), difficulty)
    rendered = definition.renderer(parameters, language)
    fingerprint = compute_exercise_fingerprint(
        template_id=template_id,
        topic=definition.topic,
        grade=grade,
        difficulty=difficulty,
        exercise_type=definition.exercise_type,
        problem=rendered.problem,
        hint=rendered.hint,
        parameters=parameters,
    )
    item = GeneratedExercise(
        exercise_id=uuid.uuid4().hex,
        topic=definition.topic,
        grade=grade,
        difficulty=difficulty,
        exercise_type=definition.exercise_type,
        template_id=template_id,
        problem=rendered.problem,
        hint=rendered.hint,
        solution=rendered.solution,
        answer_signature=rendered.answer_signature,
        knowledge_points=list(definition.knowledge_points),
        parameters=parameters,
        fingerprint=fingerprint,
    )
    validation = validate_generated_exercise(item)
    if not validation.passed:
        raise ValueError(f"generated exercise failed validation: {validation.issues}")
    return item
