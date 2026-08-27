# -*- coding: utf-8 -*-
"""Sub-second deterministic responses for common guided-practice workflows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata
import uuid

from agentic_rag.deterministic_tutor import solve_curriculum_problem
from agentic_rag.local_intents import parse_local_command
from agentic_rag.math_validation import deterministic_equation_answer, deterministic_math_checks
from agentic_rag.response_contract import (
    ValidatedExerciseState,
    normalize_response,
    validated_exercise_state,
)

CORE_SOURCE = {
    "chunk_id": None,
    "source": "data\\初中数学核心知识.md",
    "chapter": "代数",
    "rank": 1,
}

SIMILAR_MARKERS = (
    "再出一个类似的题",
    "再来一道类似",
    "出一道类似",
    "类似的题",
    "similar problem",
    "similar exercise",
    "another one like",
)

NEW_QUESTION_MARKERS = (
    "换个问题",
    "换一个问题",
    "换道题",
    "换一道题",
    "新问题",
    "new question",
    "change problem",
    "switch problem",
)

GEOMETRY_EXERCISE_MARKERS = (
    "出一个几何题",
    "出一道几何题",
    "来一个几何题",
    "来一道几何题",
    "出一个几何题我做做",
    "出一个几何体我做做",
    "几何题我做做",
    "geometry problem",
    "geometry exercise",
)


@dataclass(frozen=True)
class EquationExerciseTemplate:
    template_id: str
    equation: str
    hint: str
    hidden_answer: str


@dataclass(frozen=True)
class GeometryExerciseTemplate:
    template_id: str
    problem: str
    hint: str
    knowledge_points: tuple[str, ...]
    required_groups: tuple[tuple[str, ...], ...]
    hidden_answer: str
    validator_kind: str
    parameters: tuple[tuple[str, int], ...]


ZH_EXERCISES = (
    EquationExerciseTemplate("similar-equation-1", "3x - 4 = 11", "先想怎样消去左边的 -4，再把 x 的系数化为 1。", "x = 5"),
    EquationExerciseTemplate("similar-equation-2", "5x + 7 = 32", "先在等号两边做相同运算消去 +7，再处理 x 的系数。", "x = 5"),
    EquationExerciseTemplate("similar-equation-3", "6x - 5 = 25", "移项时注意符号变化，完成后把结果代回原方程。", "x = 5"),
    EquationExerciseTemplate("similar-equation-4", "4x + 9 = 29", "先消去常数项 +9，再把等式两边同时除以 4。", "x = 5"),
)

EN_EXERCISES = (
    EquationExerciseTemplate("similar-equation-1", "3x - 4 = 11", "First eliminate -4 on the left, then make the coefficient of x equal to 1.", "x = 5"),
    EquationExerciseTemplate("similar-equation-2", "5x + 7 = 32", "Apply the same operation to both sides to eliminate +7, then handle the coefficient of x.", "x = 5"),
    EquationExerciseTemplate("similar-equation-3", "6x - 5 = 25", "Watch the sign when moving terms, then substitute your result into the original equation.", "x = 5"),
    EquationExerciseTemplate("similar-equation-4", "4x + 9 = 29", "Eliminate +9 first, then divide both sides by 4.", "x = 5"),
)

ZH_GEOMETRY_EXERCISES = (
    GeometryExerciseTemplate(
        "geometry-isosceles-40",
        "在等腰三角形 ABC 中，AB = AC，顶角 ∠A = 40°。求 ∠B 和 ∠C 的度数。",
        "先利用等腰三角形两个底角相等，再结合三角形内角和为 180°。",
        ("等腰三角形", "三角形内角和"),
        (("b", "∠b"), ("c", "∠c"), ("70",)),
        "∠B = ∠C，且 ∠B + ∠C = 180° - 40° = 140°，所以 ∠B = ∠C = 70°。",
        "isosceles",
        (("vertex_angle", 40),),
    ),
    GeometryExerciseTemplate(
        "geometry-angle-ratio-234",
        "一个三角形三个内角的度数之比为 2:3:4，求这三个内角的度数，并判断它是什么三角形。",
        "可把三个角分别设为 2k、3k、4k，再使用三角形内角和。",
        ("三角形内角和", "方程思想"),
        (("40",), ("60",), ("80",), ("锐角",)),
        "设三个角为 2k、3k、4k，则 9k = 180°，k = 20°，三个角为 40°、60°、80°，所以是锐角三角形。",
        "angle_ratio",
        (("first", 2), ("second", 3), ("third", 4)),
    ),
    GeometryExerciseTemplate(
        "geometry-sas-proof",
        "在 △ABC 和 △DEF 中，已知 AB = DE、AC = DF、∠A = ∠D。请证明 △ABC ≌ △DEF，并写出判定依据。",
        "先确认已知角是不是两组已知边的夹角。",
        ("全等三角形", "边角边判定"),
        (("sas", "边角边"), ("全等", "≌")),
        "∠A 与 ∠D 分别是 AB、AC 和 DE、DF 的夹角，因此由边角边（SAS）可得 △ABC ≌ △DEF。",
        "sas",
        (),
    ),
    GeometryExerciseTemplate(
        "geometry-right-bisector",
        "直角三角形 ABC 中，∠C = 90°，∠A = 35°。求 ∠B；若 CD 是 ∠ACB 的平分线，再求 ∠ACD。",
        "分别使用三角形内角和与角平分线的定义。",
        ("直角三角形", "角平分线", "三角形内角和"),
        (("55",), ("45",)),
        "∠B = 180° - 90° - 35° = 55°；CD 平分 90° 的 ∠ACB，所以 ∠ACD = 45°。",
        "right_bisector",
        (("angle_a", 35),),
    ),
)

EN_GEOMETRY_EXERCISES = (
    GeometryExerciseTemplate(
        "geometry-isosceles-40",
        "In isosceles triangle ABC, AB = AC and the vertex angle A is 40°. Find angles B and C.",
        "Use the equal base angles of an isosceles triangle and the 180° angle sum.",
        ("Isosceles triangles", "Triangle angle sum"),
        (("b",), ("c",), ("70",)),
        "Angles B and C are equal and sum to 140°, so B = C = 70°.",
        "isosceles",
        (("vertex_angle", 40),),
    ),
    GeometryExerciseTemplate(
        "geometry-angle-ratio-234",
        "The angles of a triangle are in the ratio 2:3:4. Find all three angles and classify the triangle.",
        "Represent the angles by 2k, 3k, and 4k, then use their sum.",
        ("Triangle angle sum", "Algebraic modeling"),
        (("40",), ("60",), ("80",), ("acute",)),
        "Since 2k + 3k + 4k = 180°, k = 20°. The angles are 40°, 60°, and 80°, so it is acute.",
        "angle_ratio",
        (("first", 2), ("second", 3), ("third", 4)),
    ),
)

@dataclass(frozen=True)
class LocalExerciseTemplate:
    template_id: str
    topic: str
    parameters: tuple[tuple[str, int], ...]
    hidden_answer: str


LOCAL_TRANSITION_TEMPLATES = {
    "algebra": (
        LocalExerciseTemplate("topic-algebra-1", "algebra", (("coefficient", 3), ("constant", -4), ("total", 11)), "x = 5"),
        LocalExerciseTemplate("topic-algebra-2", "algebra", (("coefficient", 5), ("constant", 7), ("total", 32)), "x = 5"),
    ),
    "geometry": (
        LocalExerciseTemplate("topic-geometry-1", "geometry", (("vertex_angle", 40),), "\u2220B = \u2220C = 70\u00b0"),
        LocalExerciseTemplate("topic-geometry-2", "geometry", (("vertex_angle", 60),), "\u2220B = \u2220C = 60\u00b0"),
    ),
    "linear_function": (
        LocalExerciseTemplate("topic-linear-function-1", "linear_function", (("slope", 2), ("intercept", 1), ("x_value", 3), ("y_value", 9)), "y = 7; x = 4"),
        LocalExerciseTemplate("topic-linear-function-2", "linear_function", (("slope", -3), ("intercept", 8), ("x_value", 2), ("y_value", 2)), "y = 2; x = 2"),
    ),
}


_TOPIC_CHAPTERS = {
    "geometry": "\u51e0\u4f55",
    "algebra": "\u4ee3\u6570",
    "linear_function": "\u51fd\u6570",
}


def _local_exercise_index(topic: str, history: list[dict[str, str]], difficulty_delta: int) -> int:
    if difficulty_delta:
        return 1 if difficulty_delta > 0 else 0
    seed = topic + "\n" + "\n".join(str(item.get("content", "")) for item in history)
    return hashlib.sha256(seed.encode("utf-8")).digest()[0]


def _signed_expression(coefficient: int, variable: str, constant: int) -> str:
    sign = "+" if constant >= 0 else "-"
    return f"{coefficient}{variable} {sign} {abs(constant)}"


def _render_transition_template(template: LocalExerciseTemplate, language: str) -> tuple[str, str, list[str], str]:
    parameters = dict(template.parameters)
    if template.topic == "algebra":
        coefficient = parameters["coefficient"]
        constant = parameters["constant"]
        total = parameters["total"]
        if coefficient == 0 or (total - constant) % coefficient:
            raise ValueError("algebra template does not have an integral solution")
        answer = f"x = {(total - constant) // coefficient}"
        expression = _signed_expression(coefficient, "x", constant)
        if language == "en":
            return (
                f"Solve {expression} = {total}.",
                f"Apply the inverse of {constant:+d} to both sides, then divide by {coefficient}.",
                ["linear equation", "equality properties"],
                answer,
            )
        return (
            f"\u89e3\u65b9\u7a0b\uff1a{expression} = {total}\u3002",
            f"\u5148\u5728\u7b49\u53f7\u4e24\u8fb9\u505a\u76f8\u53cd\u8fd0\u7b97\u6d88\u53bb {constant:+d}\uff0c\u518d\u540c\u9664\u4ee5 {coefficient}\u3002",
            ["\u4e00\u5143\u4e00\u6b21\u65b9\u7a0b", "\u7b49\u5f0f\u7684\u57fa\u672c\u6027\u8d28"],
            answer,
        )
    if template.topic == "geometry":
        vertex_angle = parameters["vertex_angle"]
        remainder = 180 - vertex_angle
        if not 0 < vertex_angle < 180 or remainder % 2:
            raise ValueError("geometry template has invalid triangle angles")
        base_angle = remainder // 2
        answer = f"\u2220B = \u2220C = {base_angle}\u00b0"
        if language == "en":
            return (
                f"In isosceles triangle ABC, AB = AC and angle A = {vertex_angle}\u00b0. Find angles B and C.",
                "Use equal base angles and the 180-degree triangle angle sum.",
                ["isosceles triangles", "triangle angle sum"],
                answer,
            )
        return (
            f"\u5728\u7b49\u8170\u4e09\u89d2\u5f62 ABC \u4e2d\uff0cAB = AC\uff0c\u9876\u89d2 \u2220A = {vertex_angle}\u00b0\u3002\u6c42 \u2220B \u548c \u2220C\u3002",
            "\u5148\u5229\u7528\u7b49\u8170\u4e09\u89d2\u5f62\u4e24\u4e2a\u5e95\u89d2\u76f8\u7b49\uff0c\u518d\u4f7f\u7528\u4e09\u89d2\u5f62\u5185\u89d2\u548c 180\u00b0\u3002",
            ["\u7b49\u8170\u4e09\u89d2\u5f62", "\u4e09\u89d2\u5f62\u5185\u89d2\u548c"],
            answer,
        )
    if template.topic == "linear_function":
        slope = parameters["slope"]
        intercept = parameters["intercept"]
        x_value = parameters["x_value"]
        y_value = parameters["y_value"]
        if slope == 0 or (y_value - intercept) % slope:
            raise ValueError("linear-function template cannot be solved exactly")
        value_at_x = slope * x_value + intercept
        x_at_y = (y_value - intercept) // slope
        answer = f"y = {value_at_x}; x = {x_at_y}"
        expression = _signed_expression(slope, "x", intercept)
        if language == "en":
            return (
                f"For y = {expression}, find y when x = {x_value}, then find x when y = {y_value}.",
                "Substitute the given x first; then substitute the given y and solve the resulting linear equation.",
                ["linear function", "substitution"],
                answer,
            )
        return (
            f"\u5df2\u77e5\u4e00\u6b21\u51fd\u6570 y = {expression}\u3002\u5f53 x = {x_value} \u65f6\uff0c\u6c42 y\uff1b\u5f53 y = {y_value} \u65f6\uff0c\u6c42 x\u3002",
            "\u5148\u628a\u5df2\u77e5 x \u7684\u503c\u4ee3\u5165\u89e3\u6790\u5f0f\uff1b\u518d\u5c06\u5df2\u77e5 y \u7684\u503c\u4ee3\u5165\u540e\u89e3\u4e00\u6b21\u65b9\u7a0b\u3002",
            ["\u4e00\u6b21\u51fd\u6570", "\u4ee3\u5165\u6c42\u503c"],
            answer,
        )
    raise ValueError(f"unsupported local template topic: {template.topic}")


def _validate_transition_template(
    template: LocalExerciseTemplate, language: str
) -> tuple[str, str, list[str], ValidatedExerciseState | None]:
    try:
        problem, hint, knowledge_points, expected_answer = _render_transition_template(template, language)
    except (KeyError, TypeError, ValueError):
        return "", "", [], None
    private_state = validated_exercise_state(
        template.template_id,
        template.topic,
        template.hidden_answer,
        expected_answer,
    )
    return problem, hint, knowledge_points, private_state


def _active_exercise_topic(history: list[dict[str, str]], language: str) -> str | None:
    labels = (
        {
            "**Geometry exercise**": "geometry",
            "**Algebra exercise**": "algebra",
            "**Linear-function exercise**": "linear_function",
            "**Similar exercise**": "algebra",
        }
        if language == "en"
        else {
            "**几何练习**": "geometry",
            "**代数练习**": "algebra",
            "**一次函数练习**": "linear_function",
            "**类似练习**": "algebra",
        }
    )
    geometry = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
    equations = EN_EXERCISES if language == "en" else ZH_EXERCISES
    for message in reversed(history or []):
        content = str(message.get("content", ""))
        for label, topic in labels.items():
            if label in content:
                return topic
        if any(template.problem in content for template in geometry):
            return "geometry"
        if any(template.equation in content for template in equations):
            return "algebra"
        for topic, templates in LOCAL_TRANSITION_TEMPLATES.items():
            for template in templates:
                problem, _, _, private_state = _validate_transition_template(template, language)
                if private_state is not None and problem in content:
                    return topic
    return None


def _exercise_answer(topic: str, problem: str, hint: str, language: str) -> str:
    if language == "en":
        labels = {
            "geometry": "Geometry exercise",
            "algebra": "Algebra exercise",
            "linear_function": "Linear-function exercise",
        }
        return (
            f"**{labels[topic]}**\n{problem}\n\n**Hint**\n{hint} [1]\n\n"
            "Send your reasoning when you finish. I will check it before showing a solution."
        )
    labels = {
        "geometry": "\u51e0\u4f55\u7ec3\u4e60",
        "algebra": "\u4ee3\u6570\u7ec3\u4e60",
        "linear_function": "\u4e00\u6b21\u51fd\u6570\u7ec3\u4e60",
    }
    return (
        f"**{labels[topic]}**\n{problem}\n\n**\u63d0\u793a**\n{hint} [1]\n\n"
        "\u8bf7\u5199\u51fa\u63a8\u5bfc\u8fc7\u7a0b\u540e\u53d1\u6765\uff0c\u6211\u4f1a\u68c0\u67e5\u6b65\u9aa4\uff0c\u7b54\u6848\u6682\u4e0d\u5c55\u793a\u3002"
    )


def _invalid_local_template_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict:
    answer = (
        "I cannot prepare a reliable exercise for that request yet. Please choose algebra, geometry, or linear functions."
        if language == "en"
        else "暂时无法为这个要求准备可靠的练习，请选择代数、几何或一次函数。"
    )
    response = _base_response(query, answer, history, summary)
    response.update({
        "intent": "practice_unavailable",
        "sources": [],
        "validation_passed": True,
        "clarification": {
            "missing": ["a supported learning topic" if language == "en" else "可用的练习主题"]
        },
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(response, "clarification_required")


def _local_guided_exercise(
    topic: str,
    query: str,
    history: list[dict[str, str]],
    summary: str,
    language: str,
    difficulty_delta: int = 0,
) -> dict:
    templates = LOCAL_TRANSITION_TEMPLATES[topic]
    template = templates[_local_exercise_index(topic, history, difficulty_delta) % len(templates)]
    problem, hint, knowledge_points, private_state = _validate_transition_template(template, language)
    if private_state is None:
        return _invalid_local_template_response(query, history, summary, language)
    answer = _exercise_answer(topic, problem, hint, language)
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": _TOPIC_CHAPTERS[topic]}]
    response.update({
        "intent": f"{topic}_exercise",
        "knowledge_points": knowledge_points,
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_state": {
            "topic": topic,
            "difficulty_delta": difficulty_delta,
            "template_id": template.template_id,
        },
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(
        response, "guided_exercise", private_exercise=private_state
    )

def _topic_clarification(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict:
    missing = "learning topic" if language == "en" else "\u60f3\u7ec3\u4e60\u7684\u5b66\u4e60\u4e3b\u9898"
    answer = (
        "Which learning topic should we practise: algebra, geometry, or linear functions?"
        if language == "en"
        else "\u8bf7\u544a\u8bc9\u6211\u60f3\u7ec3\u4e60\u7684\u5b66\u4e60\u4e3b\u9898\uff1a\u4ee3\u6570\u3001\u51e0\u4f55\u6216\u4e00\u6b21\u51fd\u6570\u3002"
    )
    return normalize_response(
        {
            "answer": answer,
            "intent": "clarification",
            "sources": [],
            "validation_passed": True,
            "conversation_history": [*(history or []), {"role": "student", "content": query}, {"role": "tutor", "content": answer}],
            "conversation_summary": summary,
            "clarification": {"missing": [missing]},
            "metrics": {"tool_calls": 0},
        },
        "clarification_required",
    )


def _local_command_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    command = parse_local_command(query, language)
    if command is None:
        return None
    if command.action in {"new_question", "reset"}:
        response = _new_question_response(query, language, force=command.action == "reset")
        return normalize_response(response, "clarification_required")
    if command.action == "practice":
        return _local_guided_exercise(command.topic, query, history, summary, language)
    topic = _active_exercise_topic(history, language)
    if topic is None:
        return _topic_clarification(query, history, summary, language)
    difficulty_delta = command.difficulty_delta if command.action == "adjust_difficulty" else 0
    return _local_guided_exercise(topic, query, history, summary, language, difficulty_delta)

def _student_attempt(query: str) -> str:
    for marker in ("学生错误作答：", "学生错误作答:", "错误步骤是", "我写成", "I wrote", "incorrect attempt:"):
        if marker in query:
            return query.split(marker, 1)[1].strip()
    return ""


def _is_similar_request(query: str) -> bool:
    normalized = (query or "").strip().lower()
    return any(marker in normalized for marker in SIMILAR_MARKERS)


def _is_new_question_request(query: str) -> bool:
    normalized = (query or "").strip().lower().strip("。！!？?，,.;；")
    return normalized in NEW_QUESTION_MARKERS


def _is_geometry_exercise_request(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query or "").lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.rstrip(".?!,;:。！？；：，")
    return normalized in GEOMETRY_EXERCISE_MARKERS


def _linear_context(history: list[dict[str, str]]) -> str:
    for message in reversed(history or []):
        content = str(message.get("content", ""))
        if "x" in content.lower() and deterministic_equation_answer(content):
            return content
    return ""


def _geometry_context(history: list[dict[str, str]], language: str):
    exercises = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
    for message in reversed(history or []):
        content = str(message.get("content", ""))
        for exercise in exercises:
            if exercise.problem in content:
                solved = _solve_geometry_hidden_answer(exercise, language)
                private_state = validated_exercise_state(
                    exercise.template_id,
                    "geometry",
                    exercise.hidden_answer,
                    solved,
                )
                if private_state is not None:
                    return exercise, private_state
    return None


def _solve_equation_hidden_answer(template: EquationExerciseTemplate) -> str:
    compact = template.equation.replace(" ", "")
    match = re.fullmatch(r"([+-]?\d*)x([+-]\d+)?=([+-]?\d+)", compact)
    if not match:
        return ""
    raw_coefficient, raw_constant, raw_total = match.groups()
    coefficient = -1 if raw_coefficient == "-" else 1 if raw_coefficient in {"", "+"} else int(raw_coefficient)
    constant = int(raw_constant or 0)
    total = int(raw_total)
    if coefficient == 0 or (total - constant) % coefficient:
        return ""
    return f"x = {(total - constant) // coefficient}"


def _solve_geometry_hidden_answer(
    template: GeometryExerciseTemplate, language: str
) -> str:
    parameters = dict(template.parameters)
    if template.validator_kind == "isosceles":
        vertex = parameters["vertex_angle"]
        if not 0 < vertex < 180 or (180 - vertex) % 2:
            return ""
        base = (180 - vertex) // 2
        if language == "en":
            return f"Angles B and C are equal and sum to {180 - vertex}°, so B = C = {base}°."
        return (
            f"∠B = ∠C，且 ∠B + ∠C = 180° - {vertex}° = {180 - vertex}°，"
            f"所以 ∠B = ∠C = {base}°。"
        )
    if template.validator_kind == "angle_ratio":
        ratio = [parameters[key] for key in ("first", "second", "third")]
        total = sum(ratio)
        if total <= 0 or 180 % total:
            return ""
        unit = 180 // total
        angles = [item * unit for item in ratio]
        if language == "en":
            return (
                f"Since {ratio[0]}k + {ratio[1]}k + {ratio[2]}k = 180°, k = {unit}°. "
                f"The angles are {angles[0]}°, {angles[1]}°, and {angles[2]}°, so it is acute."
            )
        return (
            f"设三个角为 {ratio[0]}k、{ratio[1]}k、{ratio[2]}k，则 {total}k = 180°，"
            f"k = {unit}°，三个角为 {angles[0]}°、{angles[1]}°、{angles[2]}°，所以是锐角三角形。"
        )
    if template.validator_kind == "sas" and language == "zh":
        return "∠A 与 ∠D 分别是 AB、AC 和 DE、DF 的夹角，因此由边角边（SAS）可得 △ABC ≌ △DEF。"
    if template.validator_kind == "right_bisector" and language == "zh":
        angle_a = parameters["angle_a"]
        angle_b = 90 - angle_a
        if not 0 < angle_b < 90:
            return ""
        return (
            f"∠B = 180° - 90° - {angle_a}° = {angle_b}°；CD 平分 90° 的 ∠ACB，"
            "所以 ∠ACD = 45°。"
        )
    return ""


def _base_response(query: str, answer: str, history: list[dict[str, str]], summary: str) -> dict:
    return {
        "answer": answer,
        "trace_id": str(uuid.uuid4()),
        "sources": [dict(CORE_SOURCE)],
        "conversation_history": [
            *(history or []),
            {"role": "student", "content": query},
            {"role": "tutor", "content": answer},
        ],
        "conversation_summary": summary,
        "cached": False,
    }


def _new_question_response(query: str, language: str, force: bool = False) -> dict | None:
    if not force and not _is_new_question_request(query):
        return None
    if language == "en":
        answer = (
            "**Switched to a new question**\n\n"
            "Send your new complete question, or choose a topic such as:\n"
            "- Algebra\n"
            "- Geometry\n"
            "- Linear functions\n\n"
            "The new question will not reuse the previous problem state."
        )
    else:
        answer = (
            "**已切换到新问题**\n\n"
            "请直接发送新的完整题目，或者告诉我想练习的知识点，例如：\n"
            "- 代数\n"
            "- 几何\n"
            "- 一次函数\n\n"
            "新问题将不会沿用上一题的解题状态。"
        )
    return normalize_response({
        "answer": answer,
        "trace_id": str(uuid.uuid4()),
        "sources": [],
        "conversation_history": [
            {"role": "student", "content": query},
            {"role": "tutor", "content": answer},
        ],
        "conversation_summary": "",
        "cached": False,
        "intent": "new_question",
        "knowledge_points": [],
        "validation_passed": True,
        "clarification": {
            "missing": ["the new complete problem" if language == "en" else "新的完整题目"]
        },
        "metrics": {"tool_calls": 0},
    }, "clarification_required")


def _similar_exercise_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    context = _linear_context(history)
    if not _is_similar_request(query) or not context:
        return None
    exercises = EN_EXERCISES if language == "en" else ZH_EXERCISES
    seed = "\n".join(str(item.get("content", "")) for item in history)
    index = hashlib.sha256(seed.encode("utf-8")).digest()[0] % len(exercises)
    template = exercises[index]
    solved_answer = _solve_equation_hidden_answer(template)
    private_state = validated_exercise_state(
        template.template_id,
        "algebra",
        template.hidden_answer,
        solved_answer,
    )
    if private_state is None:
        return _invalid_local_template_response(query, history, summary, language)
    if language == "en":
        answer = (
            f"**Similar exercise**\nSolve: {template.equation}\n\n"
            f"**Hint**\n{template.hint} [1]\n\n"
            "Send me your steps when you finish. I will check the first mistake without revealing the answer in advance."
        )
    else:
        answer = (
            f"**类似练习**\n解方程：{template.equation}\n\n"
            f"**提示**\n{template.hint} [1]\n\n"
            "完成后把你的步骤发给我。我会先检查第一处错误，不提前公布答案。"
        )
    response = _base_response(query, answer, history, summary)
    response.update({
        "intent": "similar_exercise",
        "knowledge_points": ["一元一次方程", "等式的基本性质", "移项与验算"],
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_state": {
            "topic": "algebra",
            "difficulty_delta": 0,
            "template_id": template.template_id,
        },
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(
        response, "guided_exercise", private_exercise=private_state
    )


def _geometry_exercise_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    if not _is_geometry_exercise_request(query):
        return None
    exercises = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
    seed = f"{query}\n" + "\n".join(str(item.get("content", "")) for item in history)
    template = exercises[hashlib.sha256(seed.encode("utf-8")).digest()[0] % len(exercises)]
    solved_answer = _solve_geometry_hidden_answer(template, language)
    private_state = validated_exercise_state(
        template.template_id,
        "geometry",
        template.hidden_answer,
        solved_answer,
    )
    if private_state is None:
        return _invalid_local_template_response(query, history, summary, language)
    if language == "en":
        answer = (
            f"**Geometry exercise**\n{template.problem}\n\n"
            f"**Hint**\n{template.hint} [1]\n\n"
            "Write out your reasoning and send it to me. I will check it step by step; the answer is hidden for now."
        )
    else:
        answer = (
            f"**几何练习**\n{template.problem}\n\n"
            f"**提示**\n{template.hint} [1]\n\n"
            "请写出推导过程后发给我，我会逐步检查；答案暂不展示。"
        )
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": "几何"}]
    response.update({
        "intent": "geometry_exercise",
        "knowledge_points": list(template.knowledge_points),
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_state": {
            "topic": "geometry",
            "difficulty_delta": 0,
            "template_id": template.template_id,
        },
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(
        response, "guided_exercise", private_exercise=private_state
    )


def _geometry_answer_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    exercise = _geometry_context(history, language)
    if not exercise:
        return None
    template, private_state = exercise
    normalized = (query or "").strip().lower()
    asks_for_hint = any(marker in normalized for marker in ("提示", "不会", "没思路", "hint", "stuck"))
    passed = all(
        any(token.lower() in normalized for token in group)
        for group in template.required_groups
    )
    if language == "en":
        answer = (
            f"**Another hint**\n{template.hint}\nTry writing the key equality or theorem first; the answer remains hidden."
            if asks_for_hint else
            (f"**Check passed**\nYour conclusion is correct.\n\n**Reasoning**\n{template.hidden_answer}" if passed else
             f"**Not quite yet**\nThe conclusion is incomplete or contains an error. Recheck this point: {template.hint} I will keep the answer hidden while you revise it.")
        )
    else:
        answer = (
            f"**再给一个提示**\n{template.hint}\n先把关键等式或判定依据写出来，答案继续隐藏。"
            if asks_for_hint else
            (f"**检查通过**\n你的结论正确。\n\n**完整推导**\n{template.hidden_answer}" if passed else
             f"**暂未通过**\n你的结论还不完整或存在错误。请重点检查：{template.hint} 我会继续隐藏答案，等你修改后再核对。")
        )
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": "几何"}]
    response.update({
        "intent": "geometry_answer_check" if not asks_for_hint else "geometry_hint",
        "knowledge_points": list(template.knowledge_points),
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_state": {
            "topic": "geometry",
            "difficulty_delta": 0,
            "template_id": template.template_id,
        },
        "metrics": {"tool_calls": 0},
    })
    if passed and not asks_for_hint:
        return normalize_response(response, "verified_answer")
    return normalize_response(
        response, "guided_exercise", private_exercise=private_state
    )


def _equation_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    student_attempt = _student_attempt(query)
    answer = deterministic_equation_answer(query, student_attempt, document_count=1, language=language)
    if not answer:
        return None
    checks = deterministic_math_checks(query, answer, 1)
    if checks.get("passed") is not True:
        return _invalid_local_template_response(query, history, summary, language)
    response = _base_response(query, answer, history, summary)
    response.update({
        "intent": "error_analysis" if student_attempt else "solve",
        "knowledge_points": ["一元一次方程", "等式的基本性质", "移项与验算"],
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(response, "verified_answer")


def _curriculum_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    solved = solve_curriculum_problem(query, language)
    if not solved:
        return None
    response = _base_response(query, solved["answer"], history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": solved["chapter"]}]
    response.update({
        "intent": solved["intent"],
        "knowledge_points": solved["knowledge_points"],
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(response, "verified_answer")


def build_fast_response(
    query: str,
    history: list[dict[str, str]],
    summary: str = "",
    language: str = "zh",
) -> dict | None:
    """Return a deterministic response when the request has a safe local implementation."""
    return (
        _local_command_response(query, history, summary, language)
        or _new_question_response(query, language)
        or _geometry_exercise_response(query, history, summary, language)
        or _geometry_answer_response(query, history, summary, language)
        or _similar_exercise_response(query, history, summary, language)
        or _curriculum_response(query, history, summary, language)
        or _equation_response(query, history, summary, language)
    )
