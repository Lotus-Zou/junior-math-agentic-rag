# -*- coding: utf-8 -*-
"""Sub-second deterministic responses for common guided-practice workflows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any
import unicodedata
import uuid

from agentic_rag.domain.schemas import PublicExerciseState
from agentic_rag.deterministic_tutor import solve_curriculum_problem
from agentic_rag.exercises.checking import check_exercise_answer
from agentic_rag.exercises.generator import (
    AdaptiveExerciseGenerator,
    ExerciseGenerationError,
)
from agentic_rag.exercises.models import (
    ExerciseRequest,
    ExerciseSessionState,
    GeneratedExercise,
    PublicExerciseState as AdaptivePublicExerciseState,
)
from agentic_rag.exercises.progression import next_difficulty, parse_practice_preferences
from agentic_rag.exercises.store import ExerciseStore
from agentic_rag.exercises.templates import TEMPLATE_REGISTRY
from agentic_rag.exercises.validation import validate_generated_exercise
from agentic_rag.local_intents import parse_local_command
from agentic_rag.math_validation import deterministic_equation_answer, deterministic_math_checks
from agentic_rag.response_contract import (
    ValidatedExerciseState,
    exercise_public_fingerprint,
    normalize_response,
    validated_exercise_state,
)

CORE_SOURCE = {
    "chunk_id": None,
    "source": "data\\初中数学核心知识.md",
    "chapter": "代数",
    "rank": 1,
}

_adaptive_exercise_store = ExerciseStore()
_adaptive_exercise_generator = AdaptiveExerciseGenerator()

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
    validator_kind: str
    parameters: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RenderedGeometryExercise:
    problem: str
    hint: str
    knowledge_points: tuple[str, ...]
    hidden_answer: str
    public_fingerprint: str


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
        "isosceles",
        (("vertex_angle", 40),),
    ),
    GeometryExerciseTemplate(
        "geometry-angle-ratio-234",
        "angle_ratio",
        (("first", 2), ("second", 3), ("third", 4)),
    ),
    GeometryExerciseTemplate(
        "geometry-sas-proof",
        "sas",
        (),
    ),
    GeometryExerciseTemplate(
        "geometry-right-bisector",
        "right_bisector",
        (("angle_a", 35),),
    ),
)

EN_GEOMETRY_EXERCISES = (
    ZH_GEOMETRY_EXERCISES[0],
    ZH_GEOMETRY_EXERCISES[1],
)

@dataclass(frozen=True)
class LocalExerciseTemplate:
    template_id: str
    topic: str
    parameters: tuple[tuple[str, int], ...]
    hidden_answer: str | None = None


LOCAL_TRANSITION_TEMPLATES = {
    "algebra": (
        LocalExerciseTemplate("topic-algebra-1", "algebra", (("coefficient", 3), ("constant", -4), ("total", 11)), "x = 5"),
        LocalExerciseTemplate("topic-algebra-2", "algebra", (("coefficient", 5), ("constant", 7), ("total", 32)), "x = 5"),
    ),
    "geometry": (
        LocalExerciseTemplate("topic-geometry-1", "geometry", (("vertex_angle", 40),)),
        LocalExerciseTemplate("topic-geometry-2", "geometry", (("vertex_angle", 60),)),
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
    if template.topic == "geometry":
        if template.hidden_answer is not None:
            return "", "", [], None
        hidden_answer = expected_answer
    else:
        hidden_answer = template.hidden_answer or ""
    private_state = validated_exercise_state(
        template.template_id,
        template.topic,
        hidden_answer,
        expected_answer,
        exercise_public_fingerprint(
            template.template_id, template.topic, problem, hint
        ),
    )
    return problem, hint, knowledge_points, private_state


def _known_exercise_provenance() -> list[tuple[str, str, str, str, str]]:
    known: list[tuple[str, str, str, str, str]] = []
    for exercise_language in ("zh", "en"):
        for topic, templates in LOCAL_TRANSITION_TEMPLATES.items():
            for template in templates:
                problem, hint, _, private_state = _validate_transition_template(
                    template, exercise_language
                )
                if private_state is not None:
                    known.append(
                        (
                            topic,
                            template.template_id,
                            private_state.public_fingerprint,
                            problem,
                            hint,
                        )
                    )
        geometry_templates = (
            EN_GEOMETRY_EXERCISES
            if exercise_language == "en"
            else ZH_GEOMETRY_EXERCISES
        )
        for template in geometry_templates:
            try:
                rendered = _render_geometry_template(template, exercise_language)
            except (KeyError, TypeError, ValueError):
                continue
            if _validate_geometry_template(template, exercise_language, rendered) is not None:
                known.append(
                    (
                        "geometry",
                        template.template_id,
                        rendered.public_fingerprint,
                        rendered.problem,
                        rendered.hint,
                    )
                )
        equation_templates = EN_EXERCISES if exercise_language == "en" else ZH_EXERCISES
        for template in equation_templates:
            known.append(
                (
                    "algebra",
                    template.template_id,
                    exercise_public_fingerprint(
                        template.template_id, "algebra", template.equation, template.hint
                    ),
                    template.equation,
                    template.hint,
                )
            )
    return known


def _sanitized_exercise_topic(
    value: Any, known: list[tuple[str, str, str, str, str]]
) -> str | None:
    try:
        state = PublicExerciseState.model_validate(value)
    except (TypeError, ValueError):
        return None
    if not state.template_id or not state.fingerprint:
        return None
    for topic, template_id, fingerprint, _, _ in known:
        if (
            state.topic == topic
            and state.template_id == template_id
            and state.fingerprint == fingerprint
        ):
            return topic
    return None


def _active_exercise_topic(history: list[dict[str, Any]], language: str) -> str | None:
    del language
    known = _known_exercise_provenance()
    for message in reversed(history or []):
        if message.get("role") not in {"tutor", "assistant"}:
            continue
        topic = _sanitized_exercise_topic(message.get("exercise_state"), known)
        if topic is not None:
            return topic
        content = unicodedata.normalize("NFKC", str(message.get("content", "")))
        for topic, _, _, problem, hint in known:
            if (
                unicodedata.normalize("NFKC", problem) in content
                and unicodedata.normalize("NFKC", hint) in content
            ):
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


def _public_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**item, "role": "tutor" if item.get("role") == "assistant" else item.get("role")}
        for item in history or []
    ]


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
            "fingerprint": private_state.public_fingerprint,
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
            "conversation_history": [*_public_history(history), {"role": "student", "content": query}, {"role": "tutor", "content": answer}],
            "conversation_summary": summary,
            "clarification": {"missing": [missing]},
            "metrics": {"tool_calls": 0},
        },
        "clarification_required",
    )


def _adaptive_state_clarification(
    query: str,
    history: list[dict[str, Any]],
    summary: str,
    language: str,
) -> dict:
    answer = (
        "That exercise is no longer available. Choose algebra, geometry, or linear functions and I will create a fresh verified problem."
        if language == "en"
        else "这道练习已失效。请选择代数、几何或一次函数，我会重新生成一道经过校验的新题。"
    )
    return normalize_response(
        {
            "answer": answer,
            "intent": "exercise_expired",
            "sources": [],
            "validation_passed": True,
            "conversation_history": [
                *_public_history(history),
                {"role": "student", "content": query},
                {"role": "tutor", "content": answer},
            ],
            "conversation_summary": summary,
            "clarification": {
                "missing": [
                    "a fresh exercise topic" if language == "en" else "新的练习主题"
                ]
            },
            "metrics": {"tool_calls": 0},
        },
        "clarification_required",
    )


def _resolve_adaptive_exercise(
    value: Any,
) -> tuple[AdaptivePublicExerciseState, ExerciseSessionState, GeneratedExercise] | None:
    try:
        public = AdaptivePublicExerciseState.model_validate(value)
    except (TypeError, ValueError):
        return None
    session = _adaptive_exercise_store.get_session(public.session_id)
    exercise = _adaptive_exercise_store.get_exercise(public.exercise_id)
    if session is None or exercise is None or session.current_exercise_id != exercise.exercise_id:
        return None
    expected = {
        "exercise_id": exercise.exercise_id,
        "session_id": session.session_id,
        "topic": exercise.topic,
        "grade": exercise.grade,
        "difficulty": exercise.difficulty,
        "exercise_type": exercise.exercise_type,
        "template_id": exercise.template_id,
        "fingerprint": exercise.fingerprint,
        "knowledge_points": exercise.knowledge_points,
    }
    if public.model_dump(mode="json") != expected:
        return None
    return public, session, exercise


def _explicit_adaptive_practice_request(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query or "").strip().lower()
    if solve_curriculum_problem(query, "zh") is not None or solve_curriculum_problem(
        query, "en"
    ) is not None:
        return False
    topic = bool(
        re.search(
            r"几何|代数|一次函数|geometry|algebra|linear\s+function",
            normalized,
        )
    )
    practice = bool(
        re.search(
            r"(?:来|出|生成).{0,10}(?:题|练习)|(?:题|练习).{0,8}(?:做|练)|"
            r"give\s+me|generate\s+(?:an?\s+)?(?:exercise|problem)|practice",
            normalized,
        )
    )
    preference = len(normalized) <= 32 and bool(
        re.search(r"难一点|更难|简单|基础|困难|harder|easier|easy|challenging", normalized)
    )
    return topic and (practice or preference)


def _nearest_supported_request(request: ExerciseRequest) -> ExerciseRequest | None:
    definitions = [
        definition
        for definition in TEMPLATE_REGISTRY.values()
        if definition.topic == request.topic
        and request.grade in definition.grades
        and (
            request.exercise_type == "mixed"
            or definition.exercise_type == request.exercise_type
        )
    ]
    available = sorted({level for item in definitions for level in item.difficulties})
    if not available:
        return None
    nearest = min(available, key=lambda level: (abs(level - request.difficulty), level))
    return request.model_copy(update={"difficulty": nearest}, deep=True)


def _adaptive_private_state(exercise: GeneratedExercise) -> ValidatedExerciseState | None:
    return validated_exercise_state(
        exercise.template_id,
        exercise.topic,
        exercise.solution,
        exercise.solution,
        exercise.fingerprint,
    )


def _adaptive_generated_response(
    request: ExerciseRequest,
    query: str,
    history: list[dict[str, Any]],
    summary: str,
    language: str,
    current: ExerciseSessionState | None,
) -> dict:
    try:
        exercise = _adaptive_exercise_generator.generate(request)
    except ExerciseGenerationError:
        fallback = _nearest_supported_request(request)
        if fallback is None:
            return _invalid_local_template_response(query, history, summary, language)
        exercise = _adaptive_exercise_generator.generate(fallback)
    if not validate_generated_exercise(exercise).passed:
        return _invalid_local_template_response(query, history, summary, language)
    private_state = _adaptive_private_state(exercise)
    if private_state is None:
        return _invalid_local_template_response(query, history, summary, language)
    public = _adaptive_exercise_store.start(
        exercise,
        mastery=current.mastery if current is not None else {},
        session_id=current.session_id if current is not None else None,
    )
    answer = _exercise_answer(exercise.topic, exercise.problem, exercise.hint, language)
    response = _base_response(query, answer, history, summary)
    response.update(
        {
            "intent": f"{exercise.topic}_exercise",
            "knowledge_points": exercise.knowledge_points,
            "sources": [
                {
                    **CORE_SOURCE,
                    "chapter": _TOPIC_CHAPTERS[exercise.topic],
                }
            ],
            "validation_passed": True,
            "validation_evidence": {"kind": "deterministic", "passed": True},
            "exercise_state": public.model_dump(mode="json"),
            "metrics": {"tool_calls": 0},
        }
    )
    return normalize_response(
        response,
        "guided_exercise",
        private_exercise=private_state,
    )


def _adaptive_command_response(
    query: str,
    history: list[dict[str, Any]],
    summary: str,
    language: str,
    exercise_state: Any,
) -> dict | None:
    command = parse_local_command(query, language)
    composite_practice = command is None and _explicit_adaptive_practice_request(query)
    if command is None and not composite_practice:
        return None
    if command is not None and command.action in {"new_question", "reset"}:
        return None

    resolved = _resolve_adaptive_exercise(exercise_state) if exercise_state is not None else None
    current = resolved[1] if resolved is not None else None
    if (
        command is not None
        and command.action in {"next_exercise", "adjust_difficulty"}
        and current is None
    ):
        if exercise_state is not None:
            return _adaptive_state_clarification(query, history, summary, language)
        return None

    request = parse_practice_preferences(query, current)
    updates: dict[str, Any] = {"language": language}
    if command is not None and command.topic is not None:
        updates["topic"] = command.topic
    if command is not None and command.action == "adjust_difficulty" and current is not None:
        updates["difficulty"] = next_difficulty(
            current.current_difficulty or request.difficulty,
            "unknown",
            command.difficulty_delta,
        )
        updates["exercise_type"] = "mixed"
    request = request.model_copy(update=updates, deep=True)
    return _adaptive_generated_response(
        request,
        query,
        history,
        summary,
        language,
        current,
    )


def _looks_like_new_problem(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query or "").lower()
    return bool(
        re.search(
            r"解方程|已知.{2,}求|求证|一道.{0,4}题|solve\s+the\s+equation|"
            r"given.{2,}find|new\s+problem",
            normalized,
        )
    )


def _asks_for_solution_reveal(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query or "").strip().lower()
    keep_hidden = (
        "不要告诉我答案",
        "别告诉我答案",
        "不要给答案",
        "先不看答案",
        "答案继续隐藏",
        "don't tell me the answer",
        "do not tell me the answer",
        "without revealing the answer",
        "keep the answer hidden",
    )
    if any(marker in normalized for marker in keep_hidden) or re.search(
        r"(?:不要|别).{0,8}(?:告诉|给|公布|显示).{0,6}(?:答案|解答)|"
        r"\b(?:don't|do\s+not)\b.{0,24}\b(?:tell|show|give|reveal)\b.{0,16}\b(?:answer|solution)\b",
        normalized,
    ):
        return False
    chinese_markers = (
        "给出答案",
        "给我答案",
        "告诉我答案",
        "告诉我完整解答",
        "直接告诉我",
        "公布答案",
        "查看答案",
        "给我完整解答",
        "给出完整解答",
        "查看完整解答",
        "我要完整解答",
        "请详细解答",
        "我放弃",
    )
    if any(marker in normalized for marker in chinese_markers):
        return True
    return bool(
        re.search(
            r"\b(?:show|give|tell|reveal)\b.{0,24}\b(?:answer|solution)\b|"
            r"\bcomplete\s+solution\b|\bi\s+give\s+up\b",
            normalized,
        )
    )


def _adaptive_answer_response(
    query: str,
    history: list[dict[str, Any]],
    summary: str,
    language: str,
    exercise_state: Any,
) -> dict | None:
    if exercise_state is None or parse_local_command(query, language) is not None:
        return None
    resolved = _resolve_adaptive_exercise(exercise_state)
    if resolved is None:
        return _adaptive_state_clarification(query, history, summary, language)
    public, session, exercise = resolved
    normalized = unicodedata.normalize("NFKC", query or "").strip().lower()
    asks_for_solution = _asks_for_solution_reveal(query)
    asks_for_hint = not asks_for_solution and any(
        marker in normalized
        for marker in ("提示", "不会", "没思路", "hint", "stuck")
    )
    checked = check_exercise_answer(
        _adaptive_exercise_store,
        exercise.exercise_id,
        query,
    )
    if not asks_for_hint and not checked.passed and _looks_like_new_problem(query):
        return None

    if asks_for_solution:
        answer = (
            f"**Complete solution**\n{exercise.solution}\n\nCompare each step with your attempt, then try a new exercise of the same type."
            if language == "en"
            else f"**完整解答**\n{exercise.solution}\n\n请逐步对照你刚才的思路，再尝试一道同类型题巩固。"
        )
        outcome = "incorrect"
    elif asks_for_hint:
        answer = (
            f"**Another hint**\n{exercise.hint}\nWrite the key relation first; the answer remains hidden."
            if language == "en"
            else f"**再给一个提示**\n{exercise.hint}\n先写出关键关系，答案继续隐藏。"
        )
        outcome = "unknown"
    elif checked.passed:
        answer = (
            f"**Check passed**\nYour conclusion is correct.\n\n**Complete solution**\n{exercise.solution}"
            if language == "en"
            else f"**检查通过**\n你的结论正确。\n\n**完整解答**\n{exercise.solution}"
        )
        outcome = "correct"
    else:
        answer = (
            f"**Not quite yet**\nThe conclusion is incomplete or incorrect. Recheck: {exercise.hint} I will keep the answer hidden while you revise it."
            if language == "en"
            else f"**暂未通过**\n结论还不完整或存在错误。请重点检查：{exercise.hint} 我会继续隐藏答案，等你修改后再核对。"
        )
        outcome = "incorrect"
    _adaptive_exercise_store.record_outcome(
        session.session_id,
        exercise.exercise_id,
        outcome,
    )
    response = _base_response(query, answer, history, summary)
    response.update(
        {
            "intent": (
                "adaptive_solution_reveal"
                if asks_for_solution
                else "adaptive_hint"
                if asks_for_hint
                else "adaptive_answer_check"
            ),
            "knowledge_points": exercise.knowledge_points,
            "sources": [
                {
                    **CORE_SOURCE,
                    "chapter": _TOPIC_CHAPTERS[exercise.topic],
                }
            ],
            "validation_passed": True,
            "validation_evidence": {"kind": "deterministic", "passed": True},
            "exercise_state": public.model_dump(mode="json"),
            "metrics": {"tool_calls": 0},
        }
    )
    if asks_for_solution or (checked.passed and not asks_for_hint):
        return normalize_response(response, "verified_answer")
    private_state = _adaptive_private_state(exercise)
    if private_state is None:
        return _adaptive_state_clarification(query, history, summary, language)
    return normalize_response(
        response,
        "guided_exercise",
        private_exercise=private_state,
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


def _geometry_exercise_heading(language: str) -> tuple[str, str]:
    if language == "en":
        return "**Geometry exercise**", "**Hint**"
    return "**几何练习**", "**提示**"


def _parse_geometry_exercise_turn(content: str, language: str):
    heading, hint_heading = _geometry_exercise_heading(language)
    normalized = unicodedata.normalize("NFKC", content or "").replace("\r\n", "\n").strip()
    if normalized.count(heading) != 1 or normalized.count(hint_heading) != 1:
        return None
    match = re.fullmatch(
        rf"{re.escape(heading)}\n(?P<problem>[^\n]+)\n\n"
        rf"{re.escape(hint_heading)}\n(?P<hint>[^\n]+) \[1\]"
        r"(?:\n\n.*)?",
        normalized,
        flags=re.DOTALL,
    )
    if match is None:
        return None

    exercises = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
    matches = []
    rendered_exercises = []
    for template in exercises:
        try:
            rendered = _render_geometry_template(template, language)
        except (KeyError, TypeError, ValueError):
            continue
        normalized_problem = unicodedata.normalize("NFKC", rendered.problem)
        normalized_hint = unicodedata.normalize("NFKC", rendered.hint)
        rendered_exercises.append((template, normalized_problem, normalized_hint))
        if (
            match.group("problem") == normalized_problem
            and match.group("hint") == normalized_hint
        ):
            matches.append((template, rendered))
    if len(matches) != 1:
        return None

    template, rendered = matches[0]
    if any(
        candidate.template_id != template.template_id
        and (candidate_problem in normalized or candidate_hint in normalized)
        for candidate, candidate_problem, candidate_hint in rendered_exercises
    ):
        return None
    private_state = _validate_geometry_template(template, language, rendered)
    if private_state is None:
        return None
    return template, rendered, private_state


def _geometry_context(history: list[dict[str, str]], language: str):
    heading, _ = _geometry_exercise_heading(language)
    for message in reversed(history or []):
        if message.get("role") != "tutor":
            continue
        content = str(message.get("content", ""))
        if heading in unicodedata.normalize("NFKC", content):
            return _parse_geometry_exercise_turn(content, language)
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


def _render_geometry_template(
    template: GeometryExerciseTemplate, language: str
) -> RenderedGeometryExercise:
    if language not in {"zh", "en"}:
        raise ValueError("unsupported geometry exercise language")
    parameters = dict(template.parameters)
    if len(parameters) != len(template.parameters):
        raise ValueError("geometry template has duplicate parameters")
    if template.validator_kind == "isosceles":
        if set(parameters) != {"vertex_angle"}:
            raise ValueError("isosceles template has invalid parameters")
        vertex = parameters["vertex_angle"]
        if not 0 < vertex < 180 or (180 - vertex) % 2:
            raise ValueError("isosceles template has invalid triangle angles")
        base = (180 - vertex) // 2
        if language == "en":
            problem = (
                f"In isosceles triangle ABC, AB = AC and the vertex angle A is "
                f"{vertex}°. Find angles B and C."
            )
            hint = "Use the equal base angles of an isosceles triangle and the 180° angle sum."
            knowledge_points = ("Isosceles triangles", "Triangle angle sum")
            hidden_answer = (
                f"Angles B and C are equal and sum to {180 - vertex}°, so "
                f"B = C = {base}°."
            )
        else:
            problem = (
                f"在等腰三角形 ABC 中，AB = AC，顶角 ∠A = {vertex}°。"
                "求 ∠B 和 ∠C 的度数。"
            )
            hint = "先利用等腰三角形两个底角相等，再结合三角形内角和为 180°。"
            knowledge_points = ("等腰三角形", "三角形内角和")
            hidden_answer = (
                f"∠B = ∠C，且 ∠B + ∠C = 180° - {vertex}° = {180 - vertex}°，"
                f"所以 ∠B = ∠C = {base}°。"
            )
    elif template.validator_kind == "angle_ratio":
        if set(parameters) != {"first", "second", "third"}:
            raise ValueError("angle-ratio template has invalid parameters")
        ratio = [parameters[key] for key in ("first", "second", "third")]
        total = sum(ratio)
        if any(item <= 0 for item in ratio) or 180 % total:
            raise ValueError("angle-ratio template has invalid triangle angles")
        unit = 180 // total
        angles = [item * unit for item in ratio]
        classification = "acute" if max(angles) < 90 else "right" if max(angles) == 90 else "obtuse"
        if language == "en":
            problem = (
                f"The angles of a triangle are in the ratio {ratio[0]}:{ratio[1]}:{ratio[2]}. "
                "Find all three angles and classify the triangle."
            )
            hint = (
                f"Represent the angles by {ratio[0]}k, {ratio[1]}k, and {ratio[2]}k, "
                "then use their sum."
            )
            knowledge_points = ("Triangle angle sum", "Algebraic modeling")
            hidden_answer = (
                f"Since {ratio[0]}k + {ratio[1]}k + {ratio[2]}k = 180°, k = {unit}°. "
                f"The angles are {angles[0]}°, {angles[1]}°, and {angles[2]}°, "
                f"so it is {classification}."
            )
        else:
            problem = (
                f"一个三角形三个内角的度数之比为 {ratio[0]}:{ratio[1]}:{ratio[2]}，"
                "求这三个内角的度数，并判断它是什么三角形。"
            )
            hint = (
                f"可把三个角分别设为 {ratio[0]}k、{ratio[1]}k、{ratio[2]}k，"
                "再使用三角形内角和。"
            )
            knowledge_points = ("三角形内角和", "方程思想")
            zh_classification = {"acute": "锐角", "right": "直角", "obtuse": "钝角"}[
                classification
            ]
            hidden_answer = (
                f"设三个角为 {ratio[0]}k、{ratio[1]}k、{ratio[2]}k，则 {total}k = 180°，"
                f"k = {unit}°，三个角为 {angles[0]}°、{angles[1]}°、{angles[2]}°，"
                f"所以是{zh_classification}三角形。"
            )
    elif template.validator_kind == "sas":
        if parameters:
            raise ValueError("SAS template must not have numeric parameters")
        if language == "en":
            problem = (
                "In triangles ABC and DEF, AB = DE, AC = DF, and angle A = angle D. "
                "Prove that triangle ABC is congruent to triangle DEF and state the criterion."
            )
            hint = "Check whether the known equal angles are included between the known equal sides."
            knowledge_points = ("Congruent triangles", "SAS congruence")
            hidden_answer = (
                "Angles A and D are included between AB, AC and DE, DF respectively, "
                "so triangle ABC is congruent to triangle DEF by SAS."
            )
        else:
            problem = (
                "在 △ABC 和 △DEF 中，已知 AB = DE、AC = DF、∠A = ∠D。"
                "请证明 △ABC ≌ △DEF，并写出判定依据。"
            )
            hint = "先确认已知角是不是两组已知边的夹角。"
            knowledge_points = ("全等三角形", "边角边判定")
            hidden_answer = (
                "∠A 与 ∠D 分别是 AB、AC 和 DE、DF 的夹角，因此由边角边（SAS）"
                "可得 △ABC ≌ △DEF。"
            )
    elif template.validator_kind == "right_bisector":
        if set(parameters) != {"angle_a"}:
            raise ValueError("right-bisector template has invalid parameters")
        angle_a = parameters["angle_a"]
        angle_b = 90 - angle_a
        if not 0 < angle_b < 90:
            raise ValueError("right-bisector template has invalid triangle angles")
        if language == "en":
            problem = (
                f"In right triangle ABC, angle C = 90° and angle A = {angle_a}°. "
                "Find angle B; if CD bisects angle ACB, also find angle ACD."
            )
            hint = "Use the triangle angle sum, then the definition of an angle bisector."
            knowledge_points = ("Right triangles", "Angle bisectors", "Triangle angle sum")
            hidden_answer = (
                f"Angle B = 180° - 90° - {angle_a}° = {angle_b}°. CD bisects the "
                "90° angle ACB, so angle ACD = 45°."
            )
        else:
            problem = (
                f"直角三角形 ABC 中，∠C = 90°，∠A = {angle_a}°。求 ∠B；"
                "若 CD 是 ∠ACB 的平分线，再求 ∠ACD。"
            )
            hint = "分别使用三角形内角和与角平分线的定义。"
            knowledge_points = ("直角三角形", "角平分线", "三角形内角和")
            hidden_answer = (
                f"∠B = 180° - 90° - {angle_a}° = {angle_b}°；"
                "CD 平分 90° 的 ∠ACB，所以 ∠ACD = 45°。"
            )
    else:
        raise ValueError("unsupported geometry validator kind")

    return RenderedGeometryExercise(
        problem=problem,
        hint=hint,
        knowledge_points=knowledge_points,
        hidden_answer=hidden_answer,
        public_fingerprint=exercise_public_fingerprint(
            template.template_id, "geometry", problem, hint
        ),
    )


def _validate_geometry_template(
    template: GeometryExerciseTemplate,
    language: str,
    rendered: RenderedGeometryExercise,
) -> ValidatedExerciseState | None:
    try:
        canonical = _render_geometry_template(template, language)
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(rendered, RenderedGeometryExercise) or rendered != canonical:
        return None
    return validated_exercise_state(
        template.template_id,
        "geometry",
        canonical.hidden_answer,
        rendered.hidden_answer,
        canonical.public_fingerprint,
    )


@dataclass(frozen=True)
class AngleAssignmentClaim:
    labels: tuple[str, ...]
    value: float
    affirmed: bool


@dataclass(frozen=True)
class AngleSequenceClaim:
    values: tuple[int, int, int]
    affirmed: bool


@dataclass(frozen=True)
class TriangleClassificationClaim:
    classification: str
    affirmed: bool


@dataclass(frozen=True)
class CongruenceClaim:
    left: str
    right: str
    affirmed: bool


def _claim_suffix_is_negative(answer: str, claim_end: int) -> bool:
    clause_tail = re.split(r"[。.!?;；]", answer[claim_end:], maxsplit=1)[0]
    return bool(
        re.match(
            r"\s*(?:[,，:：]\s*)?(?:(?:(?:这|该)?(?:一)?(?:结论|说法|等式)?\s*)?"
            r"(?:并非正确|并不正确|不正确|错误)|"
            r"(?:is|are)\s+(?:false|incorrect|wrong|not\s+correct)(?:\b|$))",
            clause_tail,
            flags=re.IGNORECASE,
        )
    )


def _claim_prefix_is_negative(answer: str, claim_start: int) -> bool:
    clause_prefix = re.split(r"[。.!?;；]", answer[:claim_start])[-1]
    english_polarity = r"(?:false|incorrect|wrong|not\s+correct)"
    return bool(
        re.search(
            rf"(?:"
            rf"(?:it|this|that)\s+is\s+{english_polarity}\s+that\s*|"
            rf"(?:(?:the\s+)?(?:following|claim|statement|conclusion)\s+is\s+"
            rf"{english_polarity}|{english_polarity})\s*[:：]\s*|"
            r"(?:错误|不正确)(?:的)?(?:结论|说法|等式)?\s*是\s*[:：]?\s*|"
            r"(?:该|这个|此)?(?:结论|说法|等式)\s*(?:是|为)\s*"
            r"(?:错误|不正确)(?:的)?\s*[:：]?\s*"
            r")$",
            clause_prefix,
            flags=re.IGNORECASE,
        )
    )


def _claim_is_negative(answer: str, claim_start: int, claim_end: int) -> bool:
    return (
        _claim_prefix_is_negative(answer, claim_start)
        or _claim_suffix_is_negative(answer, claim_end)
    )


_ANGLE_LABEL = r"(?:∠\s*(?:acd|b|c)|angles?\s+(?:acd|b|c)|\b(?:acd|b|c)\b)"


def _angle_label(value: str) -> str:
    return re.sub(r"∠|angles?|\s", "", value, flags=re.IGNORECASE).lower()


def _angle_assignment_claims(answer: str) -> list[AngleAssignmentClaim]:
    claims: list[AngleAssignmentClaim] = []

    def add(
        labels: tuple[str, ...],
        value: str,
        affirmed: bool,
        start: int,
        end: int,
    ) -> None:
        claims.append(
            AngleAssignmentClaim(
                labels=tuple(_angle_label(label) for label in labels),
                value=float(value),
                affirmed=affirmed and not _claim_is_negative(answer, start, end),
            )
        )

    chain_pattern = re.compile(
        rf"(?P<first>{_ANGLE_LABEL})\s*=\s*(?P<second>{_ANGLE_LABEL})\s*=\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:°|degrees?)?",
        flags=re.IGNORECASE,
    )
    for match in chain_pattern.finditer(answer):
        add(
            (match.group("first"), match.group("second")),
            match.group("value"),
            True,
            match.start(),
            match.end(),
        )

    grouped_pattern = re.compile(
        rf"(?P<first>{_ANGLE_LABEL})\s*(?:和|与|and)\s*(?P<second>{_ANGLE_LABEL})"
        r"\s*(?:都|both)?\s*(?P<operator>都不是|均不为|are\s+not|aren't|"
        r"=|为|是|are|equal)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:°|degrees?)?",
        flags=re.IGNORECASE,
    )
    for match in grouped_pattern.finditer(answer):
        operator = re.sub(r"\s+", " ", match.group("operator").lower())
        add(
            (match.group("first"), match.group("second")),
            match.group("value"),
            operator not in {"都不是", "均不为", "are not", "aren't"},
            match.start(),
            match.end(),
        )

    direct_pattern = re.compile(
        rf"(?P<label>{_ANGLE_LABEL})\s*(?P<operator>不等于|不是|≠|!=|"
        r"is\s+not|isn't|does\s+not\s+equal|=|为|是|equals?|is)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:°|degrees?)?",
        flags=re.IGNORECASE,
    )
    for match in direct_pattern.finditer(answer):
        operator = re.sub(r"\s+", " ", match.group("operator").lower())
        add(
            (match.group("label"),),
            match.group("value"),
            operator
            not in {"不等于", "不是", "≠", "!=", "is not", "isn't", "does not equal"},
            match.start(),
            match.end(),
        )

    base_angles_pattern = re.compile(
        r"(?P<labels>两(?:个)?底角|两个底角|both\s+(?:the\s+)?base\s+angles)\s*"
        r"(?P<operator>均不为|都不是|are\s+not|aren't|均为|都是|都为|为|是|are)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:°|degrees?)?",
        flags=re.IGNORECASE,
    )
    for match in base_angles_pattern.finditer(answer):
        operator = re.sub(r"\s+", " ", match.group("operator").lower())
        add(
            ("b", "c"),
            match.group("value"),
            operator not in {"均不为", "都不是", "are not", "aren't"},
            match.start(),
            match.end(),
        )
    return claims


def _expected_angle_assignments(
    answer: str, expected: dict[str, int]
) -> bool:
    relevant_claims = [
        claim
        for claim in _angle_assignment_claims(answer)
        if any(label in expected for label in claim.labels)
    ]
    if not relevant_claims or any(not claim.affirmed for claim in relevant_claims):
        return False
    assignments: dict[str, list[float]] = {label: [] for label in expected}
    for claim in relevant_claims:
        for label in claim.labels:
            if label in expected:
                assignments[label].append(claim.value)
    return all(
        assignments[label]
        and all(value == expected_value for value in assignments[label])
        for label, expected_value in expected.items()
    )


_DEGREE_VALUE = r"(?P<{name}>\d+(?:\.0+)?)\s*(?:°|degrees?)"


def _angle_sequence_claims(answer: str) -> list[AngleSequenceClaim]:
    patterns = (
        re.compile(
            r"(?:三个角|三个内角|三内角|各角|角度)(?:的度数)?\s*(?:分别)?\s*"
            r"(?P<operator>不为|不是|为|是|=)\s*"
            + _DEGREE_VALUE.format(name="first")
            + r"\s*[、,，]\s*"
            + _DEGREE_VALUE.format(name="second")
            + r"\s*[、,，]\s*"
            + _DEGREE_VALUE.format(name="third"),
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:the\s+)?(?:three\s+)?angles(?:\s+respectively)?\s*"
            r"(?P<operator>are\s+not|aren't|are|measure)\s*(?:respectively\s+)?"
            + _DEGREE_VALUE.format(name="first")
            + r"\s*[,，]\s*"
            + _DEGREE_VALUE.format(name="second")
            + r"\s*(?:[,，]\s*(?:and\s+)?|and\s+)"
            + _DEGREE_VALUE.format(name="third"),
            flags=re.IGNORECASE,
        ),
    )
    claims = []
    for pattern in patterns:
        for match in pattern.finditer(answer):
            operator = re.sub(r"\s+", " ", match.group("operator").lower())
            values = tuple(
                int(float(match.group(name)))
                for name in ("first", "second", "third")
            )
            claims.append(
                AngleSequenceClaim(
                    values=values,
                    affirmed=operator not in {"不为", "不是", "are not", "aren't"}
                    and not _claim_is_negative(
                        answer, match.start(), match.end()
                    ),
                )
            )
    return claims


def _classification_claims(answer: str) -> list[TriangleClassificationClaim]:
    patterns = (
        re.compile(
            r"(?:(?:所以|因此|故)\s*)?(?:(?:该|这个|此)?三角形\s*)?"
            r"(?P<operator>不是|并非|是|为)\s*(?P<classification>锐角|直角|钝角)三角形",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:(?:the|this)\s+triangle|it)\s+"
            r"(?P<operator>is\s+not|isn't|is)\s+(?:an?\s+)?"
            r"(?P<classification>acute|right|obtuse)(?:\s+triangle)?",
            flags=re.IGNORECASE,
        ),
    )
    aliases = {
        "锐角": "acute",
        "直角": "right",
        "钝角": "obtuse",
        "acute": "acute",
        "right": "right",
        "obtuse": "obtuse",
    }
    claims = []
    for pattern in patterns:
        for match in pattern.finditer(answer):
            operator = re.sub(r"\s+", " ", match.group("operator").lower())
            claims.append(
                TriangleClassificationClaim(
                    classification=aliases[match.group("classification").lower()],
                    affirmed=operator not in {"不是", "并非", "is not", "isn't"}
                    and not _claim_is_negative(
                        answer, match.start(), match.end()
                    ),
                )
            )
    return claims


_TRIANGLE_NAME = r"(?:△\s*|triangle\s+)?[a-z]{3}"


def _triangle_name(value: str) -> str:
    return re.sub(r"△|triangle|\s", "", value, flags=re.IGNORECASE).lower()


def _congruence_claims(answer: str) -> list[CongruenceClaim]:
    patterns = (
        re.compile(
            rf"(?P<left>{_TRIANGLE_NAME})\s*(?P<operator>不全等于?|全等于?|≌|"
            r"is\s+not\s+congruent\s+to|isn't\s+congruent\s+to|"
            rf"is\s+congruent\s+to)\s*(?P<right>{_TRIANGLE_NAME})",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"(?P<left>{_TRIANGLE_NAME})\s*(?:与|和)\s*(?P<right>{_TRIANGLE_NAME})\s*"
            r"(?P<operator>不全等|全等)",
            flags=re.IGNORECASE,
        ),
    )
    claims = []
    for pattern in patterns:
        for match in pattern.finditer(answer):
            operator = re.sub(r"\s+", " ", match.group("operator").lower())
            claims.append(
                CongruenceClaim(
                    left=_triangle_name(match.group("left")),
                    right=_triangle_name(match.group("right")),
                    affirmed=operator
                    not in {"不全等", "不全等于", "is not congruent to", "isn't congruent to"}
                    and not _claim_is_negative(
                        answer, match.start(), match.end()
                    ),
                )
            )
    return claims


def _angle_ratio_answer_is_correct(
    template: GeometryExerciseTemplate, answer: str
) -> bool:
    parameters = dict(template.parameters)
    ratio = [parameters[key] for key in ("first", "second", "third")]
    unit = 180 // sum(ratio)
    expected_angles = tuple(item * unit for item in ratio)
    expected_classification = (
        "acute"
        if max(expected_angles) < 90
        else "right"
        if max(expected_angles) == 90
        else "obtuse"
    )
    angle_claims = _angle_sequence_claims(answer)
    classification_claims = _classification_claims(answer)
    return (
        bool(angle_claims)
        and bool(classification_claims)
        and all(
            claim.affirmed and claim.values == expected_angles
            for claim in angle_claims
        )
        and all(
            claim.affirmed and claim.classification == expected_classification
            for claim in classification_claims
        )
    )


def _sas_answer_is_correct(answer: str) -> bool:
    claims = _congruence_claims(answer)
    expected_triangles = {"abc", "def"}
    if not claims or any(
        not claim.affirmed or {claim.left, claim.right} != expected_triangles
        for claim in claims
    ):
        return False
    criterion_ok = bool(
        re.search(
            r"边角边|\bsas\b|\bside(?:-|\s+)angle(?:-|\s+)side\b",
            answer,
            flags=re.IGNORECASE,
        )
    )
    criterion_negated = bool(
        re.search(
            r"(?:不是|并非)\s*边角边|边角边\s*(?:并非正确|不正确|错误)|"
            r"\bnot\s+(?:sas|side(?:-|\s+)angle(?:-|\s+)side)\b|"
            r"\bsas\s+(?:is\s+)?(?:false|incorrect|not\s+correct)\b",
            answer,
            flags=re.IGNORECASE,
        )
    )
    contradictory_criterion = bool(
        re.search(
            r"边边边|角边角|角角边|\b(?:sss|asa|aas|rhs|hl)\b",
            answer,
            flags=re.IGNORECASE,
        )
    )
    return criterion_ok and not criterion_negated and not contradictory_criterion


def _geometry_answer_is_correct(
    template: GeometryExerciseTemplate, student_answer: str, language: str
) -> bool:
    normalized = unicodedata.normalize("NFKC", student_answer or "").strip().lower()
    if not normalized:
        return False
    try:
        rendered = _render_geometry_template(template, language)
    except (KeyError, TypeError, ValueError):
        return False
    if _validate_geometry_template(template, language, rendered) is None:
        return False

    parameters = dict(template.parameters)
    if template.validator_kind == "isosceles":
        base_angle = (180 - parameters["vertex_angle"]) // 2
        return _expected_angle_assignments(
            normalized, {"b": base_angle, "c": base_angle}
        )
    if template.validator_kind == "right_bisector":
        return _expected_angle_assignments(
            normalized, {"b": 90 - parameters["angle_a"], "acd": 45}
        )
    if template.validator_kind == "angle_ratio":
        return _angle_ratio_answer_is_correct(template, normalized)
    if template.validator_kind == "sas":
        return _sas_answer_is_correct(normalized)
    return False


def _base_response(query: str, answer: str, history: list[dict[str, str]], summary: str) -> dict:
    return {
        "answer": answer,
        "trace_id": str(uuid.uuid4()),
        "sources": [dict(CORE_SOURCE)],
        "conversation_history": [
            *_public_history(history),
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
    public_fingerprint = exercise_public_fingerprint(
        template.template_id, "algebra", template.equation, template.hint
    )
    private_state = validated_exercise_state(
        template.template_id,
        "algebra",
        template.hidden_answer,
        solved_answer,
        public_fingerprint,
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
            "fingerprint": public_fingerprint,
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
    try:
        rendered = _render_geometry_template(template, language)
    except (KeyError, TypeError, ValueError):
        return _invalid_local_template_response(query, history, summary, language)
    private_state = _validate_geometry_template(template, language, rendered)
    if private_state is None:
        return _invalid_local_template_response(query, history, summary, language)
    if language == "en":
        answer = (
            f"**Geometry exercise**\n{rendered.problem}\n\n"
            f"**Hint**\n{rendered.hint} [1]\n\n"
            "Write out your reasoning and send it to me. I will check it step by step; the answer is hidden for now."
        )
    else:
        answer = (
            f"**几何练习**\n{rendered.problem}\n\n"
            f"**提示**\n{rendered.hint} [1]\n\n"
            "请写出推导过程后发给我，我会逐步检查；答案暂不展示。"
        )
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": "几何"}]
    response.update({
        "intent": "geometry_exercise",
        "knowledge_points": list(rendered.knowledge_points),
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_state": {
            "topic": "geometry",
            "difficulty_delta": 0,
            "template_id": template.template_id,
            "fingerprint": rendered.public_fingerprint,
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
    template, rendered, private_state = exercise
    normalized = (query or "").strip().lower()
    asks_for_solution = _asks_for_solution_reveal(query)
    asks_for_hint = not asks_for_solution and any(marker in normalized for marker in ("提示", "不会", "没思路", "hint", "stuck"))
    passed = _geometry_answer_is_correct(template, query, language)
    if language == "en":
        answer = (
            f"**Complete solution**\n{rendered.hidden_answer}\n\nCompare each step with your attempt, then try a similar exercise."
            if asks_for_solution else
            f"**Another hint**\n{rendered.hint}\nTry writing the key equality or theorem first; the answer remains hidden."
            if asks_for_hint else
            (f"**Check passed**\nYour conclusion is correct.\n\n**Reasoning**\n{rendered.hidden_answer}" if passed else
             f"**Not quite yet**\nThe conclusion is incomplete or contains an error. Recheck this point: {rendered.hint} I will keep the answer hidden while you revise it.")
        )
    else:
        answer = (
            f"**完整解答**\n{rendered.hidden_answer}\n\n请逐步对照你刚才的思路，再尝试一道同类型题巩固。"
            if asks_for_solution else
            f"**再给一个提示**\n{rendered.hint}\n先把关键等式或判定依据写出来，答案继续隐藏。"
            if asks_for_hint else
            (f"**检查通过**\n你的结论正确。\n\n**完整推导**\n{rendered.hidden_answer}" if passed else
             f"**暂未通过**\n你的结论还不完整或存在错误。请重点检查：{rendered.hint} 我会继续隐藏答案，等你修改后再核对。")
        )
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": "几何"}]
    response.update({
        "intent": (
            "geometry_solution_reveal"
            if asks_for_solution
            else "geometry_hint"
            if asks_for_hint
            else "geometry_answer_check"
        ),
        "knowledge_points": list(rendered.knowledge_points),
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": True},
        "exercise_state": {
            "topic": "geometry",
            "difficulty_delta": 0,
            "template_id": template.template_id,
            "fingerprint": rendered.public_fingerprint,
        },
        "metrics": {"tool_calls": 0},
    })
    if asks_for_solution or (passed and not asks_for_hint):
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
    exercise_state: Any = None,
) -> dict | None:
    """Return a deterministic response when the request has a safe local implementation."""
    return (
        _adaptive_command_response(
            query,
            history,
            summary,
            language,
            exercise_state,
        )
        or _adaptive_answer_response(
            query,
            history,
            summary,
            language,
            exercise_state,
        )
        or _local_command_response(query, history, summary, language)
        or _new_question_response(query, language)
        or _geometry_exercise_response(query, history, summary, language)
        or _geometry_answer_response(query, history, summary, language)
        or _similar_exercise_response(query, history, summary, language)
        or _curriculum_response(query, history, summary, language)
        or _equation_response(query, history, summary, language)
    )
