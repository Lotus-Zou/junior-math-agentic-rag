# -*- coding: utf-8 -*-
"""Sub-second deterministic responses for common guided-practice workflows."""

from __future__ import annotations

import hashlib
import uuid

from agentic_rag.deterministic_tutor import solve_curriculum_problem
from agentic_rag.local_intents import parse_local_command
from agentic_rag.math_validation import deterministic_equation_answer, deterministic_math_checks
from agentic_rag.response_contract import normalize_response

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
    "出一个几何",
    "出一道几何",
    "来一个几何",
    "来一道几何",
    "几何题我做",
    "geometry problem",
    "geometry exercise",
)

ZH_EXERCISES = (
    ("3x - 4 = 11", "先想怎样消去左边的 -4，再把 x 的系数化为 1。"),
    ("5x + 7 = 32", "先在等号两边做相同运算消去 +7，再处理 x 的系数。"),
    ("6x - 5 = 25", "移项时注意符号变化，完成后把结果代回原方程。"),
    ("4x + 9 = 29", "先消去常数项 +9，再把等式两边同时除以 4。"),
)

EN_EXERCISES = (
    ("3x - 4 = 11", "First eliminate -4 on the left, then make the coefficient of x equal to 1."),
    ("5x + 7 = 32", "Apply the same operation to both sides to eliminate +7, then handle the coefficient of x."),
    ("6x - 5 = 25", "Watch the sign when moving terms, then substitute your result into the original equation."),
    ("4x + 9 = 29", "Eliminate +9 first, then divide both sides by 4."),
)

ZH_GEOMETRY_EXERCISES = (
    (
        "在等腰三角形 ABC 中，AB = AC，顶角 ∠A = 40°。求 ∠B 和 ∠C 的度数。",
        "先利用等腰三角形两个底角相等，再结合三角形内角和为 180°。",
        ["等腰三角形", "三角形内角和"],
        (("b", "∠b"), ("c", "∠c"), ("70",)),
        "∠B = ∠C，且 ∠B + ∠C = 180° - 40° = 140°，所以 ∠B = ∠C = 70°。",
    ),
    (
        "一个三角形三个内角的度数之比为 2:3:4，求这三个内角的度数，并判断它是什么三角形。",
        "可把三个角分别设为 2k、3k、4k，再使用三角形内角和。",
        ["三角形内角和", "方程思想"],
        (("40",), ("60",), ("80",), ("锐角",)),
        "设三个角为 2k、3k、4k，则 9k = 180°，k = 20°，三个角为 40°、60°、80°，所以是锐角三角形。",
    ),
    (
        "在 △ABC 和 △DEF 中，已知 AB = DE、AC = DF、∠A = ∠D。请证明 △ABC ≌ △DEF，并写出判定依据。",
        "先确认已知角是不是两组已知边的夹角。",
        ["全等三角形", "边角边判定"],
        (("sas", "边角边"), ("全等", "≌")),
        "∠A 与 ∠D 分别是 AB、AC 和 DE、DF 的夹角，因此由边角边（SAS）可得 △ABC ≌ △DEF。",
    ),
    (
        "直角三角形 ABC 中，∠C = 90°，∠A = 35°。求 ∠B；若 CD 是 ∠ACB 的平分线，再求 ∠ACD。",
        "分别使用三角形内角和与角平分线的定义。",
        ["直角三角形", "角平分线", "三角形内角和"],
        (("55",), ("45",)),
        "∠B = 180° - 90° - 35° = 55°；CD 平分 90° 的 ∠ACB，所以 ∠ACD = 45°。",
    ),
)

EN_GEOMETRY_EXERCISES = (
    (
        "In isosceles triangle ABC, AB = AC and the vertex angle A is 40°. Find angles B and C.",
        "Use the equal base angles of an isosceles triangle and the 180° angle sum.",
        ["Isosceles triangles", "Triangle angle sum"],
        (("b",), ("c",), ("70",)),
        "Angles B and C are equal and sum to 140°, so B = C = 70°.",
    ),
    (
        "The angles of a triangle are in the ratio 2:3:4. Find all three angles and classify the triangle.",
        "Represent the angles by 2k, 3k, and 4k, then use their sum.",
        ["Triangle angle sum", "Algebraic modeling"],
        (("40",), ("60",), ("80",), ("acute",)),
        "Since 2k + 3k + 4k = 180°, k = 20°. The angles are 40°, 60°, and 80°, so it is acute.",
    ),
)

LINEAR_FUNCTION_EXERCISES = (
    (
        "\u5df2\u77e5\u4e00\u6b21\u51fd\u6570 y = 2x + 1\u3002\u5f53 x = 3 \u65f6\uff0c\u6c42 y\uff1b\u5f53 y = 9 \u65f6\uff0c\u6c42 x\u3002",
        "\u628a x \u7684\u503c\u4ee3\u5165 y = 2x + 1\uff1b\u518d\u628a y = 9 \u4ee3\u5165\u540e\u89e3\u4e00\u6b21\u65b9\u7a0b\u3002",
        ["\u4e00\u6b21\u51fd\u6570", "\u4ee3\u5165\u6c42\u503c"],
        "y = 7; x = 4",
    ),
    (
        "\u4e00\u6b21\u51fd\u6570 y = -3x + 8 \u7684\u56fe\u50cf\u4e0e y \u8f74\u4ea4\u4e8e\u54ea\u4e2a\u70b9\uff1f\u5f53 x = 2 \u65f6\uff0c\u6c42 y\u3002",
        "\u4ee4 x = 0 \u627e y \u8f74\u622a\u8ddd\uff1b\u518d\u628a x = 2 \u4ee3\u5165\u89e3\u6790\u5f0f\u3002",
        ["\u4e00\u6b21\u51fd\u6570", "\u659c\u622a\u5f0f", "\u63cf\u70b9\u4f5c\u56fe"],
        "(0, 8); y = 2",
    ),
)


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


def _active_exercise_topic(history: list[dict[str, str]], language: str) -> str | None:
    messages = "\n".join(str(item.get("content", "")) for item in history)
    geometry = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
    if any(problem in messages for problem, *_ in geometry):
        return "geometry"
    equations = EN_EXERCISES if language == "en" else ZH_EXERCISES
    if any(equation in messages for equation, _ in equations):
        return "algebra"
    if any(problem in messages for problem, *_ in LINEAR_FUNCTION_EXERCISES):
        return "linear_function"
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
        "\u8bf7\u5199\u51fa\u63a8\u5bfc\u8fc7\u7a0b\u540e\u53d1\u6765\uff0c\u6211\u4f1a\u68c0\u67e5\u6b65\u9aa4\uff0c\u6682\u4e0d\u5c55\u793a\u7b54\u6848\u3002"
    )


def _local_guided_exercise(
    topic: str,
    query: str,
    history: list[dict[str, str]],
    summary: str,
    language: str,
    difficulty_delta: int = 0,
) -> dict:
    if topic == "geometry":
        exercises = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
        problem, hint, knowledge_points, _, hidden_answer = exercises[_local_exercise_index(topic, history, difficulty_delta) % len(exercises)]
    elif topic == "algebra":
        exercises = EN_EXERCISES if language == "en" else ZH_EXERCISES
        problem, hint = exercises[_local_exercise_index(topic, history, difficulty_delta) % len(exercises)]
        knowledge_points = ["\u4e00\u5143\u4e00\u6b21\u65b9\u7a0b", "\u7b49\u5f0f\u7684\u57fa\u672c\u6027\u8d28"]
        hidden_answer = deterministic_equation_answer(f"\u89e3\u65b9\u7a0b {problem}", document_count=1, language=language)
    else:
        problem, hint, knowledge_points, hidden_answer = LINEAR_FUNCTION_EXERCISES[
            _local_exercise_index(topic, history, difficulty_delta) % len(LINEAR_FUNCTION_EXERCISES)
        ]
    answer = _exercise_answer(topic, problem, hint, language)
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": _TOPIC_CHAPTERS[topic]}]
    response.update({
        "intent": f"{topic}_exercise",
        "knowledge_points": list(knowledge_points),
        "validation_passed": True,
        "validation_evidence": {"kind": "deterministic", "passed": bool(hidden_answer), "template": topic},
        "exercise_answer_hidden": True,
        "exercise_state": {"topic": topic, "difficulty_delta": difficulty_delta},
        "metrics": {"tool_calls": 0},
    })
    return normalize_response(response, "guided_exercise")


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
    normalized = (query or "").strip().lower()
    return any(marker in normalized for marker in GEOMETRY_EXERCISE_MARKERS)


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
            if exercise[0] in content:
                return exercise
    return None


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
    return {
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
        "critic_report": {
            "is_valid": True,
            "factual_faithfulness": True,
            "math_logic_valid": True,
            "issues": [],
            "hallucination_detected": False,
            "validation_mode": "local_conversation_control",
        },
        "metrics": {
            "tool_calls": 0,
            "generation_mode": "local_conversation_control",
            "rerank_mode": "skipped_fast_path",
        },
    }


def _similar_exercise_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    context = _linear_context(history)
    if not _is_similar_request(query) or not context:
        return None
    exercises = EN_EXERCISES if language == "en" else ZH_EXERCISES
    seed = "\n".join(str(item.get("content", "")) for item in history)
    index = hashlib.sha256(seed.encode("utf-8")).digest()[0] % len(exercises)
    equation, hint = exercises[index]
    if language == "en":
        answer = (
            f"**Similar exercise**\nSolve: {equation}\n\n"
            f"**Hint**\n{hint} [1]\n\n"
            "Send me your steps when you finish. I will check the first mistake without revealing the answer in advance."
        )
    else:
        answer = (
            f"**类似练习**\n解方程：{equation}\n\n"
            f"**提示**\n{hint} [1]\n\n"
            "完成后把你的步骤发给我。我会先检查第一处错误，不提前公布答案。"
        )
    response = _base_response(query, answer, history, summary)
    response.update({
        "intent": "similar_exercise",
        "knowledge_points": ["一元一次方程", "等式的基本性质", "移项与验算"],
        "validation_passed": True,
        "critic_report": {
            "is_valid": True,
            "factual_faithfulness": True,
            "math_logic_valid": True,
            "issues": [],
            "hallucination_detected": False,
            "validation_mode": "local_template",
            "exercise_answer_hidden": True,
        },
        "metrics": {"tool_calls": 0, "generation_mode": "local_template", "rerank_mode": "skipped_fast_path"},
    })
    return response


def _geometry_exercise_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    if not _is_geometry_exercise_request(query):
        return None
    exercises = EN_GEOMETRY_EXERCISES if language == "en" else ZH_GEOMETRY_EXERCISES
    seed = f"{query}\n" + "\n".join(str(item.get("content", "")) for item in history)
    problem, hint, knowledge_points, _, _ = exercises[hashlib.sha256(seed.encode("utf-8")).digest()[0] % len(exercises)]
    if language == "en":
        answer = (
            f"**Geometry exercise**\n{problem}\n\n"
            f"**Hint**\n{hint} [1]\n\n"
            "Write out your reasoning and send it to me. I will check it step by step; the answer is hidden for now."
        )
    else:
        answer = (
            f"**几何练习**\n{problem}\n\n"
            f"**提示**\n{hint} [1]\n\n"
            "请写出推导过程后发给我，我会逐步检查；答案暂不展示。"
        )
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": "几何"}]
    response.update({
        "intent": "geometry_exercise",
        "knowledge_points": knowledge_points,
        "validation_passed": True,
        "critic_report": {
            "is_valid": True,
            "factual_faithfulness": True,
            "math_logic_valid": True,
            "issues": [],
            "hallucination_detected": False,
            "validation_mode": "local_template",
            "exercise_answer_hidden": True,
        },
        "metrics": {"tool_calls": 0, "generation_mode": "local_template", "rerank_mode": "skipped_fast_path"},
    })
    return response


def _geometry_answer_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    exercise = _geometry_context(history, language)
    if not exercise:
        return None
    _, hint, knowledge_points, required_groups, solution = exercise
    normalized = (query or "").strip().lower()
    asks_for_hint = any(marker in normalized for marker in ("提示", "不会", "没思路", "hint", "stuck"))
    passed = all(any(token.lower() in normalized for token in group) for group in required_groups)
    if language == "en":
        answer = (
            f"**Another hint**\n{hint}\nTry writing the key equality or theorem first; the answer remains hidden."
            if asks_for_hint else
            (f"**Check passed**\nYour conclusion is correct.\n\n**Reasoning**\n{solution}" if passed else
             f"**Not quite yet**\nThe conclusion is incomplete or contains an error. Recheck this point: {hint} I will keep the answer hidden while you revise it.")
        )
    else:
        answer = (
            f"**再给一个提示**\n{hint}\n先把关键等式或判定依据写出来，答案继续隐藏。"
            if asks_for_hint else
            (f"**检查通过**\n你的结论正确。\n\n**完整推导**\n{solution}" if passed else
             f"**暂未通过**\n你的结论还不完整或存在错误。请重点检查：{hint} 我会继续隐藏答案，等你修改后再核对。")
        )
    response = _base_response(query, answer, history, summary)
    response["sources"] = [{**CORE_SOURCE, "chapter": "几何"}]
    response.update({
        "intent": "geometry_answer_check" if not asks_for_hint else "geometry_hint",
        "knowledge_points": knowledge_points,
        "validation_passed": True,
        "critic_report": {
            "is_valid": True,
            "factual_faithfulness": True,
            "math_logic_valid": passed if not asks_for_hint else True,
            "issues": [] if passed or asks_for_hint else ["学生答案未满足本题的关键结论"],
            "hallucination_detected": False,
            "validation_mode": "local_geometry",
            "student_answer_correct": passed if not asks_for_hint else None,
        },
        "metrics": {"tool_calls": 0, "generation_mode": "local_geometry", "rerank_mode": "skipped_fast_path"},
    })
    return response


def _equation_response(query: str, history: list[dict[str, str]], summary: str, language: str) -> dict | None:
    student_attempt = _student_attempt(query)
    answer = deterministic_equation_answer(query, student_attempt, document_count=1, language=language)
    if not answer:
        return None
    checks = deterministic_math_checks(query, answer, 1)
    response = _base_response(query, answer, history, summary)
    response.update({
        "intent": "error_analysis" if student_attempt else "solve",
        "knowledge_points": ["一元一次方程", "等式的基本性质", "移项与验算"],
        "validation_passed": checks["passed"],
        "critic_report": {
            "is_valid": checks["passed"],
            "factual_faithfulness": True,
            "math_logic_valid": checks["passed"],
            "issues": checks["issues"],
            "hallucination_detected": False,
            "validation_mode": "local_sympy",
            "deterministic": checks,
        },
        "metrics": {"tool_calls": 0, "generation_mode": "local_sympy", "rerank_mode": "skipped_fast_path"},
    })
    return response


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
        "critic_report": {
            "is_valid": True,
            "factual_faithfulness": True,
            "math_logic_valid": True,
            "issues": [],
            "hallucination_detected": False,
            "validation_mode": "local_curriculum",
        },
        "metrics": {"tool_calls": 0, "generation_mode": "local_curriculum", "rerank_mode": "skipped_fast_path"},
    })
    return response


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
