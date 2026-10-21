"""Preference parsing and bounded mastery-based exercise progression."""

from __future__ import annotations

import re
from typing import Literal
import unicodedata

from agentic_rag.exercises.models import (
    ExerciseRequest,
    ExerciseSessionState,
    ExerciseTopic,
    ExerciseType,
)


Outcome = Literal["correct", "correct_after_hint", "incorrect", "unknown"]


def _clamp_difficulty(value: int) -> int:
    return max(1, min(5, value))


def next_difficulty(current: int, outcome: Outcome, explicit_delta: int = 0) -> int:
    if not isinstance(current, int) or isinstance(current, bool) or not 1 <= current <= 5:
        raise ValueError("current difficulty must be an integer from 1 through 5")
    if outcome not in {"correct", "correct_after_hint", "incorrect", "unknown"}:
        raise ValueError(f"unsupported exercise outcome: {outcome}")
    adjusted = _clamp_difficulty(current + explicit_delta)
    outcome_delta = {"correct": 1, "correct_after_hint": 0, "incorrect": -1, "unknown": 0}[
        outcome
    ]
    return _clamp_difficulty(adjusted + outcome_delta)


def update_mastery(
    mastery: dict[str, float], knowledge_points: list[str], outcome: Outcome
) -> dict[str, float]:
    observed = {
        "correct": 1.0,
        "correct_after_hint": 0.6,
        "incorrect": 0.0,
        "unknown": None,
    }.get(outcome)
    if outcome not in {"correct", "correct_after_hint", "incorrect", "unknown"}:
        raise ValueError(f"unsupported exercise outcome: {outcome}")
    updated = dict(mastery)
    if observed is None:
        return updated
    for point in knowledge_points:
        if not point:
            continue
        old = min(1.0, max(0.0, float(updated.get(point, 0.5))))
        updated[point] = min(1.0, max(0.0, 0.8 * old + 0.2 * observed))
    return updated


def _parse_topic(text: str) -> ExerciseTopic | None:
    if re.search(r"一次函数|线性函数|linear(?:\s+function)?|slope|intercept", text):
        return "linear_function"
    if re.search(r"几何|三角形|全等|geometry|triangle|congru", text):
        return "geometry"
    if re.search(r"代数|方程|因式|algebra|equation|factor", text):
        return "algebra"
    return None


def _parse_grade(text: str) -> int | None:
    chinese = {"七": 7, "八": 8, "九": 9}
    match = re.search(r"([七八九])年级", text)
    if match:
        return chinese[match.group(1)]
    match = re.search(r"(?:初中\s*)?([789])\s*年级", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:grade|year)\s*([789])\b|\b([789])(?:th)?\s*grade", text)
    if match:
        return int(match.group(1) or match.group(2))
    return None


def _parse_type(text: str) -> ExerciseType | None:
    if re.search(r"竞赛|奥数|competition|contest|olympiad", text):
        return "mixed"
    if re.search(r"证明|proof", text):
        return "proof"
    if re.search(r"应用|application|word problem", text):
        return "application"
    if re.search(r"综合|mixed|comprehensive", text):
        return "mixed"
    if re.search(r"计算|calculation|calculate", text):
        return "calculation"
    return None


def _explicit_difficulty(text: str, base: int) -> int | None:
    if re.search(r"竞赛|奥数|competition|contest|olympiad|非常难|挑战|very\s+hard|expert", text):
        return 5
    if re.search(r"困难|高难|challenging", text):
        return 4
    if re.search(r"难一点|更难|harder", text):
        return _clamp_difficulty(base + 1)
    if re.search(r"简单|基础|easy|basic", text):
        return 1
    if re.search(r"中等|medium|intermediate", text):
        return 2
    return None


def _mastery_adjusted_difficulty(current: ExerciseSessionState | None, base: int) -> int:
    if current is None or not current.mastery:
        return base
    average = sum(current.mastery.values()) / len(current.mastery)
    if average >= 0.8:
        return _clamp_difficulty(base + 1)
    if average < 0.4:
        return _clamp_difficulty(base - 1)
    return base


def parse_practice_preferences(
    query: str, current: ExerciseSessionState | None
) -> ExerciseRequest:
    text = unicodedata.normalize("NFKC", query or "").strip().lower()
    language = "en" if re.search(r"[a-z]", text) and not re.search(r"[\u4e00-\u9fff]", text) else "zh"
    competition_request = bool(
        re.search(r"竞赛|奥数|competition|contest|olympiad", text)
    )
    explicit_topic = _parse_topic(text)
    topic = (
        explicit_topic
        or ("algebra" if competition_request else None)
        or (current.current_topic if current else None)
        or "geometry"
    )
    grade = _parse_grade(text) or (current.current_grade if current else None) or 8
    base_difficulty = (current.current_difficulty if current else None) or 2
    difficulty = _explicit_difficulty(text, base_difficulty)
    if difficulty is None:
        difficulty = _mastery_adjusted_difficulty(current, base_difficulty)
    explicit_type = _parse_type(text)
    if explicit_type is not None:
        exercise_type = explicit_type
    elif explicit_topic is not None and current is not None and explicit_topic != current.current_topic:
        exercise_type = "calculation"
    else:
        exercise_type = (
            current.current_exercise_type if current and current.current_exercise_type else "calculation"
        )
    return ExerciseRequest(
        topic=topic,
        language=language,
        grade=grade,
        difficulty=difficulty,
        exercise_type=exercise_type,
        recent_fingerprints=list(current.recent_fingerprints) if current else [],
        recent_prompt_fingerprints=list(current.recent_prompt_fingerprints) if current else [],
        recent_answer_signatures=list(current.recent_answer_signatures) if current else [],
    )
