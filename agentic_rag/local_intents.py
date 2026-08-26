"""Exact local routing for short study commands."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
import unicodedata


LocalAction = Literal["practice", "next_exercise", "adjust_difficulty", "new_question", "reset"]
Topic = Literal["geometry", "algebra", "linear_function"]


@dataclass(frozen=True)
class LocalCommand:
    action: LocalAction
    topic: Topic | None = None
    difficulty_delta: int = 0


_COMMANDS = {
    "\u51e0\u4f55": LocalCommand("practice", "geometry"),
    "geometry": LocalCommand("practice", "geometry"),
    "\u4ee3\u6570": LocalCommand("practice", "algebra"),
    "algebra": LocalCommand("practice", "algebra"),
    "\u4e00\u6b21\u51fd\u6570": LocalCommand("practice", "linear_function"),
    "linear function": LocalCommand("practice", "linear_function"),
    "\u518d\u6765\u4e00\u9053": LocalCommand("next_exercise"),
    "\u518d\u6765\u4e00\u9898": LocalCommand("next_exercise"),
    "another exercise": LocalCommand("next_exercise"),
    "\u96be\u4e00\u70b9": LocalCommand("adjust_difficulty", difficulty_delta=1),
    "harder": LocalCommand("adjust_difficulty", difficulty_delta=1),
    "\u7b80\u5355\u4e00\u70b9": LocalCommand("adjust_difficulty", difficulty_delta=-1),
    "easier": LocalCommand("adjust_difficulty", difficulty_delta=-1),
    "\u6362\u4e2a\u95ee\u9898": LocalCommand("new_question"),
    "\u6362\u4e00\u4e2a\u95ee\u9898": LocalCommand("new_question"),
    "\u6362\u9053\u9898": LocalCommand("new_question"),
    "\u6362\u4e00\u9053\u9898": LocalCommand("new_question"),
    "\u65b0\u95ee\u9898": LocalCommand("new_question"),
    "new question": LocalCommand("new_question"),
    "change problem": LocalCommand("new_question"),
    "switch problem": LocalCommand("new_question"),
    "\u91cd\u65b0\u5f00\u59cb": LocalCommand("reset"),
    "\u91cd\u7f6e": LocalCommand("reset"),
    "reset": LocalCommand("reset"),
}


def _normalize(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query or "").lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.rstrip(".?!,;:\u3002\uff01\uff1f\uff0c\uff1b\uff1a")


def parse_local_command(query: str, language: str) -> LocalCommand | None:
    """Return an exact short-command match without interpreting full problems."""
    del language
    return _COMMANDS.get(_normalize(query))
