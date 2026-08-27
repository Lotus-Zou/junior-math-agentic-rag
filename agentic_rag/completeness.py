"""Conservative, deterministic completeness analysis for student math requests."""

from __future__ import annotations

import re
import unicodedata

from agentic_rag.domain.schemas import CompletenessResult


_ZH_DEICTIC = {
    "这题怎么做",
    "这题怎么解",
    "怎么做",
    "怎么解",
    "不会",
    "没思路",
    "帮我看看",
}
_EN_DEICTIC = {
    "how do i solve this",
    "how to solve this",
    "how do i do this",
    "i don't know",
    "i am stuck",
    "stuck",
}
_DIAGRAM_REFERENCE = re.compile(
    r"如图|见图|根据图|图中所示|as\s+shown|shown\s+in\s+the\s+(?:figure|diagram)|"
    r"see\s+the\s+(?:figure|diagram)",
    flags=re.IGNORECASE,
)
_CLEARLY_NON_MATH = re.compile(
    r"写.{0,12}(?:诗|作文|小说)|翻译.{0,12}(?:文章|句子)|天气|菜谱|歌词|"
    r"write.{0,5}(?:poem|story|essay)|translate.{0,8}(?:article|sentence)|weather|recipe",
    flags=re.IGNORECASE,
)


def _normalize(query: str) -> str:
    value = unicodedata.normalize("NFKC", query or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(".?!,;:。！？,；：")


def analyze_completeness(
    query: str,
    language: str,
    *,
    has_image: bool = False,
) -> CompletenessResult:
    normalized = _normalize(query)
    if normalized in (_EN_DEICTIC if language == "en" else _ZH_DEICTIC):
        return CompletenessResult(
            status="missing_conditions",
            missing=["full problem" if language == "en" else "完整题干"],
            follow_up=(
                "Please send the full problem, including all known conditions and what must be found or proved."
                if language == "en"
                else "请发送完整题目，包括全部已知条件以及需要求解或证明的目标。"
            ),
        )
    if not has_image and _DIAGRAM_REFERENCE.search(normalized):
        return CompletenessResult(
            status="requires_image",
            missing=[
                "the diagram or all relations shown in it"
                if language == "en"
                else "图形或图中已知关系"
            ],
            follow_up=(
                "Please upload the diagram or describe every marked length, angle, parallel line, and target from the diagram."
                if language == "en"
                else "请上传图形，或写出图中标注的边长、角度、平行关系以及所求目标。"
            ),
        )
    if _CLEARLY_NON_MATH.search(normalized):
        return CompletenessResult(
            status="out_of_scope",
            missing=["a junior mathematics question" if language == "en" else "初中数学问题"],
            follow_up=(
                "Please ask a junior mathematics question or share a step you want checked."
                if language == "en"
                else "请提交初中数学题目，或发来需要检查的解题步骤。"
            ),
        )
    return CompletenessResult(status="complete")
