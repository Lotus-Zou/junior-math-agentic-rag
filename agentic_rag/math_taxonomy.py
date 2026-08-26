# -*- coding: utf-8 -*-
"""Math taxonomy, formula-aware chunk sizing, and retrieval tokenization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

CHAPTER_KEYWORDS = {
    "函数": ("函数", "坐标", "图像", "正比例", "反比例", "一次函数", "二次函数", "自变量"),
    "几何": ("三角形", "四边形", "圆", "角", "平行", "垂直", "全等", "相似", "勾股", "面积", "体积", "证明"),
    "代数": ("方程", "不等式", "因式分解", "整式", "分式", "根式", "实数", "有理数", "未知数", "解集"),
    "统计与概率": ("概率", "统计", "平均数", "中位数", "众数", "方差", "频率", "样本"),
}
GRADE_KEYWORDS = {
    "七年级": ("有理数", "整式", "一元一次方程", "相交线", "平行线", "统计调查"),
    "八年级": ("全等三角形", "轴对称", "勾股", "一次函数", "因式分解", "分式"),
    "九年级": ("一元二次方程", "二次函数", "旋转", "圆", "相似", "锐角三角函数", "概率"),
}
PREREQUISITES = {
    "一元一次方程": ("有理数运算", "等式性质", "整式运算"),
    "一元二次方程": ("一元一次方程", "因式分解", "平方根"),
    "一次函数": ("平面直角坐标系", "一元一次方程"),
    "二次函数": ("一元二次方程", "一次函数", "配方法"),
    "反比例函数": ("平面直角坐标系", "分式"),
    "全等三角形": ("三角形", "平行线与角"),
    "相似三角形": ("全等三角形", "比例线段"),
    "勾股定理": ("平方根", "直角三角形"),
    "圆周角": ("圆", "角", "三角形"),
    "等可能概率": ("统计", "分数运算"),
}
FORMULA_PATTERN = re.compile(
    r"\$[^$\n]+\$|\\\([^\n]+?\\\)|\\\[[\s\S]+?\\\]|"
    r"(?:[A-Za-z0-9_{}()]+\s*){1,4}[=<>≤≥±√^/](?:\s*[A-Za-z0-9_{}().+\-*/^]+){1,6}"
)


@dataclass(frozen=True)
class MathClassification:
    chapter: str
    grade: str
    knowledge_points: List[str]
    question_type: str


def _matches(text: str, keywords: Iterable[str]) -> List[str]:
    return [keyword for keyword in keywords if keyword in text]


def classify_math_text(text: str) -> MathClassification:
    normalized = re.sub(r"\s+", "", text or "")
    explicit_chapter = next(
        (chapter for chapter in CHAPTER_KEYWORDS if re.search(rf"(?:^|\n)#{{1,3}}\s*{chapter}", text or "")),
        None,
    )
    chapter_scores = {chapter: _matches(normalized, words) for chapter, words in CHAPTER_KEYWORDS.items()}
    chapter = explicit_chapter or max(chapter_scores, key=lambda item: len(chapter_scores[item]))
    if not explicit_chapter and not chapter_scores[chapter]:
        chapter = "综合"
    grade_scores = {grade: _matches(normalized, words) for grade, words in GRADE_KEYWORDS.items()}
    grade = max(grade_scores, key=lambda item: len(grade_scores[item]))
    if not grade_scores[grade]:
        grade = "初中"
    knowledge_points = chapter_scores.get(chapter, [])[:4] or [chapter]
    if any(word in normalized for word in ("证明", "求证")):
        question_type = "证明题"
    elif any(word in normalized for word in ("图像", "作图", "画出")):
        question_type = "图像题"
    elif any(word in normalized for word in ("为什么", "概念", "定义")):
        question_type = "概念题"
    elif any(word in normalized for word in ("应用题", "路程", "工程", "利润")):
        question_type = "应用题"
    else:
        question_type = "计算题"
    return MathClassification(chapter, grade, knowledge_points, question_type)


def extract_formulas(text: str) -> List[str]:
    """Extract LaTeX and common inline mathematical expressions without splitting them."""
    formulas = []
    for match in FORMULA_PATTERN.finditer(text or ""):
        formula = match.group(0).strip()
        if formula and formula not in formulas:
            formulas.append(formula)
    return formulas


def formula_density(text: str) -> float:
    non_space = max(1, len(re.sub(r"\s+", "", text or "")))
    return sum(len(formula) for formula in extract_formulas(text)) / non_space


def adaptive_chunk_size(text: str) -> int:
    """Use 200-400 tokens for formula-heavy blocks and 600-800 for concepts."""
    formulas = extract_formulas(text)
    if len(formulas) >= 2:
        return 280
    if formulas:
        return 380
    return 700


def infer_prerequisites(knowledge_points: Iterable[str]) -> List[str]:
    prerequisites = []
    for point in knowledge_points:
        for known_point, required in PREREQUISITES.items():
            if known_point in point or point in known_point:
                prerequisites.extend(required)
    return list(dict.fromkeys(prerequisites))


def tokenize_math(text: str) -> List[str]:
    """Tokenize LaTeX commands, symbols, identifiers, numbers, and Chinese n-grams for BM25."""
    normalized = (text or "").lower()
    tokens = re.findall(r"\\[a-zA-Z]+|[A-Za-z]+|\d+(?:\.\d+)?|[=<>≤≥+\-*/^(){}\[\]√±]", normalized)
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        tokens.append(run)
        for size in (2, 3, 4):
            tokens.extend(run[index:index + size] for index in range(max(0, len(run) - size + 1)))
    return tokens