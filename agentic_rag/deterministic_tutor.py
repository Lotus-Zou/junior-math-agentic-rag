# -*- coding: utf-8 -*-
"""Strict local solvers for common junior-high curriculum problem families."""

from __future__ import annotations

import math
import re
import unicodedata
from fractions import Fraction


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("−", "-").replace("×", "*").replace("÷", "/")


def _num(value: str) -> Fraction:
    return Fraction(value.strip())


def _fmt(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _attempt(query: str) -> bool:
    lowered = query.lower()
    return any(item in lowered for item in ("我的答案", "错误作答", "我写成", "第一处错误", "my answer", "i wrote"))


def _is_linear_function_query(query: str) -> bool:
    lowered = query.lower()
    return any(item in lowered for item in ("一次函数", "linear function", "slope", "intercept"))


def _guided(language: str, knowledge: str, diagnosis: str, plan: str, steps: list[str], check: str) -> str:
    if language == "en":
        labels = ("Knowledge point", "Error analysis", "Plan", "Steps", "Check")
    else:
        labels = ("知识点定位", "错因分析", "解题思路", "分步过程", "自检")
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return (
        f"**{labels[0]}**\n{knowledge} [1]\n\n**{labels[1]}**\n{diagnosis}\n\n"
        f"**{labels[2]}**\n{plan} [1]\n\n**{labels[3]}**\n{numbered}\n\n**{labels[4]}**\n{check} [1]"
    )


def _result(answer: str, chapter: str, points: list[str], query: str, intent: str = "solve") -> dict:
    return {"answer": answer, "chapter": chapter, "knowledge_points": points, "intent": "error_analysis" if _attempt(query) else intent}


def _linear_function_points(query: str, language: str) -> dict | None:
    pattern = re.compile(r"(?:经过点|through.*?points?)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*(?:和|及|,?\s*and)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", re.I)
    match = pattern.search(query)
    if not match or not _is_linear_function_query(query):
        return None
    x1, y1, x2, y2 = map(_num, match.groups())
    if x1 == x2:
        return None
    k, b = (y2 - y1) / (x2 - x1), y1 - ((y2 - y1) / (x2 - x1)) * x1
    sign_b = f"+ {_fmt(b)}" if b >= 0 else f"- {_fmt(-b)}"
    equation = f"y = {_fmt(k)}x {sign_b}"
    if language == "en":
        parts = ("Slope and y = kx + b.", "Do not swap the slope and intercept.", "Find k from the two points, then substitute one point to find b.", [f"k = ({_fmt(y2)} - {_fmt(y1)})/({_fmt(x2)} - {_fmt(x1)}) = {_fmt(k)}.", f"b = {_fmt(y1)} - {_fmt(k)}*{_fmt(x1)} = {_fmt(b)}.", f"Thus {equation}."], "Both given points satisfy the equation.")
    else:
        parts = ("一次函数斜率公式与斜截式 y = kx + b。", "第一处常见错误是混淆斜率与截距；斜率必须先用纵坐标变化量除以横坐标变化量。", "先用两点求 k，再代入任一点求 b。", [f"k = ({_fmt(y2)} - {_fmt(y1)})/({_fmt(x2)} - {_fmt(x1)}) = {_fmt(k)}。", f"b = {_fmt(y1)} - {_fmt(k)}×{_fmt(x1)} = {_fmt(b)}。", f"所以解析式为 {equation}。"], "把两个已知点分别代入解析式，等式都成立。")
    return _result(_guided(language, *parts), "函数", ["一次函数", "斜率", "待定系数法"], query)


def _linear_function_formula(query: str, language: str) -> dict | None:
    if not _is_linear_function_query(query):
        return None
    match = re.search(r"y\s*=\s*([+-]?(?:(?:\d+(?:\.\d+)?)|(?:\d+/\d+))?)\s*\*?\s*x\s*([+-]\s*(?:(?:\d+(?:\.\d+)?)|(?:\d+/\d+)))?", query, re.I)
    if not match:
        return None
    raw_k = match.group(1).replace(" ", "")
    k = Fraction(1 if raw_k in ("", "+") else -1 if raw_k == "-" else raw_k)
    b = _num(match.group(2).replace(" ", "")) if match.group(2) else Fraction(0)
    y1, trend_zh, trend_en = k + b, ("下降" if k < 0 else "上升"), ("falls" if k < 0 else "rises")
    if language == "en":
        parts = ("In y = kx + b, k is the slope and b is the y-intercept value.", "Keep the sign of the coefficient when reading the slope.", "Read k and b, plot two exact points, then draw their line.", [f"k = {_fmt(k)} and b = {_fmt(b)}.", f"The y-intercept is (0, {_fmt(b)}).", f"At x = 1, y = {_fmt(y1)}, so another point is (1, {_fmt(y1)}).", f"Draw the line through them; it {trend_en} from left to right."], f"Δy/Δx = ({_fmt(y1)} - {_fmt(b)})/(1 - 0) = {_fmt(k)}.")
    else:
        parts = ("一次函数斜截式 y = kx + b；k 是斜率，b 是 y 轴截距的纵坐标。", "读取斜率时必须保留系数的正负号；截距是 x = 0 时的函数值。", "从解析式读出 k、b，再取两个准确点描点连线。", [f"斜率 k = {_fmt(k)}，截距 b = {_fmt(b)}。", f"令 x = 0，得到 y = {_fmt(b)}，图像经过 (0, {_fmt(b)})。", f"令 x = 1，得到 y = {_fmt(y1)}，再取点 (1, {_fmt(y1)})。", f"标出两点，过两点画直线并向两端延长；图像从左向右{trend_zh}。"], f"两点间斜率为 ({_fmt(y1)} - {_fmt(b)})/(1 - 0) = {_fmt(k)}，与解析式一致。")
    return _result(_guided(language, *parts), "函数", ["一次函数", "斜率与截距", "两点作图法"], query, "knowledge_query")


def _congruence_missing_condition(query: str, language: str) -> dict | None:
    compact = re.sub(r"\s+", "", query).lower()
    has_given_sides = all(
        re.search(pattern, compact, re.I)
        for pattern in (r"ab=de", r"ac=df")
    )
    asks_for_congruence = (
        ("全等" in compact and any(marker in compact for marker in ("还需要", "什么条件", "补充条件")))
        or ("congruent" in compact and any(marker in compact for marker in ("whatothercondition", "whatcondition", "additionalcondition")))
    )
    if not has_given_sides or not asks_for_congruence:
        return None

    if language == "en":
        parts = (
            "Triangle congruence: SAS and SSS.",
            "Two corresponding sides alone do not guarantee congruence; a suitable third corresponding condition is needed.",
            "Use the included angles between the two known side pairs for SAS. Another valid sufficient condition is equality of the third sides.",
            [
                "Add angle A = angle D. These are the included angles between AB, AC and DE, DF.",
                "Then AB = DE, angle A = angle D, and AC = DF, so triangle ABC is congruent to triangle DEF by SAS.",
                "Alternatively, adding BC = EF gives all three corresponding sides equal, so SSS also proves congruence.",
            ],
            "The vertex correspondence is A to D, B to E, and C to F, consistent with both given side equalities.",
        )
    else:
        parts = (
            "全等三角形的边角边（SAS）与边边边（SSS）判定。",
            "只有两组对应边相等还不能保证两个三角形全等，还需补充合适的对应条件。",
            "优先补充两组已知边的夹角相等，构成边角边；也可以补充第三组对应边相等。",
            [
                "补充 ∠A = ∠D。∠A 是 AB 与 AC 的夹角，∠D 是 DE 与 DF 的夹角。",
                "于是 AB = DE、∠A = ∠D、AC = DF，根据边角边（SAS），可得 △ABC ≌ △DEF。",
                "另一种充分条件是补充 BC = EF，此时三组对应边分别相等，可根据边边边（SSS）证明全等。",
            ],
            "对应关系为 A↔D、B↔E、C↔F，与 AB=DE、AC=DF 以及补充条件一致。",
        )
    return _result(
        _guided(language, *parts),
        "几何",
        ["全等三角形", "边角边判定", "边边边判定"],
        query,
        "knowledge_query",
    )


def _inequality(query: str, language: str) -> dict | None:
    match = re.search(r"解不等式\s*([+-]?\d*)\s*x\s*([<>≤≥])\s*([+-]?\d+(?:/\d+)?)", query)
    if not match:
        return None
    raw_a, operator, raw_c = match.groups()
    a = Fraction(-1 if raw_a == "-" else 1 if raw_a in ("", "+") else int(raw_a))
    if not a:
        return None
    opposite = {"<": ">", ">": "<", "≤": "≥", "≥": "≤"}
    result_op, boundary = (opposite[operator] if a < 0 else operator), _num(raw_c) / a
    if language == "en":
        parts = ("Properties of inequalities.", "Dividing by a negative number reverses the inequality sign.", "Divide by the coefficient of x and check its sign.", [f"Divide both sides by {_fmt(a)}.", f"Therefore x {result_op} {_fmt(boundary)}."], "A test value from the solution set satisfies the original inequality.")
    else:
        parts = ("一元一次不等式与不等式的基本性质。", "第一处常见错误是用负数乘除两边时没有改变不等号方向。", "两边除以 x 的系数；系数为负时不等号反向。", [f"两边同时除以 {_fmt(a)}。", f"因为系数{'小于 0，不等号反向' if a < 0 else '大于 0，方向不变'}，得到 x {result_op} {_fmt(boundary)}。"], "从解集中取一个数代回原不等式，可以验证成立。")
    return _result(_guided(language, *parts), "代数", ["一元一次不等式", "不等式性质"], query)


def _difference_squares(query: str, language: str) -> dict | None:
    match = re.search(r"因式分解\s*x\s*\^\s*2\s*-\s*(\d+)", query, re.I)
    if not match:
        return None
    square, root = int(match.group(1)), math.isqrt(int(match.group(1)))
    if root * root != square:
        return None
    factor = f"(x - {root})(x + {root})"
    parts = (("Difference of squares.", "The factors need opposite signs.", "Write both terms as squares and apply the formula.", [f"x^2 - {square} = x^2 - {root}^2.", f"= {factor}."], f"Expanding gives x^2 - {square}." ) if language == "en" else ("平方差公式 a^2 - b^2 = (a - b)(a + b)。", "第一处常见错误是误用完全平方；两个因式应一正一负。", "把常数写成平方，再套用平方差公式。", [f"x^2 - {square} = x^2 - {root}^2。", f"所以结果是 {factor}。"], f"展开 {factor} 得 x^2 - {square}。"))
    return _result(_guided(language, *parts), "代数", ["因式分解", "平方差公式"], query)


def _quadratic(query: str, language: str) -> dict | None:
    compact = query.replace(" ", "").replace("²", "^2")
    match = re.search(r"解方程x\^2([+-]\d+)x([+-]\d+)=0", compact, re.I)
    if not match:
        return None
    linear, constant = map(int, match.groups())
    discriminant = linear * linear - 4 * constant
    root_d = math.isqrt(discriminant) if discriminant >= 0 else -1
    if root_d < 0 or root_d * root_d != discriminant:
        return None
    roots = sorted({Fraction(-linear + root_d, 2), Fraction(-linear - root_d, 2)})
    root_text = (" or " if language == "en" else " 或 ").join(f"x = {_fmt(root)}" for root in roots)
    factors = "".join(f"(x {'-' if root >= 0 else '+'} {_fmt(abs(root))})" for root in roots)
    if len(roots) == 1:
        factors *= 2
    parts = (("Factoring and the zero-product property.", "Do not drop a root.", "Factor, then set each factor to zero.", [f"{factors} = 0.", root_text + "."], "Each listed root makes the original expression zero.") if language == "en" else ("一元二次方程的因式分解法与零积性质。", "第一处常见错误是只保留一个根；每个因式都要分别令其为 0。", "先因式分解，再利用零积性质。", [f"原方程化为 {factors} = 0。", f"所以 {root_text}。"], "把每个根分别代回，原方程左边都等于 0。"))
    return _result(_guided(language, *parts), "代数", ["一元二次方程", "因式分解"], query)


def _pythagorean(query: str, language: str) -> dict | None:
    match = re.search(r"两直角边长为\s*(\d+(?:\.\d+)?)\s*和\s*(\d+(?:\.\d+)?)", query)
    if not match:
        return None
    a, b = map(_num, match.groups())
    c2, root = a * a + b * b, math.isqrt((a * a + b * b).numerator) if (a * a + b * b).denominator == 1 else -1
    if root < 0 or root * root != c2:
        return None
    parts = (("Pythagorean theorem.", "Do not add the legs directly.", "Use c^2 = a^2 + b^2.", [f"c^2 = {_fmt(a)}^2 + {_fmt(b)}^2 = {_fmt(c2)}.", f"c = {root}."], f"{root}^2 equals the sum of the two squared legs.") if language == "en" else ("勾股定理。", "第一处常见错误是直接把两直角边相加，没有使用平方关系。", "设斜边为 c，使用 c^2 = a^2 + b^2。", [f"c^2 = {_fmt(a)}^2 + {_fmt(b)}^2 = {_fmt(c2)}。", f"取正值，c = {root}。"], f"{root}^2 等于两直角边的平方和。"))
    return _result(_guided(language, *parts), "几何", ["勾股定理", "直角三角形"], query)


def _probability(query: str, language: str) -> dict | None:
    match = re.search(r"有\s*(\d+)\s*个红球和\s*(\d+)\s*个白球", query)
    if not match:
        return None
    red, white = map(int, match.groups()); total = red + white; value = Fraction(red, total)
    parts = (("Equally likely probability.", "The denominator is the total number of balls.", "Divide favorable outcomes by all outcomes.", [f"Total = {red} + {white} = {total}.", f"P(red) = {red}/{total} = {_fmt(value)}."], "The result is between 0 and 1.") if language == "en" else ("等可能事件概率。", "第一处常见错误是把白球数当成分母；分母应为球的总数。", "用红球数除以球的总数。", [f"总数为 {red} + {white} = {total}。", f"P(红球) = {red}/{total} = {_fmt(value)}。"], "结果在 0 与 1 之间，范围合理。"))
    return _result(_guided(language, *parts), "统计与概率", ["等可能概率", "样本空间"], query)


def _mean(query: str, language: str) -> dict | None:
    match = re.search(r"求数据\s*([-+]?\d+(?:\.\d+)?(?:\s*[,，、]\s*[-+]?\d+(?:\.\d+)?){1,30})\s*的平均数", query)
    if not match:
        return None
    values = [_num(item) for item in re.split(r"[,，、]", match.group(1))]; total = sum(values, Fraction()); mean = total / len(values)
    parts = (("Arithmetic mean.", "The sum must be divided by the number of values.", "Add all values, then divide by their count.", [f"Sum = {_fmt(total)}; count = {len(values)}.", f"Mean = {_fmt(total)}/{len(values)} = {_fmt(mean)}."], "The mean lies between the minimum and maximum.") if language == "en" else ("算术平均数。", "第一处常见错误是求和后没有除以数据个数。", "先求总和，再除以数据个数。", [f"总和为 {_fmt(total)}，共有 {len(values)} 个数据。", f"平均数 = {_fmt(total)}/{len(values)} = {_fmt(mean)}。"], "平均数位于最小值与最大值之间。"))
    return _result(_guided(language, *parts), "统计与概率", ["平均数", "数据分析"], query)


def _circle(query: str, language: str) -> dict | None:
    match = re.search(r"同弧所对圆周角为\s*(\d+(?:\.\d+)?)\s*度", query)
    if not match:
        return None
    angle = _num(match.group(1)); center = 2 * angle
    parts = (("Inscribed angle theorem.", "Do not reverse the factor of two.", "The central angle is twice the inscribed angle.", [f"2 * {_fmt(angle)}° = {_fmt(center)}°."], "Half the result equals the given inscribed angle.") if language == "en" else ("圆周角定理。", "第一处常见错误是把倍数关系用反。", "同弧所对圆心角等于圆周角的 2 倍。", [f"圆心角 = 2×{_fmt(angle)}° = {_fmt(center)}°。"], "结果的一半等于已知圆周角。"))
    return _result(_guided(language, *parts), "几何", ["圆周角定理", "圆心角"], query)


def _similar(query: str, language: str) -> dict | None:
    match = re.search(r"相似比为\s*1\s*:\s*(\d+(?:\.\d+)?).*?小三角形对应边长\s*(\d+(?:\.\d+)?)", query)
    if not match:
        return None
    ratio, small = map(_num, match.groups()); large = ratio * small
    parts = (("Corresponding sides of similar triangles.", "Do not reverse the correspondence.", "Multiply the small side by the scale factor.", [f"Scale factor = {_fmt(ratio)}.", f"Large side = {_fmt(small)}*{_fmt(ratio)} = {_fmt(large)}."], f"{_fmt(small)}:{_fmt(large)} = 1:{_fmt(ratio)}.") if language == "en" else ("相似三角形对应边成比例。", "第一处常见错误是把大小三角形的对应关系写反。", "用小三角形对应边乘放大倍数。", [f"放大倍数为 {_fmt(ratio)}。", f"大三角形对应边 = {_fmt(small)}×{_fmt(ratio)} = {_fmt(large)}。"], f"{_fmt(small)}:{_fmt(large)} = 1:{_fmt(ratio)}，比例一致。"))
    return _result(_guided(language, *parts), "几何", ["相似三角形", "对应边成比例"], query)


SOLVERS = (
    _linear_function_points,
    _linear_function_formula,
    _congruence_missing_condition,
    _inequality,
    _difference_squares,
    _quadratic,
    _pythagorean,
    _probability,
    _mean,
    _circle,
    _similar,
)


def solve_curriculum_problem(query: str, language: str = "zh") -> dict | None:
    normalized = _norm(query)
    for solver in SOLVERS:
        result = solver(normalized, language)
        if result:
            return result
    return None
