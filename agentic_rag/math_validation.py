# -*- coding: utf-8 -*-
"""Deterministic checks used alongside the independent LLM Critic."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


EQUATION_PATTERN = re.compile(r"([0-9xX()+\-*/^\s]+)=([0-9xX()+\-*/^\s]+)")


def _extract_equation_sides(query: str) -> tuple[str, str] | None:
    normalized = query or ""
    for delimiter in (r"\(", r"\)", r"\[", r"\]", "$"):
        normalized = normalized.replace(delimiter, " ")
    normalized = normalized.replace("−", "-").replace("×", "*").replace("÷", "/")
    equation = EQUATION_PATTERN.search(normalized)
    if not equation:
        return None
    return equation.group(1).strip(), equation.group(2).strip()


def deterministic_equation_answer(
    query: str,
    student_answer: str = "",
    document_count: int = 0,
    language: str = "zh",
) -> str:
    """Build a transparent local fallback for a single-variable linear equation."""
    try:
        from sympy import Poly, expand, solve, symbols
        from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
    except ImportError:
        return ""
    equation = _extract_equation_sides(query)
    if not equation:
        return ""
    x = symbols("x")
    transforms = standard_transformations + (implicit_multiplication_application,)
    try:
        left = parse_expr(equation[0].replace("^", "**"), local_dict={"x": x, "X": x}, transformations=transforms)
        right = parse_expr(equation[1].replace("^", "**"), local_dict={"x": x, "X": x}, transformations=transforms)
        expression = expand(left - right)
        if Poly(expression, x).degree() != 1:
            return ""
        solutions = solve(expression, x)
        if len(solutions) != 1:
            return ""
        coefficient = expression.coeff(x)
        constant = expression.subs(x, 0)
        isolated = -constant
        solution = solutions[0]
    except Exception:
        return ""

    def math_text(value) -> str:
        return str(value).replace("**", "^").replace("*", "×")

    citation = " [1]" if document_count else ""
    original = f"{math_text(left)} = {math_text(right)}"
    isolated_step = f"{math_text(coefficient)}x = {math_text(isolated)}"
    solution_step = f"x = {math_text(solution)}"
    substitution = math_text(left.subs(x, solution))
    expected = math_text(right.subs(x, solution))
    has_student_attempt = bool((student_answer or "").strip()) or any(
        marker in (query or "") for marker in ("我写成", "我算成", "错误作答", "incorrect attempt", "I wrote")
    )

    if language == "en":
        mistake = (
            "Your attempted transformation changes the equation. Moving a term is shorthand for applying the same inverse operation to both sides; its sign therefore changes."
            if has_student_attempt else
            "The key is to preserve equality by applying the same operation to both sides."
        )
        return (
            f"Knowledge point\nLinear equations in one variable and the properties of equality.{citation}\n\n"
            f"Error analysis\n{mistake}\n\n"
            f"Plan\nIsolate the term containing x, then divide by its coefficient.{citation}\n\n"
            f"Steps\n1. Start with {original}.\n2. Apply the same inverse operation to both sides: {isolated_step}.\n"
            f"3. Divide both sides by {math_text(coefficient)}: {solution_step}.\n\n"
            f"Check\nSubstitute {solution_step} into the original equation: left side = {substitution}, right side = {expected}. They are equal, so {solution_step}."
        )

    mistake = (
        "你的变形改变了原方程。移项只是“两边同时做相同逆运算”的简写，因此跨过等号后符号要改变。"
        if has_student_attempt else
        "关键是利用等式性质，在等号两边同时进行相同运算。"
    )
    return (
        f"知识点定位\n一元一次方程与等式的基本性质。{citation}\n\n"
        f"错因分析\n{mistake}\n\n"
        f"解题思路\n先把含 x 的项单独留在一边，再除以 x 的系数。{citation}\n\n"
        f"分步过程\n1. 原方程为 {original}。\n2. 等号两边做相同的逆运算，得到 {isolated_step}。\n"
        f"3. 两边同时除以 {math_text(coefficient)}，得到 {solution_step}。\n\n"
        f"自检\n把 {solution_step} 代回原方程：左边 = {substitution}，右边 = {expected}，两边相等，所以 {solution_step}。"
    )


def _equation_check(query: str, answer: str) -> list[str]:
    normalized_query = (query or "").lower()
    if not any(
        marker in normalized_query
        for marker in ("解方程", "方程的解", "solve the equation", "solve equation")
    ):
        return []
    try:
        from sympy import solve, symbols
        from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
    except ImportError:
        return ["SymPy 未安装，跳过代数代入复核"]
    equation = _extract_equation_sides(query)
    roots = re.findall(r"(?<![0-9A-Za-z])[xX](?:_?\d+)?\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", answer or "")
    if not equation or not roots:
        return []
    x = symbols("x")
    transforms = standard_transformations + (implicit_multiplication_application,)
    try:
        left = parse_expr(equation[0].replace("^", "**"), local_dict={"x": x, "X": x}, transformations=transforms)
        right = parse_expr(equation[1].replace("^", "**"), local_dict={"x": x, "X": x}, transformations=transforms)
        expected = solve(left - right, x)
        if not expected:
            return []
        parsed_roots = [parse_expr(root, transformations=transforms) for root in roots]
        missing = [solution for solution in expected if not any(candidate.equals(solution) for candidate in parsed_roots)]
        return [f"回答中没有给出原方程的正确解 x={solution}" for solution in missing]
    except Exception:
        return ["方程格式无法由确定性校验器解析，需 Critic 复核"]


def _strict_threshold_check(query: str, answer: str) -> list[str]:
    """Distinguish reaching a threshold from first exceeding it."""
    normalized_query = (query or "").lower()
    strict_markers = (
        "starts earning money",
        "start earning money",
        "make a profit",
        "starts making money",
        "start making money",
        "开始盈利",
        "开始赚钱",
        "产生利润",
        "超过成本",
    )
    if not any(marker in normalized_query for marker in strict_markers):
        return []

    final_match = re.search(
        r"(?:答案|answer)\s*[：:]?\s*[$¥￥]?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        answer or "",
        flags=re.IGNORECASE,
    )
    if not final_match:
        return []
    try:
        final_value = Decimal(final_match.group(1).replace(",", ""))
    except InvalidOperation:
        return []

    divisions = re.findall(
        r"[$¥￥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:÷|/)\s*"
        r"[$¥￥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*=\s*"
        r"[$¥￥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        answer or "",
    )
    for numerator_text, denominator_text, quotient_text in divisions:
        try:
            numerator = Decimal(numerator_text.replace(",", ""))
            denominator = Decimal(denominator_text.replace(",", ""))
            shown_quotient = Decimal(quotient_text.replace(",", ""))
        except InvalidOperation:
            continue
        if denominator == 0:
            continue
        exact_quotient = numerator / denominator
        if (
            exact_quotient == exact_quotient.to_integral_value()
            and shown_quotient == exact_quotient
            and final_value == exact_quotient
        ):
            return [
                "达到盈亏平衡只表示累计收益等于成本；题目要求开始盈利时，需要进入下一个完整周期"
            ]
    return []


def deterministic_math_checks(query: str, answer: str, document_count: int) -> dict:
    issues = _equation_check(query, answer)
    issues.extend(_strict_threshold_check(query, answer))
    if re.search(r"/[ ]*0(?:\D|$)", answer or ""):
        issues.append("推导中出现除以 0")
    citations = set(re.findall(r"\[(\d+)\]", answer or ""))
    invalid = sorted(int(item) for item in citations if int(item) < 1 or int(item) > document_count)
    if invalid:
        issues.append(f"引用编号不存在: {invalid}")
    if document_count and not citations:
        issues.append("关键步骤没有教材引用")
    return {"passed": not issues, "issues": issues, "citations": sorted(citations)}
