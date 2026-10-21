# -*- coding: utf-8 -*-
"""Generate a deterministic 1,000-case Eval dataset with risk strata."""

import csv
from pathlib import Path

OUTPUT_PATH = Path(__file__).with_name("math_benchmark_1000.csv")
CASE_TYPES = ("normal", "hallucination_risk", "colloquial")
SOURCE_BY_POINT = {
    "一元一次方程": "data/sources/grade7-linear-equations.md",
    "一元一次不等式": "data/sources/grade7-inequalities.md",
    "因式分解": "data/sources/grade8-factorization.md",
    "一次函数": "data/sources/grade8-linear-functions.md",
    "一元二次方程": "data/sources/grade9-quadratics.md",
    "勾股定理": "data/sources/grade8-pythagorean.md",
    "等可能概率": "data/sources/grade9-probability.md",
    "平均数": "data/sources/grade8-statistics.md",
    "圆周角": "data/sources/grade9-circle-angles.md",
    "相似三角形": "data/sources/grade9-similarity.md",
}


def build_rows():
    raw = []
    triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
    for index in range(100):
        a, b, m = index % 8 + 2, index + 1, index % 11 + 1
        raw.append(("七年级", "代数", "一元一次方程", f"解方程 {a}(x-{b})={a*m}。", "方程|移项|检验|代入", f"x={b+m}", f"x={b+m+1}", "移项或计算错误"))
        n = index + 2
        raw.append(("七年级", "代数", "一元一次不等式", f"解不等式 -{a}x<{a*n}。", "不等式|负数|方向|解集", f"x>-{n}", f"x<-{n}", "负数乘除未改变方向"))
        n = index + 2
        raw.append(("八年级", "代数", "因式分解", f"因式分解 x^2-{n*n}。", "因式分解|平方差|完全分解", f"(x-{n})(x+{n})", f"(x-{n})^2", "公式条件误用"))
        k, intercept = index % 7 + 1, index - 50
        wrong_function = f"y={intercept}x+{k}" if k != intercept else f"y={k+1}x+{intercept}"
        raw.append(("八年级", "函数", "一次函数", f"一次函数经过点 (0,{intercept}) 和 (1,{intercept+k})，求解析式。", "一次函数|斜率|代入|截距", f"y={k}x+{intercept}", wrong_function, "斜率与截距混淆"))
        p, q = index + 1, index + 2
        raw.append(("九年级", "代数", "一元二次方程", f"解方程 x^2-{p+q}x+{p*q}=0。", "一元二次方程|因式分解|检验|根", f"x={p} 或 x={q}", f"x={p+q}", "漏根"))
        leg_a, leg_b, hypotenuse = triples[index % len(triples)]
        scale = index // len(triples) + 1
        leg_a, leg_b, hypotenuse = leg_a * scale, leg_b * scale, hypotenuse * scale
        raw.append(("八年级", "几何", "勾股定理", f"直角三角形两直角边长为 {leg_a} 和 {leg_b}，求斜边。", "勾股定理|斜边|平方|直角", str(hypotenuse), str(leg_a + leg_b), "勾股关系误用"))
        red, total = index + 1, index * 2 + 7
        raw.append(("九年级", "统计与概率", "等可能概率", f"袋中有 {red} 个红球和 {total-red} 个白球，随机取一球，求红球概率。", "概率|等可能|结果数|样本空间", f"{red}/{total}", f"{red}/{total-red}", "样本空间错误"))
        start = index + 1
        values = [start + offset for offset in range(5)]
        raw.append(("八年级", "统计与概率", "平均数", f"求数据 {','.join(map(str, values))} 的平均数。", "平均数|数据|总数|算术平均", str(start+2), str(sum(values)), "未除以数据个数"))
        angle = (index + 2) / 2
        raw.append(("九年级", "几何", "圆周角", f"同弧所对圆周角为 {angle:g} 度，求圆心角。", "圆周角|圆心角|同弧|二倍", f"{angle*2:g} 度", f"{angle/2:g} 度", "圆周角定理方向错误"))
        side, ratio = index + 2, index % 4 + 2
        raw.append(("九年级", "几何", "相似三角形", f"相似三角形相似比为 1:{ratio}，小三角形对应边长 {side}，求大三角形对应边。", "相似三角形|相似比|对应边|比例", str(side*ratio), str(side/ratio), "对应关系错误"))

    rows = []
    for item_id, item in enumerate(raw, start=1):
        grade, chapter, point, formal_question, keywords, answer, wrong_answer, error_class = item
        case_type = CASE_TYPES[(item_id - 1) % len(CASE_TYPES)]
        if case_type == "colloquial":
            question = f"这题我卡住了，能带我一步步做吗？原题：{formal_question}"
            student_answer, intent = "", "solve"
        elif case_type == "hallucination_risk":
            question = f"原题：{formal_question} 我的答案是 {wrong_answer}，请找出第一处错误。"
            student_answer, intent = wrong_answer, "error_analysis"
        else:
            question, student_answer, intent = formal_question, "", "solve"
        rows.append({
            "question_id": f"MATH-{item_id:04d}", "case_type": case_type,
            "grade": grade, "chapter": chapter, "knowledge_point": point,
            "question": question, "student_wrong_answer": student_answer,
            "problem_text": formal_question,
            "error_class": error_class, "ideal_intent": intent,
            "expected_context_keywords": keywords, "reference_answer": answer,
            "relevant_source": SOURCE_BY_POINT[point],
        })
    return rows


def main():
    rows = build_rows()
    if len(rows) != 1000:
        raise AssertionError(f"expected 1000 rows, got {len(rows)}")
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {len(rows)} rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
