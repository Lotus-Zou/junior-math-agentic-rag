import pytest

from agentic_rag.completeness import analyze_completeness


@pytest.mark.parametrize("query", ["这题怎么做", "不会", "怎么解"])
def test_deictic_request_needs_full_problem(query):
    result = analyze_completeness(query, "zh")

    assert result.status == "missing_conditions"
    assert result.missing == ["完整题干"]
    assert "完整题目" in result.follow_up


def test_missing_diagram_relations_are_named():
    result = analyze_completeness("如图，在△ABC中求∠A", "zh", has_image=False)

    assert result.status == "requires_image"
    assert result.missing == ["图形或图中已知关系"]
    assert "图中" in result.follow_up


def test_supplied_diagram_allows_problem_to_continue():
    result = analyze_completeness("如图，在△ABC中求∠A", "zh", has_image=True)

    assert result.status == "complete"


@pytest.mark.parametrize(
    "query",
    ["解方程 2x+3=11", "一次函数 y=-2x+3 的斜率是什么"],
)
def test_complete_problem_continues(query):
    assert analyze_completeness(query, "zh").status == "complete"


def test_unknown_full_math_text_is_not_overblocked():
    result = analyze_completeness("请判断命题 x²≥0 是否恒成立并说明理由", "zh")

    assert result.status == "complete"


def test_english_deictic_and_diagram_followups_are_actionable():
    deictic = analyze_completeness("How do I solve this?", "en")
    diagram = analyze_completeness("As shown, find angle A", "en", has_image=False)

    assert deictic.status == "missing_conditions"
    assert "full problem" in deictic.follow_up.lower()
    assert diagram.status == "requires_image"
    assert "diagram" in diagram.follow_up.lower()


def test_non_math_request_is_out_of_scope_only_when_unambiguous():
    result = analyze_completeness("帮我写一首关于春天的诗", "zh")

    assert result.status == "out_of_scope"
    assert result.missing == ["初中数学问题"]
