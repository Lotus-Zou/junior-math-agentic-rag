import pytest

from agentic_rag.local_intents import parse_local_command


@pytest.mark.parametrize(
    ("query", "action", "topic", "delta"),
    [
        ("\u51e0\u4f55", "practice", "geometry", 0),
        ("\u4ee3\u6570", "practice", "algebra", 0),
        ("\u4e00\u6b21\u51fd\u6570", "practice", "linear_function", 0),
        ("\u518d\u6765\u4e00\u9053", "next_exercise", None, 0),
        ("\u96be\u4e00\u70b9", "adjust_difficulty", None, 1),
        ("\u7b80\u5355\u4e00\u70b9", "adjust_difficulty", None, -1),
        ("\u6362\u4e2a\u95ee\u9898", "new_question", None, 0),
    ],
)
def test_short_commands(query, action, topic, delta):
    command = parse_local_command(query, "zh")

    assert (command.action, command.topic, command.difficulty_delta) == (action, topic, delta)


@pytest.mark.parametrize(
    ("query", "action", "topic", "delta"),
    [
        ("geometry", "practice", "geometry", 0),
        ("algebra", "practice", "algebra", 0),
        ("linear function", "practice", "linear_function", 0),
        ("another exercise", "next_exercise", None, 0),
        ("harder", "adjust_difficulty", None, 1),
        ("easier", "adjust_difficulty", None, -1),
        ("new question", "new_question", None, 0),
    ],
)
def test_english_short_commands(query, action, topic, delta):
    command = parse_local_command(query, "en")

    assert (command.action, command.topic, command.difficulty_delta) == (action, topic, delta)


def test_complete_problem_is_not_a_topic_command():
    assert parse_local_command("\u5728\u4e09\u89d2\u5f62ABC\u4e2d\uff0cA=40\u5ea6\uff0c\u6c42B\u548cC", "zh") is None


def test_parser_normalizes_width_whitespace_and_trailing_punctuation():
    command = parse_local_command("  \uff27\uff45\uff4f\uff4d\uff45\uff54\uff52\uff59\uff1f\uff01  ", "en")

    assert (command.action, command.topic, command.difficulty_delta) == ("practice", "geometry", 0)