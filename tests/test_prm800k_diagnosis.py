import json

from evaluation.prm800k_diagnosis import extract_first_error, load_cases


def _row(*, rating: int = -1, quality_control: bool = False):
    return {
        "labeler": "human-labeler",
        "is_quality_control_question": quality_control,
        "is_initial_screening_question": False,
        "question": {
            "problem": "What is 2 + 2?",
            "pre_generated_steps": ["2 + 2 = 5"],
        },
        "label": {
            "steps": [{"completions": [{"text": "2 + 2 = 5", "rating": rating}]}],
            "finish_reason": "found_error",
        },
    }


def test_load_cases_uses_human_negative_step_and_excludes_quality_control(tmp_path):
    dataset = tmp_path / "phase2_test.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row) for row in (_row(), _row(quality_control=True))),
        encoding="utf-8",
    )

    cases = load_cases(dataset, limit=300, seed=7)

    assert len(cases) == 1
    assert cases[0].expected_first_error == 1
    assert cases[0].ratings == (-1,)
    assert cases[0].labeler == "human-labeler"
    assert "First error: N" in cases[0].question


def test_first_error_parser_requires_explicit_final_line():
    assert extract_first_error("Analysis\nFirst error: 7") == 7
    assert extract_first_error("No consequential defect.\nFirst error: none") == 0
    assert extract_first_error("All steps are valid.\nFirst error: No Error") == 0
    assert extract_first_error("Step 7 appears wrong") is None
    assert extract_first_error("First error: unknown") is None


def test_labels_may_stop_at_first_error_before_generated_solution_ends(tmp_path):
    row = _row()
    row["question"]["pre_generated_steps"].append("A later unreviewed step")
    dataset = tmp_path / "phase2_test.jsonl"
    dataset.write_text(json.dumps(row), encoding="utf-8")

    cases = load_cases(dataset, limit=300, seed=7)

    assert len(cases) == 1
    assert len(cases[0].steps) == 2
    assert cases[0].ratings == (-1,)


def test_zero_rating_before_first_negative_is_allowed_by_official_semantics(tmp_path):
    row = _row()
    row["question"]["pre_generated_steps"] = ["An unverified step", "2 + 2 = 5"]
    row["label"]["steps"] = [
        {"completions": [{"text": "An unverified step", "rating": 0}]},
        {"completions": [{"text": "2 + 2 = 5", "rating": -1}]},
    ]
    dataset = tmp_path / "phase2_test.jsonl"
    dataset.write_text(json.dumps(row), encoding="utf-8")

    cases = load_cases(dataset, limit=300, seed=7)

    assert len(cases) == 1
    assert cases[0].expected_first_error == 2
    assert cases[0].ratings == (0, -1)


def test_rating_is_selected_from_completion_matching_original_trajectory(tmp_path):
    row = _row()
    row["label"]["steps"][0] = {
        "completions": [
            {"text": "A different candidate step", "rating": 1},
            {"text": "2 + 2 = 5", "rating": -1},
        ],
        "chosen_completion": None,
    }
    dataset = tmp_path / "phase2_test.jsonl"
    dataset.write_text(json.dumps(row), encoding="utf-8")

    cases = load_cases(dataset, limit=300, seed=7)

    assert len(cases) == 1
    assert cases[0].expected_first_error == 1


def test_repeated_problem_is_counted_once(tmp_path):
    first = _row()
    second = _row()
    second["labeler"] = "another-human-labeler"
    dataset = tmp_path / "phase2_test.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row) for row in (first, second)),
        encoding="utf-8",
    )

    cases = load_cases(dataset, limit=300, seed=7)

    assert len(cases) == 1
    assert cases[0].labeler == "human-labeler"
