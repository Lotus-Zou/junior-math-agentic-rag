from evaluation.exercise_quality_harness import run_exercise_quality


def test_quality_harness():
    report = run_exercise_quality(seed_count=100)

    assert report["invalid_count"] == 0
    assert report["answer_leak_count"] == 0
    assert report["unique_ratio"] >= 0.90, report["cases"]
    assert report["prompt_unique_ratio"] >= 0.90, report["cases"]
    assert report["covered_topics"] == ["algebra", "geometry", "linear_function"]
    assert report["passed"] is True
