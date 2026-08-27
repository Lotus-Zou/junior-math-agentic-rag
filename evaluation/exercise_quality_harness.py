"""Executable release gate for adaptive exercise validity, secrecy, and diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag.exercises.generator import AdaptiveExerciseGenerator
from agentic_rag.exercises.models import ExerciseRequest
from agentic_rag.exercises.store import ExerciseStore
from agentic_rag.exercises.validation import validate_generated_exercise


CASES_PATH = ROOT / "evaluation" / "exercise_cases.jsonl"
PRIVATE_FIELD_NAMES = ("solution", "answer_signature", "parameters")


def _load_cases() -> list[dict[str, Any]]:
    cases = []
    for line_number, raw_line in enumerate(
        CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        required = {"id", "topic", "grade", "difficulty", "exercise_type"}
        if set(value) != required:
            raise ValueError(f"exercise case line {line_number} has an invalid schema")
        cases.append(value)
    if not cases:
        raise ValueError("exercise quality case registry is empty")
    return cases


def run_exercise_quality(seed_count: int = 100) -> dict[str, Any]:
    if not isinstance(seed_count, int) or isinstance(seed_count, bool) or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    invalid_count = 0
    answer_leak_count = 0
    case_results = []
    covered_topics = set()

    for case in _load_cases():
        generator = AdaptiveExerciseGenerator(max_attempts=100)
        fingerprints = set()
        prompt_texts = set()
        for seed in range(seed_count):
            item = generator.generate(
                ExerciseRequest(
                    topic=case["topic"],
                    grade=case["grade"],
                    difficulty=case["difficulty"],
                    exercise_type=case["exercise_type"],
                    seed=seed,
                )
            )
            validation = validate_generated_exercise(item)
            if not validation.passed:
                invalid_count += 1
            public = ExerciseStore(ttl_seconds=60).start(item, mastery={})
            public_text = public.model_dump_json()
            if any(field in public_text for field in PRIVATE_FIELD_NAMES):
                answer_leak_count += 1
            fingerprints.add(item.fingerprint)
            prompt_texts.add((item.problem, item.hint))
            covered_topics.add(item.topic)
        unique_ratio = len(fingerprints) / seed_count
        prompt_unique_ratio = len(prompt_texts) / seed_count
        case_results.append(
            {
                "id": case["id"],
                "unique_ratio": unique_ratio,
                "prompt_unique_ratio": prompt_unique_ratio,
            }
        )

    minimum_unique_ratio = min(item["unique_ratio"] for item in case_results)
    minimum_prompt_unique_ratio = min(
        item["prompt_unique_ratio"] for item in case_results
    )
    return {
        "passed": (
            invalid_count == 0
            and answer_leak_count == 0
            and minimum_unique_ratio >= 0.90
            and minimum_prompt_unique_ratio >= 0.90
            and covered_topics == {"algebra", "geometry", "linear_function"}
        ),
        "seed_count": seed_count,
        "invalid_count": invalid_count,
        "answer_leak_count": answer_leak_count,
        "unique_ratio": minimum_unique_ratio,
        "prompt_unique_ratio": minimum_prompt_unique_ratio,
        "covered_topics": sorted(covered_topics),
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive exercise quality gate")
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    report = run_exercise_quality(seed_count=args.seed_count)
    if args.report or not report["passed"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
