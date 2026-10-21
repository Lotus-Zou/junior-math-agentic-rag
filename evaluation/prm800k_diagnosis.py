"""Human-labeled first-error diagnosis benchmark using PRM800K phase 2."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from threading import Lock
import time
from typing import Any, Iterable

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.public_benchmarks import VersionRunLock


DEFAULT_DATASET = Path(__file__).with_name("datasets") / "prm800k" / "phase2_test.jsonl"
REPORT_DIR = Path(__file__).with_name("reports") / "prm800k"
PRM800K_REPOSITORY = "tasksource/PRM800K"
PRM800K_REVISION = "547b19506677a59037ee888838834b65e9b1ddd4"
PRM800K_LICENSE = "MIT"
PRM800K_SHA256 = "6B172EFA884AC8341A946DD82E06947C135B7254109FB3F7AA907C715D98AAAD"
_OUT_OF_CURRICULUM = re.compile(
    r"\b(?:derivative|integral|calculus|logarithm|complex number|imaginary|matrix|"
    r"determinant|polynomial|parabola|ellipse|hyperbola|trigonometric|sine|cosine|tangent|"
    r"roots of unity|modulo|congruence class)\b|"
    r"\\(?:int|sum|prod|lim|binom)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DiagnosisCase:
    case_id: str
    source_index: int
    labeler: str
    problem: str
    steps: tuple[str, ...]
    expected_first_error: int
    ratings: tuple[int, ...]

    @property
    def question(self) -> str:
        numbered = "\n".join(
            f"Step {index}. {step}" for index, step in enumerate(self.steps, start=1)
        )
        return (
            "Solve this mathematics problem by auditing the student's proposed solution.\n"
            f"Problem: {self.problem}\n\nStudent solution:\n{numbered}\n\n"
            "Identify the first mathematically incorrect step. Explain the defect briefly, "
            "then write only 'First error: N' on the final line, where N is the step number."
        )

    @property
    def ground_truth(self) -> str:
        step = self.steps[self.expected_first_error - 1]
        return (
            f"The first incorrect step is step {self.expected_first_error}: {step}"
        )


def _first_error(row: dict[str, Any]) -> tuple[int, tuple[int, ...]] | None:
    steps = row.get("label", {}).get("steps", [])
    trajectory = tuple(
        str(item)
        for item in row.get("question", {}).get("pre_generated_steps", [])
    )
    ratings: list[int] = []
    for index, step in enumerate(steps):
        completions = step.get("completions") or []
        if not completions:
            return None
        target = (
            re.sub(r"\s+", " ", trajectory[index]).strip()
            if index < len(trajectory)
            else ""
        )
        matching = [
            completion
            for completion in completions
            if re.sub(r"\s+", " ", str(completion.get("text", ""))).strip()
            == target
        ]
        chosen = step.get("chosen_completion")
        if matching:
            completion = matching[0]
        elif isinstance(chosen, int) and 0 <= chosen < len(completions):
            completion = completions[chosen]
        elif len(completions) == 1:
            completion = completions[0]
        else:
            return None
        rating = completion.get("rating")
        if not isinstance(rating, int) or rating not in {-1, 0, 1}:
            return None
        ratings.append(int(rating))
    negatives = [index for index, rating in enumerate(ratings, start=1) if rating == -1]
    if not negatives:
        return None
    expected = negatives[0]
    return expected, tuple(ratings)


def _problem_key(problem: str) -> str:
    normalized = re.sub(r"\s+", " ", problem).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_cases(
    path: Path,
    *,
    limit: int,
    seed: int,
    curriculum_only: bool = True,
) -> list[DiagnosisCase]:
    eligible: list[DiagnosisCase] = []
    seen_problems: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("is_quality_control_question") or row.get("is_initial_screening_question"):
            continue
        if row.get("label", {}).get("finish_reason") != "found_error":
            continue
        first_error = _first_error(row)
        steps = tuple(str(item) for item in row.get("question", {}).get("pre_generated_steps", []))
        if first_error is None or not steps:
            continue
        expected, ratings = first_error
        if expected > len(steps) or len(ratings) > len(steps):
            continue
        case = DiagnosisCase(
            case_id=f"prm800k-phase2-test-{index}",
            source_index=index,
            labeler=str(row.get("labeler", "")),
            problem=str(row.get("question", {}).get("problem", "")),
            steps=steps,
            expected_first_error=expected,
            ratings=ratings,
        )
        searchable = "\n".join((case.problem, *case.steps))
        if curriculum_only and _OUT_OF_CURRICULUM.search(searchable):
            continue
        if len(case.question) > 7900:
            continue
        problem_key = _problem_key(case.problem)
        if not case.problem.strip() or problem_key in seen_problems:
            continue
        seen_problems.add(problem_key)
        eligible.append(case)
    if limit <= 0 or limit >= len(eligible):
        return eligible
    selected = random.Random(seed).sample(eligible, limit)
    return sorted(selected, key=lambda item: item.source_index)


def extract_first_error(answer: str) -> int | None:
    matches = re.findall(
        r"^\s*(?:\*\*)?first\s+error(?:\*\*)?\s*:\s*(\d+)\s*$",
        str(answer),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if matches:
        return int(matches[-1])
    no_error = re.findall(
        r"^\s*(?:\*\*)?first\s+error(?:\*\*)?\s*:\s*(?:none|no\s+error)\s*$",
        str(answer),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return 0 if no_error else None


def _evaluate_case(case: DiagnosisCase, *, api_url: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    error = ""
    with httpx.Client(timeout=timeout) as client:
        try:
            response = client.post(
                f"{api_url.rstrip('/')}/ask",
                json={"query": case.question, "language": "en"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            error = type(exc).__name__
    answer = str(payload.get("answer", ""))
    predicted = extract_first_error(answer)
    sources = payload.get("sources", []) if isinstance(payload.get("sources"), list) else []
    contexts = [
        str(source.get("excerpt", ""))
        for source in sources
        if isinstance(source, dict) and str(source.get("excerpt", "")).strip()
    ]
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    return {
        **asdict(case),
        "question": case.question,
        "ground_truth": case.ground_truth,
        "predicted_first_error": predicted,
        "prediction_parsed": float(predicted is not None),
        "error_detected": float(predicted is not None and predicted > 0),
        "no_error_prediction": float(predicted == 0),
        "first_error_exact": float(predicted == case.expected_first_error),
        "first_error_within_one": float(
            predicted is not None
            and predicted > 0
            and abs(predicted - case.expected_first_error) <= 1
        ),
        "answer": answer,
        "contexts": contexts,
        "source_count": len(sources),
        "trace_id": str(payload.get("trace_id", "")),
        "response_type": str(payload.get("response_type", "")),
        "validation_passed": bool(payload.get("validation_passed", False)),
        "model_attempts": int(metrics.get("model_attempts") or 0),
        "model_successes": int(metrics.get("model_successes") or 0),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "error": error,
    }


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("answer"):
            predicted = extract_first_error(str(record["answer"]))
            record["predicted_first_error"] = predicted
            record["prediction_parsed"] = float(predicted is not None)
            record["error_detected"] = float(
                predicted is not None and predicted > 0
            )
            record["no_error_prediction"] = float(predicted == 0)
            record["first_error_exact"] = float(
                predicted == int(record["expected_first_error"])
            )
            record["first_error_within_one"] = float(
                predicted is not None
                and predicted > 0
                and abs(predicted - int(record["expected_first_error"])) <= 1
            )
        records[str(record["case_id"])] = record
    return records


def evaluate_cases(
    cases: Iterable[DiagnosisCase],
    *,
    api_url: str,
    timeout: float,
    workers: int,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    case_list = list(cases)
    existing = _load_checkpoint(checkpoint_path)
    valid_existing = {
        case_id: record
        for case_id, record in existing.items()
        if not record.get("error") and record.get("predicted_first_error") is not None
    }
    pending = [case for case in case_list if case.case_id not in valid_existing]
    records_by_id = dict(valid_existing)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = Lock()

    def persist(record: dict[str, Any]) -> None:
        records_by_id[str(record["case_id"])] = record
        with write_lock, checkpoint_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_evaluate_case, case, api_url=api_url, timeout=timeout): case
            for case in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            persist(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(
                    f"prm checkpoint: {len(records_by_id)}/{len(case_list)}",
                    flush=True,
                )

    records = [records_by_id[case.case_id] for case in case_list if case.case_id in records_by_id]
    frame = pd.DataFrame(records)
    latency = frame["latency_ms"].astype(float)
    summary = {
        "dataset": PRM800K_REPOSITORY,
        "split": "phase2_test",
        "revision": PRM800K_REVISION,
        "license": PRM800K_LICENSE,
        "sha256": PRM800K_SHA256,
        "evaluation_scope": "deterministic_human_labeled_subset",
        "requested": len(case_list),
        "evaluated": len(frame),
        "prediction_parse_rate": float(frame["prediction_parsed"].mean()),
        "error_detection_rate": float(frame["error_detected"].mean()),
        "no_error_prediction_rate": float(frame["no_error_prediction"].mean()),
        "first_error_exact_accuracy": float(frame["first_error_exact"].mean()),
        "first_error_within_one_accuracy": float(frame["first_error_within_one"].mean()),
        "empty_prediction_rate": float(frame["predicted_first_error"].isna().mean()),
        "validated_response_rate": float(frame["validation_passed"].mean()),
        "mean_latency_ms": float(latency.mean()),
        "p95_latency_ms": float(latency.quantile(0.95)),
        "resumed_cases": len(valid_existing),
        "new_cases": len(pending),
        "metric_scope": (
            "English competition-mathematics human step labels. This is not a Chinese "
            "junior-high error-category benchmark and is reported separately from RAGAS."
        ),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRM800K human first-error diagnosis evaluation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--ragas-batch-size", type=int, default=2)
    parser.add_argument("--ragas-workers", type=int, default=2)
    parser.add_argument("--include-out-of-domain", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    args = parser.parse_args()

    actual_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest().upper()
    if actual_sha != PRM800K_SHA256:
        raise RuntimeError(f"PRM800K SHA256 mismatch: {actual_sha}")
    cases = load_cases(
        args.dataset,
        limit=args.limit,
        seed=args.seed,
        curriculum_only=not args.include_out_of_domain,
    )
    version_dir = REPORT_DIR / args.version
    version_dir.mkdir(parents=True, exist_ok=True)
    lock_path = version_dir / "run.lock"
    if args.force_unlock:
        lock_path.unlink(missing_ok=True)

    with VersionRunLock(lock_path):
        frame, summary = evaluate_cases(
            cases,
            api_url=args.api_url,
            timeout=args.timeout,
            workers=args.workers,
            checkpoint_path=version_dir / "e2e_checkpoint.jsonl",
        )
        summary["selection_policy"] = (
            "official_phase2_found_error_labels_deduplicated_by_problem"
            if args.include_out_of_domain
            else (
                "official_phase2_found_error_labels_deduplicated_by_problem_"
                "excluding_explicit_out_of_curriculum_topics"
            )
        )
        frame.to_csv(version_dir / "e2e_cases.csv", index=False, encoding="utf-8-sig")
        bad = frame[frame["first_error_exact"].eq(0)]
        bad.to_csv(version_dir / "bad_cases.csv", index=False, encoding="utf-8-sig")

        if args.ragas:
            from evaluation.ragas_runner import evaluate_ragas_records

            ragas_rows = [
                {
                    "case_id": row["case_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "contexts": row["contexts"],
                    "ground_truth": row["ground_truth"],
                }
                for _, row in frame.iterrows()
            ]
            ragas_frame, ragas_summary = evaluate_ragas_records(
                ragas_rows,
                checkpoint_path=version_dir / "ragas_checkpoint.jsonl",
                batch_size=max(1, args.ragas_batch_size),
                max_workers=max(1, args.ragas_workers),
            )
            ragas_frame.to_csv(
                version_dir / "ragas_cases.csv",
                index=False,
                encoding="utf-8-sig",
            )
            summary["ragas"] = ragas_summary

        report_path = version_dir / "summary.json"
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({**summary, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
