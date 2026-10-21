"""Reproducible public mathematics benchmark runner against the production API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
import re
from threading import Lock
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd
from datasets import load_dataset


REPORT_DIR = Path(__file__).with_name("reports") / "public_benchmarks"
CEVAL_DATASET = "ceval/ceval-exam"
CEVAL_CONFIG = "middle_school_mathematics"
CEVAL_SPLIT = "val"
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    benchmark: str
    question: str
    expected: str
    language: str


def ceval_case(row: dict[str, Any]) -> BenchmarkCase:
    options = "\n".join(f"{key}. {row[key]}" for key in ("A", "B", "C", "D"))
    prompt = (
        f"{row['question']}\n{options}\n"
        "请完成推导，并在最后一行严格写“答案：A/B/C/D”中的一个字母。"
    )
    return BenchmarkCase(
        case_id=f"ceval-{row['id']}",
        benchmark="ceval_middle_school_mathematics",
        question=prompt,
        expected=str(row["answer"]).strip().upper(),
        language="zh",
    )


def _gsm8k_expected(answer: str) -> str:
    marker = str(answer).rsplit("####", 1)[-1]
    return normalize_number(marker)


def gsm8k_case(row: dict[str, Any], index: int) -> BenchmarkCase:
    prompt = (
        "Solve this mathematics problem.\n"
        f"{row['question']}\n"
        "Show the calculation, then write only the final numeric value on the last line as 'Answer: value'."
    )
    return BenchmarkCase(
        case_id=f"gsm8k-{index}",
        benchmark="gsm8k",
        question=prompt,
        expected=_gsm8k_expected(row["answer"]),
        language="en",
    )


def load_cases(benchmark: str, limit: int) -> list[BenchmarkCase]:
    if benchmark == "ceval":
        rows = load_dataset(
            CEVAL_DATASET, CEVAL_CONFIG, split=CEVAL_SPLIT
        )
        selected = rows if limit <= 0 else rows.select(range(min(limit, len(rows))))
        return [ceval_case(dict(row)) for row in selected]
    if benchmark == "gsm8k":
        rows = load_dataset(
            GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT
        )
        selected = rows if limit <= 0 else rows.select(range(min(limit, len(rows))))
        return [gsm8k_case(dict(row), index) for index, row in enumerate(selected)]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def normalize_number(value: str) -> str:
    fraction_match = re.findall(
        r"(-?\d+(?:,\d{3})*)\s*/\s*(-?\d+(?:,\d{3})*)",
        str(value),
    )
    if fraction_match:
        numerator, denominator = fraction_match[-1]
        try:
            fraction = Fraction(
                int(numerator.replace(",", "")),
                int(denominator.replace(",", "")),
            )
        except (ValueError, ZeroDivisionError):
            return ""
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    match = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return ""
    raw = match[-1].replace(",", "")
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return raw
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def extract_prediction(benchmark: str, answer: str) -> str:
    if benchmark == "ceval_middle_school_mathematics":
        patterns = (
            r"(?:答案|选择|选项)\s*[：:]?\s*([ABCD])\b",
            r"\b([ABCD])\s*$",
        )
        for pattern in patterns:
            matches = re.findall(pattern, answer.upper(), flags=re.MULTILINE)
            if matches:
                return matches[-1]
        return ""
    final_lines = re.findall(
        r"^\s*(?:\*\*)?(?:answer|答案)(?:\*\*)?\s*[：:]\s*(.*?)\s*$",
        str(answer),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not final_lines:
        return ""
    return normalize_number(final_lines[-1])


def _evaluate_case(
    case: BenchmarkCase,
    *,
    api_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = ""
    payload: dict[str, Any] = {}
    with httpx.Client(timeout=timeout_seconds) as client:
        try:
            response = client.post(
                f"{api_url.rstrip('/')}/ask",
                json={"query": case.question, "language": case.language},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            error = type(exc).__name__
    answer = str(payload.get("answer", ""))
    predicted = extract_prediction(case.benchmark, answer)
    return {
        **asdict(case),
        "predicted": predicted,
        "correct": float(predicted == case.expected),
        "response_type": payload.get("response_type", ""),
        "validation_passed": payload.get("validation_passed", False),
        "model_successes": payload.get("metrics", {}).get("model_successes", 0),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "error": error,
        "answer": answer,
    }


def _checkpoint_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("answer") and record.get("benchmark"):
            predicted = extract_prediction(
                str(record["benchmark"]), str(record["answer"])
            )
            record["predicted"] = predicted
            record["correct"] = float(predicted == str(record.get("expected", "")))
        records[str(record["case_id"])] = record
    return records


class VersionRunLock:
    """Prevent two processes from writing the same version checkpoint."""

    def __init__(self, path: Path):
        self.path = path
        self.file_descriptor: int | None = None

    def __enter__(self) -> "VersionRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file_descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(
                f"评测 version 已在运行: {self.path} ({owner or 'unknown owner'})"
            ) from exc
        os.write(
            self.file_descriptor,
            json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode("utf-8"),
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.file_descriptor is not None:
            os.close(self.file_descriptor)
            self.file_descriptor = None
        self.path.unlink(missing_ok=True)


def evaluate_cases(
    cases: Iterable[BenchmarkCase],
    *,
    api_url: str,
    timeout_seconds: float,
    checkpoint_path: Path | None = None,
    workers: int = 1,
    resume: bool = True,
    retry_invalid: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    case_list = list(cases)
    checkpoint_records = _checkpoint_records(checkpoint_path) if resume else {}
    invalid_existing = (
        {
            case_id: record
            for case_id, record in checkpoint_records.items()
            if record.get("error") or not str(record.get("predicted", "")).strip()
        }
        if retry_invalid
        else {}
    )
    existing = {
        case_id: record
        for case_id, record in checkpoint_records.items()
        if case_id not in invalid_existing
    }
    pending = [case for case in case_list if case.case_id not in existing]
    records_by_id = dict(existing)
    checkpoint_lock = Lock()
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            checkpoint_path.write_text("", encoding="utf-8")

    def persist(record: dict[str, Any]) -> None:
        records_by_id[str(record["case_id"])] = record
        if checkpoint_path is None:
            return
        with checkpoint_lock:
            with checkpoint_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _evaluate_case,
                case,
                api_url=api_url,
                timeout_seconds=timeout_seconds,
            ): case
            for case in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            persist(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(
                    f"checkpoint: {len(records_by_id)}/{len(case_list)}",
                    flush=True,
                )

    records = [
        records_by_id[case.case_id]
        for case in case_list
        if case.case_id in records_by_id
    ]
    frame = pd.DataFrame(records)
    has_prediction = frame["predicted"].fillna("").astype(str).str.strip().ne("")
    latency = frame["latency_ms"].astype(float)
    benchmark_results = {
        str(name): {
            "evaluated": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "errors": int((group["error"] != "").sum()),
            "blank_predictions": int(
                group["predicted"].fillna("").astype(str).str.strip().eq("").sum()
            ),
            "numeric_wrong": int(
                (
                    group["predicted"].fillna("").astype(str).str.strip().ne("")
                    & group["correct"].eq(0)
                ).sum()
            ),
            "conditional_accuracy": float(
                group.loc[
                    group["predicted"].fillna("").astype(str).str.strip().ne(""),
                    "correct",
                ].mean()
            ),
            "mean_latency_ms": float(group["latency_ms"].mean()),
        }
        for name, group in frame.groupby("benchmark", sort=True)
    }
    summary = {
        "benchmarks": sorted(frame["benchmark"].unique().tolist()),
        "evaluated": len(frame),
        "results": benchmark_results,
        "validated_response_rate": (
            float(frame["validation_passed"].mean()) if len(frame) else 0.0
        ),
        "mean_latency_ms": float(frame["latency_ms"].mean()) if len(frame) else 0.0,
        "p50_latency_ms": float(latency.quantile(0.50)) if len(frame) else 0.0,
        "p95_latency_ms": float(latency.quantile(0.95)) if len(frame) else 0.0,
        "p99_latency_ms": float(latency.quantile(0.99)) if len(frame) else 0.0,
        "blank_predictions": int((~has_prediction).sum()) if len(frame) else 0,
        "numeric_wrong": int((has_prediction & frame["correct"].eq(0)).sum()) if len(frame) else 0,
        "conditional_accuracy": (
            float(frame.loc[has_prediction, "correct"].mean())
            if has_prediction.any()
            else 0.0
        ),
        "errors": int((frame["error"] != "").sum()) if len(frame) else 0,
        "resumed_cases": len(existing),
        "retried_cases": len(
            set(invalid_existing).intersection(case.case_id for case in case_list)
        ),
        "new_cases": len(pending),
        "dataset_scope": (
            "Public benchmark score; not a junior-high "
            "mistake-correction or RAG faithfulness score."
        ),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C-Eval/GSM8K public benchmarks against the production API."
    )
    parser.add_argument(
        "--benchmark",
        choices=("ceval", "gsm8k", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=190)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="清空当前 version 的 JSONL checkpoint 并从头运行。",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="只从完整 checkpoint 重建报告，不调用生产 API 或重试空预测。",
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="仅在确认没有同 version 进程后删除遗留锁。",
    )
    args = parser.parse_args()
    if args.report_only and args.no_resume:
        parser.error("--report-only 不能与 --no-resume 同时使用")

    selected = ("ceval", "gsm8k") if args.benchmark == "all" else (args.benchmark,)
    cases = [
        case
        for benchmark in selected
        for case in load_cases(benchmark, args.limit)
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = REPORT_DIR / f"{args.version}_checkpoint.jsonl"
    lock_path = REPORT_DIR / f"{args.version}.lock"
    if args.force_unlock:
        lock_path.unlink(missing_ok=True)
    with VersionRunLock(lock_path):
        frame, summary = evaluate_cases(
            cases,
            api_url=args.api_url,
            timeout_seconds=args.timeout,
            checkpoint_path=checkpoint_path,
            workers=max(1, args.workers),
            resume=not args.no_resume,
            retry_invalid=not args.report_only,
        )
        summary.update(
            {
                "evaluation_scope": (
                    "full_public_labeled_split"
                    if args.limit <= 0
                    else "sampled_smoke"
                ),
                "splits": {
                    "ceval_middle_school_mathematics": "val",
                    "gsm8k": "test",
                },
                "checkpoint": str(checkpoint_path),
                "report_only": args.report_only,
            }
        )
        frame.to_csv(
            REPORT_DIR / f"{args.version}_cases.csv",
            index=False,
            encoding="utf-8-sig",
        )
        report_path = REPORT_DIR / f"{args.version}_summary.json"
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({**summary, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
