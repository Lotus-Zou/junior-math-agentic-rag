#!/usr/bin/env python3
"""Re-run known development cases, never represent them as unseen acceptance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.public_benchmarks import extract_prediction
from tools.evaluate_competition_target import runtime_identity
from tools.reproduce_experiments import post_question


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "evaluation/reports/reconstructed-20261020-public-tal.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".checkpoint.jsonl")
    if args.output.exists() or checkpoint.exists():
        parser.error("Use a new output path; existing measurements are immutable")
    cases = json.loads(args.baseline.read_text())["results"]
    identity = runtime_identity()
    identity["code_sha256"].update({
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("app.py", "agentic_rag/response_contract.py",
                     "agentic_rag/reliability.py", "tools/compare_development_questions.py")
    })
    rows = []

    def evaluate(item):
        index, case = item
        row = post_question(case["query"], index=1200000 + index)
        row.update(case_id=case["case_id"], expected=case["expected"])
        row["prediction"] = extract_prediction(
            "ceval_middle_school_mathematics", row.get("response", {}).get("answer", ""),
        )
        row["exact_match"] = (row["status"] == 200 and bool(row.get("verified"))
                              and row["prediction"] == row["expected"])
        return row

    def save():
        ordered = sorted(rows, key=lambda row: row["index"])
        successful = sorted(row["latency_ms"] for row in rows if row.get("verified"))
        summary = {
            "evaluated": len(rows), "planned": len(cases), "complete": len(rows) == len(cases),
            "correct": sum(row["exact_match"] for row in rows),
            "accuracy": sum(row["exact_match"] for row in rows) / len(rows),
            "verified_answers": sum(bool(row.get("verified")) for row in rows),
            "clarifications": sum(row.get("response", {}).get("response_type") == "clarification_required" for row in rows),
            "cache_hits": sum(bool(row.get("response", {}).get("cached")) for row in rows),
            "successful_latency_median_ms": statistics.median(successful) if successful else None,
            "baseline_correct": sum(bool(row.get("exact_match")) for row in cases),
            "independent_acceptance": False,
        }
        report = {"generated_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "Known development questions only, all original cases retained, no best-of retries; not an independent accuracy acceptance",
                  "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  "runtime": identity, "summary": summary, "results": ordered}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, checkpoint.open("x") as log:
        for row in pool.map(evaluate, enumerate(cases)):
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            rows.append(row)
            save()
    if runtime_identity() != {key: value for key, value in identity.items() if key != "code_sha256"} | {
        "code_sha256": {key: value for key, value in identity["code_sha256"].items()
                          if key not in {"app.py", "agentic_rag/response_contract.py", "agentic_rag/reliability.py", "tools/compare_development_questions.py"}}
    }:
        raise RuntimeError("Runtime changed during measurement; do not accept this report")


if __name__ == "__main__":
    main()
