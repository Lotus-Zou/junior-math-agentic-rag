"""Compare new, matched experiments without manufacturing historical results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("answer_relevance", "faithfulness", "source_recall_at_k")


def fingerprint(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def valid_score(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value) and 0 <= value <= 1)


def export_run(*, frame, report, summary: dict, output: Path, k: int) -> None:
    import config

    examples = frame.to_dict("records")
    indexed = {row["question_id"]: row for row in report.to_dict("records")}
    cases = []
    for example in examples:
        row = indexed[example["question_id"]]
        scores = {"answer_relevance": row.get("answer_relevancy"),
                  "faithfulness": row.get("faithfulness"),
                  "source_recall_at_k": row.get("source_recall_at_k")}
        cases.append({"case_id": example["question_id"],
                      "question_sha256": fingerprint(example),
                      "answer_sha256": fingerprint(row["answer"]),
                      "completed": row.get("response_type") == "verified_answer"
                      and row.get("validation_passed") is True,
                      "scores": {key: float(value) if valid_score(value) else None
                                 for key, value in scores.items()}})
    metadata = {"dataset_sha256": fingerprint(sorted(examples, key=lambda row: row["question_id"])),
                "evaluator_sha256": fingerprint({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                                 for name in ("evaluation/evaluation.py", "evaluation/ragas_runner.py",
                                                              "evaluation/paired_evaluation.py")}),
                "judge_model": config.CRITIC_MODEL_NAME,
                "judge_settings": {"provider": config.MODEL_PROVIDER,
                                   "endpoint_sha256": fingerprint(config.OPENAI_API_BASE),
                                   "wire_api": config.MODEL_WIRE_API,
                                   "reasoning_effort": config.MODEL_REASONING_EFFORT},
                "embedding_model": (config.EMBEDDING_MODEL_NAME if config.EMBEDDING_PROVIDER == "openai"
                                    else config.MATH_EMBEDDING_MODEL_PATH),
                "embedding_settings": {"provider": config.EMBEDDING_PROVIDER,
                                       "endpoint_sha256": fingerprint(config.EMBEDDING_API_BASE or config.OPENAI_API_BASE)},
                "metric_definition": {"ragas_version": version("ragas"), "k": k,
                                      "recall": "exact labelled relevant_source basename",
                                      "failed_case_score": 0}}
    result = {"schema_version": 1, "experiment_kind": "new_evaluation_not_historical_evidence",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "metadata": metadata,
              "system": {"version": summary["version"], "model": config.LLM_MODEL_NAME,
                         "prompt_code_sha256": hashlib.sha256((ROOT / "agentic_rag/chains.py").read_bytes()).hexdigest()},
              "cases": cases}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _indexed(run: dict) -> dict:
    if run.get("schema_version") != 1:
        raise ValueError("Unsupported paired experiment schema")
    cases = run.get("cases", [])
    if not cases:
        raise ValueError("Experiments must contain cases")
    result = {}
    for row in cases:
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise ValueError("Duplicate or invalid case ID")
        for name in ("question_sha256", "answer_sha256"):
            value = row.get(name, "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {name}")
        if type(row.get("completed")) is not bool:
            raise ValueError("Completion must be an explicit boolean")
        result[case_id] = row
    return result


def compare_runs(before: dict, after: dict, *, samples: int = 2000, seed: int = 20261020) -> dict:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    required = ("dataset_sha256", "evaluator_sha256", "judge_model", "judge_settings",
                "embedding_model", "embedding_settings", "metric_definition")
    for field in required:
        left = before.get("metadata", {}).get(field)
        if not left or left != after.get("metadata", {}).get(field):
            raise ValueError(f"Mismatched or missing evaluation identity: {field}")
    baseline, candidate = _indexed(before), _indexed(after)
    if set(baseline) != set(candidate):
        raise ValueError("Experiments must contain exactly the same cases")
    ids = sorted(baseline)
    if any(baseline[key]["question_sha256"] != candidate[key]["question_sha256"] for key in ids):
        raise ValueError("Case content changed")
    metrics = {}
    for metric in METRICS:
        paired = []
        for key in ids:
            values = []
            for run in (baseline, candidate):
                row = run[key]
                value = row.get("scores", {}).get(metric)
                if not row["completed"]:
                    values.append(0.0)
                else:
                    values.append(float(value) if valid_score(value) else None)
            if all(value is not None for value in values):
                paired.append(values)
        complete = len(paired) == len(ids)
        item = {"comparable": complete, "paired_valid_cases": len(paired),
                "planned_cases": len(ids), "before": None, "after": None,
                "delta": None, "delta_bootstrap_95_ci": None}
        if complete:
            differences = [right - left for left, right in paired]
            rng = random.Random(seed)
            means = sorted(statistics.fmean(rng.choices(differences, k=len(ids))) for _ in range(samples))
            item.update(before=statistics.fmean(row[0] for row in paired),
                        after=statistics.fmean(row[1] for row in paired),
                        delta=statistics.fmean(differences),
                        delta_bootstrap_95_ci=[means[int((samples - 1) * .025)], means[int((samples - 1) * .975)]])
        metrics[metric] = item
    return {"experiment_kind": "new_paired_comparison", "historical_claim_verified": False,
            "complete": all(row["comparable"] for row in metrics.values()),
            "cases": len(ids), "bootstrap_samples": samples, "seed": seed,
            "failed_cases": {"before": sum(not row["completed"] for row in baseline.values()),
                             "after": sum(not row["completed"] for row in candidate.values())},
            "metrics": metrics,
            "limitations": ["Failed answers remain in the denominator with score zero.",
                            "Missing judge scores prevent a metric comparison.",
                            "Model aliases do not independently establish immutable weight identity.",
                            "These are new measurements, not proof of past school operations or scores."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
