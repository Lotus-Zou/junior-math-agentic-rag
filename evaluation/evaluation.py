# -*- coding: utf-8 -*-
"""Independent Eval-Test harness for regression, A/B, and quality gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).with_name("math_benchmark_1000.csv")
REPORT_DIR = Path(__file__).with_name("reports")
LATEST_SUMMARY = Path(__file__).with_name("latest_summary.json")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_and_validate_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH).fillna("")
    required = {
        "question_id", "case_type", "grade", "chapter", "knowledge_point", "question",
        "student_wrong_answer", "error_class", "ideal_intent", "expected_context_keywords",
        "reference_answer", "relevant_source",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"评测集缺少字段: {sorted(missing)}")
    if len(frame) != 1000:
        raise ValueError(f"评测集必须为 1000 条，当前为 {len(frame)} 条")
    if frame["question_id"].duplicated().any():
        raise ValueError("question_id 存在重复")
    if set(frame["case_type"]) != {"normal", "hallucination_risk", "colloquial"}:
        raise ValueError("case_type 必须覆盖 normal、hallucination_risk、colloquial")
    return frame


def _keywords(expected: str) -> list[str]:
    return [keyword.strip() for keyword in expected.split("|") if keyword.strip()]


def keyword_recall_at_k(contexts: list[str], expected: str, k: int) -> float:
    keywords = _keywords(expected)
    if not keywords:
        return 1.0
    retrieved = "\n".join(contexts[:k])
    return sum(keyword in retrieved for keyword in keywords) / len(keywords)


def keyword_context_precision(contexts: list[str], expected: str, k: int) -> float:
    selected = contexts[:k]
    if not selected:
        return 0.0
    keywords = _keywords(expected)
    return sum(any(keyword in context for keyword in keywords) for context in selected) / len(selected)


def _selected(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    return frame if limit <= 0 else frame.head(limit)


def run_retrieval(frame: pd.DataFrame, strategy: str, limit: int, k: int, version: str) -> tuple[pd.DataFrame, dict]:
    from agentic_rag.math_retriever import math_retriever

    records = []
    for _, row in _selected(frame, limit).iterrows():
        started = time.time()
        documents, trace = math_retriever.retrieve_candidates([row["question"]], row["chapter"], [row["knowledge_point"]], strategy=strategy)
        contexts = [document.page_content for document in documents[:k]]
        metadata_text = " ".join(str(document.metadata) for document in documents[:k])
        records.append({
            "question_id": row["question_id"], "case_type": row["case_type"], "strategy": strategy,
            "context_precision": keyword_context_precision(contexts, row["expected_context_keywords"], k),
            "context_recall": keyword_recall_at_k(contexts, row["expected_context_keywords"], k),
            "knowledge_point_accuracy": float(row["knowledge_point"] in "\n".join(contexts) or row["knowledge_point"] in metadata_text),
            "latency_ms": round((time.time() - started) * 1000, 2),
            "trace": json.dumps(trace, ensure_ascii=False),
        })
    report = pd.DataFrame(records)
    summary = {
        "version": version, "mode": "retrieval", "strategy": strategy,
        "dataset_size": len(frame), "evaluated": len(report),
        "context_precision": float(report["context_precision"].mean()),
        "context_recall": float(report["context_recall"].mean()),
        "knowledge_point_accuracy": float(report["knowledge_point_accuracy"].mean()),
        "latency_ms": float(report["latency_ms"].mean()),
    }
    return report, summary


def run_ab(frame: pd.DataFrame, limit: int, k: int, version: str) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for strategy in ("dense", "hybrid", "hybrid_graph"):
        report, summary = run_retrieval(frame, strategy, limit, k, version)
        report.to_csv(REPORT_DIR / f"{version}_{strategy}.csv", index=False, encoding="utf-8-sig")
        summaries[strategy] = summary
    baseline = summaries["dense"]["context_recall"]
    summaries["comparison"] = {
        "hybrid_recall_lift": summaries["hybrid"]["context_recall"] - baseline,
        "hybrid_graph_recall_lift": summaries["hybrid_graph"]["context_recall"] - baseline,
    }
    return summaries


def run_e2e(frame: pd.DataFrame, limit: int, k: int, version: str, skip_ragas: bool) -> tuple[pd.DataFrame, dict]:
    from agentic_rag.graph import build_graph
    from agentic_rag.guardrails import guided_answer_violations

    graph = build_graph()
    records, ragas_rows = [], []
    for _, row in _selected(frame, limit).iterrows():
        started = time.time()
        state = graph.invoke({"query": row["question"], "conversation_history": [], "correction_attempts": 0, "validation_issues": []}, config={"recursion_limit": 64})
        contexts = [document.page_content for document in state.get("documents", [])]
        answer = state.get("response", "")
        records.append({
            "question_id": row["question_id"], "case_type": row["case_type"], "trace_id": state.get("trace_id"),
            "context_precision": keyword_context_precision(contexts, row["expected_context_keywords"], k),
            "context_recall": keyword_recall_at_k(contexts, row["expected_context_keywords"], k),
            "knowledge_point_accuracy": float(row["knowledge_point"] in "、".join(state.get("knowledge_points", []))),
            "intent_accuracy": float(state.get("intent") == row["ideal_intent"]),
            "error_diagnosis_accuracy": float(not row["student_wrong_answer"] or any(term in answer for term in row["error_class"].replace("或", "、").split("、"))),
            "step_correctness": float(bool(state.get("validation_passed"))),
            "direct_answer_violation": float(any("直接输出标准答案" in item for item in guided_answer_violations(answer))),
            "hallucination_detected": float(bool(state.get("critic_report", {}).get("hallucination_detected"))),
            "latency_ms": round((time.time() - started) * 1000, 2), "answer": answer,
        })
        ragas_rows.append({"question": row["question"], "answer": answer, "contexts": contexts, "ground_truth": row["reference_answer"]})
    report = pd.DataFrame(records)
    summary = {
        "version": version, "mode": "e2e", "dataset_size": len(frame), "evaluated": len(report),
        **{column: float(report[column].mean()) for column in (
            "context_precision", "context_recall", "knowledge_point_accuracy", "intent_accuracy",
            "error_diagnosis_accuracy", "step_correctness", "direct_answer_violation", "hallucination_detected", "latency_ms",
        )},
    }
    summary["direct_answer_violation_rate"] = summary.pop("direct_answer_violation")
    if not skip_ragas and ragas_rows:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from agentic_rag.chains import get_embedding_function, llm
        result = evaluate(Dataset.from_list(ragas_rows), metrics=[context_precision, context_recall, faithfulness, answer_relevancy], llm=llm, embeddings=get_embedding_function())
        ragas_frame = result.to_pandas()
        for source, target in (("context_precision", "context_precision"), ("context_recall", "context_recall"), ("faithfulness", "faithfulness"), ("answer_relevancy", "answer_relevance")):
            if source in ragas_frame:
                summary[target] = float(ragas_frame[source].mean())
                report[source] = ragas_frame[source]
    return report, summary


def quality_gate(summary: dict) -> tuple[bool, list[str]]:
    from config import QUALITY_THRESHOLDS
    failures = []
    for metric, threshold in QUALITY_THRESHOLDS.items():
        if metric not in summary:
            failures.append(f"缺少门禁指标 {metric}")
        elif metric == "direct_answer_violation_rate" and summary[metric] > threshold:
            failures.append(f"{metric}={summary[metric]:.4f} > {threshold:.4f}")
        elif metric != "direct_answer_violation_rate" and summary[metric] < threshold:
            failures.append(f"{metric}={summary[metric]:.4f} < {threshold:.4f}")
    return not failures, failures


def main():
    parser = argparse.ArgumentParser(description="初中数学 Eval-Test Harness")
    parser.add_argument("--mode", choices=("validate", "retrieval", "ab", "e2e", "gate"), default="validate")
    parser.add_argument("--strategy", choices=("dense", "hybrid", "hybrid_graph"), default="hybrid_graph")
    parser.add_argument("--limit", type=int, default=100, help="0 表示全部 1000 条")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()
    frame = load_and_validate_dataset()
    print(f"数据集校验通过：{len(frame)} 条；case={frame['case_type'].value_counts().to_dict()}")
    if args.mode == "validate":
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "ab":
        summary = run_ab(frame, args.limit, args.k, args.version)
    elif args.mode == "gate":
        summary = json.loads(LATEST_SUMMARY.read_text(encoding="utf-8"))
        passed, failures = quality_gate(summary)
        print(json.dumps({"passed": passed, "failures": failures}, ensure_ascii=False, indent=2))
        if not passed:
            raise SystemExit(2)
        return
    elif args.mode == "retrieval":
        report, summary = run_retrieval(frame, args.strategy, args.limit, args.k, args.version)
        report.to_csv(REPORT_DIR / f"{args.version}_{args.strategy}.csv", index=False, encoding="utf-8-sig")
    else:
        report, summary = run_e2e(frame, args.limit, args.k, args.version, args.skip_ragas)
        report.to_csv(REPORT_DIR / f"{args.version}_e2e.csv", index=False, encoding="utf-8-sig")
    LATEST_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()