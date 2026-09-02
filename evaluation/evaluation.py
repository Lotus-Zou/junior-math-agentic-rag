# -*- coding: utf-8 -*-
"""Independent Eval-Test harness for regression, A/B, and quality gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import sys
import time
import types
import uuid
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).with_name("math_benchmark_1000.csv")
RAGAS_DATASET_PATH = Path(__file__).with_name("ragas_math_knowledge.csv")
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


def load_ragas_dataset() -> pd.DataFrame:
    frame = pd.read_csv(RAGAS_DATASET_PATH).fillna("")
    required = {
        "question_id", "case_type", "grade", "chapter", "knowledge_point",
        "question", "student_wrong_answer", "error_class", "ideal_intent",
        "expected_context_keywords", "reference_answer", "relevant_source",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RAGAS 评测集缺少字段: {sorted(missing)}")
    if frame.empty or frame["question_id"].duplicated().any():
        raise ValueError("RAGAS 评测集必须非空且 question_id 唯一")
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


def _selected(frame: pd.DataFrame, limit: int, case_id: str = "") -> pd.DataFrame:
    selected = frame
    if case_id:
        selected = frame[frame["question_id"] == case_id]
        if selected.empty:
            raise ValueError(f"评测集中不存在 case: {case_id}")
    return selected if limit <= 0 else selected.head(limit)


def run_retrieval(frame: pd.DataFrame, strategy: str, limit: int, k: int, version: str, case_id: str = "") -> tuple[pd.DataFrame, dict]:
    from agentic_rag.math_retriever import math_retriever

    records = []
    for _, row in _selected(frame, limit, case_id).iterrows():
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


def run_ab(frame: pd.DataFrame, limit: int, k: int, version: str, case_id: str = "") -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for strategy in ("dense", "hybrid", "hybrid_graph"):
        report, summary = run_retrieval(frame, strategy, limit, k, version, case_id)
        report.to_csv(REPORT_DIR / f"{version}_{strategy}.csv", index=False, encoding="utf-8-sig")
        summaries[strategy] = summary
    baseline = summaries["dense"]["context_recall"]
    summaries["comparison"] = {
        "hybrid_recall_lift": summaries["hybrid"]["context_recall"] - baseline,
        "hybrid_graph_recall_lift": summaries["hybrid_graph"]["context_recall"] - baseline,
    }
    return summaries


def run_e2e(frame: pd.DataFrame, limit: int, k: int, version: str, skip_ragas: bool, case_id: str = "") -> tuple[pd.DataFrame, dict]:
    from agentic_rag.graph import build_skill_pipeline_graph
    from agentic_rag.guardrails import guided_answer_violations

    graph = build_skill_pipeline_graph()
    records, ragas_rows = [], []
    for _, row in _selected(frame, limit, case_id).iterrows():
        started = time.time()
        trace_id = str(uuid.uuid4())
        state = graph.invoke(
            {
                "query": row["question"],
                "language": "zh",
                "response_language": "zh",
                "conversation_history": [],
                "conversation_summary": "",
                "trace_id": trace_id,
                "deadline_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=180)
                ).timestamp(),
            },
            config={"recursion_limit": 16},
        )
        pipeline = state.get("skill_pipeline", {})
        rendered = pipeline.get("response_render")
        answer = str(getattr(rendered, "answer", "") or "")
        retrieval = pipeline.get("web_search")
        if not getattr(retrieval, "candidates", None):
            retrieval = pipeline.get("rerank_filter")
        contexts = [
            item.content for item in getattr(retrieval, "candidates", [])[:k]
        ]
        critic = pipeline.get("answer_repair_critic") or pipeline.get("answer_critic")
        critic_details = getattr(critic, "critic", {}) or {}
        knowledge_points = list(getattr(rendered, "knowledge_points", []) or [])
        intent = str(getattr(rendered, "intent", "") or "")
        records.append({
            "question_id": row["question_id"], "case_type": row["case_type"], "trace_id": getattr(rendered, "trace_id", trace_id),
            "expected_intent": row["ideal_intent"], "actual_intent": intent,
            "expected_knowledge_point": row["knowledge_point"],
            "actual_knowledge_points": json.dumps(knowledge_points, ensure_ascii=False),
            "context_precision": keyword_context_precision(contexts, row["expected_context_keywords"], k),
            "context_recall": keyword_recall_at_k(contexts, row["expected_context_keywords"], k),
            "knowledge_point_accuracy": float(row["knowledge_point"] in "、".join(knowledge_points)),
            "intent_accuracy": float(intent == row["ideal_intent"]),
            "error_diagnosis_accuracy": float(not row["student_wrong_answer"] or any(term in answer for term in row["error_class"].replace("或", "、").split("、"))),
            "step_correctness": float(bool(getattr(rendered, "validation_passed", False))),
            "direct_answer_violation": float(any("直接输出标准答案" in item for item in guided_answer_violations(answer))),
            "hallucination_detected": float(bool(critic_details.get("hallucination_detected"))),
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        REPORT_DIR / f"{version}_e2e.raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not skip_ragas and ragas_rows:
        if importlib.util.find_spec("langchain_community.chat_models.vertexai") is None:
            compatibility = types.ModuleType(
                "langchain_community.chat_models.vertexai"
            )
            compatibility.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[compatibility.__name__] = compatibility
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        # RAGAS 0.4 exposes collections as modules; the top-level compatibility
        # exports are initialized Metric instances required by evaluate().
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_huggingface import HuggingFaceEmbeddings
        from agentic_rag.chains import _build_llm
        from config import LLM_MODEL_NAME, MATH_EMBEDDING_MODEL_PATH

        # Responses API reasoning models reject the legacy n/temperature
        # parameters that RAGAS otherwise forwards through LangChain.
        judge_llm = _build_llm(
            LLM_MODEL_NAME,
            timeout=180,
            max_retries=4,
        )
        ragas_llm = LangchainLLMWrapper(
            judge_llm,
            bypass_n=True,
            bypass_temperature=True,
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=MATH_EMBEDDING_MODEL_PATH,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )
        result = evaluate(
            Dataset.from_list(ragas_rows),
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(
                timeout=180,
                max_retries=4,
                max_wait=12,
                max_workers=2,
                log_tenacity=True,
            ),
            batch_size=2,
            raise_exceptions=False,
        )
        ragas_frame = result.to_pandas()
        ragas_frame.to_csv(
            REPORT_DIR / f"{version}_ragas.raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
        required_ragas_columns = {
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        }
        missing_or_invalid = sorted(
            column
            for column in required_ragas_columns
            if column not in ragas_frame
            or ragas_frame[column].isna().any()
        )
        if missing_or_invalid:
            raise RuntimeError(
                "RAGAS Judge 未返回有效指标: "
                + ", ".join(missing_or_invalid)
            )
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
    parser.add_argument("--case-id", default="", help="只运行指定 question_id，例如 RAG-002")
    parser.add_argument(
        "--dataset",
        choices=("domain-1000", "ragas"),
        default="domain-1000",
        help="领域 1000 条回归集，或专门走检索链路的 RAGAS 知识问答集。",
    )
    args = parser.parse_args()
    frame = (
        load_and_validate_dataset()
        if args.dataset == "domain-1000"
        else load_ragas_dataset()
    )
    print(f"数据集校验通过：{len(frame)} 条；case={frame['case_type'].value_counts().to_dict()}")
    if args.mode == "validate":
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "ab":
        summary = run_ab(frame, args.limit, args.k, args.version, args.case_id)
    elif args.mode == "gate":
        summary = json.loads(LATEST_SUMMARY.read_text(encoding="utf-8"))
        passed, failures = quality_gate(summary)
        print(json.dumps({"passed": passed, "failures": failures}, ensure_ascii=False, indent=2))
        if not passed:
            raise SystemExit(2)
        return
    elif args.mode == "retrieval":
        report, summary = run_retrieval(frame, args.strategy, args.limit, args.k, args.version, args.case_id)
        report.to_csv(REPORT_DIR / f"{args.version}_{args.strategy}.csv", index=False, encoding="utf-8-sig")
    else:
        report, summary = run_e2e(frame, args.limit, args.k, args.version, args.skip_ragas, args.case_id)
        report.to_csv(REPORT_DIR / f"{args.version}_e2e.csv", index=False, encoding="utf-8-sig")
    LATEST_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
