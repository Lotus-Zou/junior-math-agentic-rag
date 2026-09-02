"""Checkpointed RAGAS evaluation over production answer records."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, Iterable

import pandas as pd


METRIC_COLUMNS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def _ensure_legacy_vertexai_shim() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        available = importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        available = False
    if not available and module_name not in sys.modules:
        compatibility = types.ModuleType(module_name)
        compatibility.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = compatibility


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["case_id"])] = record
    return records


def _valid_metric_record(record: dict[str, Any]) -> bool:
    for column in METRIC_COLUMNS:
        value = record.get(column)
        if not isinstance(value, (int, float)) or math.isnan(float(value)):
            return False
    return not record.get("error")


def _build_ragas_runtime():
    _ensure_legacy_vertexai_shim()

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from agentic_rag.chains import _build_llm
    from config import LLM_MODEL_NAME, MATH_EMBEDDING_MODEL_PATH

    judge = _build_llm(LLM_MODEL_NAME, timeout=180, max_retries=4)
    llm = LangchainLLMWrapper(
        judge,
        bypass_n=True,
        bypass_temperature=True,
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=MATH_EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return llm, embeddings


def evaluate_ragas_records(
    rows: Iterable[dict[str, Any]],
    *,
    checkpoint_path: Path,
    batch_size: int = 2,
    max_workers: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all rows, resuming only records with four valid RAGAS metrics."""
    _ensure_legacy_vertexai_shim()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    row_list = list(rows)
    existing = _load_checkpoint(checkpoint_path)
    valid_existing = {
        case_id: record
        for case_id, record in existing.items()
        if _valid_metric_record(record)
    }
    pending = [row for row in row_list if str(row["case_id"]) not in valid_existing]
    records_by_id = dict(valid_existing)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    llm, embeddings = _build_ragas_runtime() if pending else (None, None)

    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        dataset_rows = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": list(row.get("contexts", [])),
                "ground_truth": row["ground_truth"],
            }
            for row in batch
        ]
        try:
            result = evaluate(
                Dataset.from_list(dataset_rows),
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(
                    timeout=180,
                    max_retries=4,
                    max_wait=12,
                    max_workers=max(1, max_workers),
                    log_tenacity=True,
                ),
                batch_size=max(1, batch_size),
                raise_exceptions=False,
            ).to_pandas()
            batch_records = []
            for index, row in enumerate(batch):
                scored = result.iloc[index]
                record = {
                    "case_id": str(row["case_id"]),
                    **{
                        column: (
                            float(scored[column])
                            if column in scored and pd.notna(scored[column])
                            else None
                        )
                        for column in METRIC_COLUMNS
                    },
                    "error": "",
                }
                if not _valid_metric_record(record):
                    record["error"] = "invalid_metrics"
                batch_records.append(record)
        except Exception as exc:
            batch_records = [
                {
                    "case_id": str(row["case_id"]),
                    **{column: None for column in METRIC_COLUMNS},
                    "error": type(exc).__name__,
                }
                for row in batch
            ]

        with checkpoint_path.open("a", encoding="utf-8") as stream:
            for record in batch_records:
                records_by_id[record["case_id"]] = record
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"ragas checkpoint: {min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )

    records = [
        records_by_id[str(row["case_id"])]
        for row in row_list
        if str(row["case_id"]) in records_by_id
    ]
    frame = pd.DataFrame(records)
    valid = frame.apply(lambda item: _valid_metric_record(item.to_dict()), axis=1)
    summary: dict[str, Any] = {
        "requested": len(row_list),
        "evaluated": len(frame),
        "valid_cases": int(valid.sum()),
        "invalid_cases": int((~valid).sum()),
        "metric_coverage": float(valid.mean()) if len(frame) else 0.0,
        "complete": bool(len(frame) == len(row_list) and valid.all()),
        "resumed_cases": len(valid_existing),
        "new_cases": len(pending),
        "max_workers": max(1, max_workers),
    }
    for column in METRIC_COLUMNS:
        values = pd.to_numeric(frame.loc[valid, column], errors="coerce")
        summary[column] = float(values.mean()) if len(values) else None
    summary["answer_relevance"] = summary.pop("answer_relevancy")
    return frame, summary
