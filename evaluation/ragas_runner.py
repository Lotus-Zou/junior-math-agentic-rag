"""Checkpointed RAGAS evaluation over production answer records."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
import hashlib
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


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _evaluation_identity() -> str:
    import config

    return _fingerprint({
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ragas_version": version("ragas"), "metrics": METRIC_COLUMNS,
        "judge": config.CRITIC_MODEL_NAME, "provider": config.MODEL_PROVIDER,
        "endpoint": config.OPENAI_API_BASE, "wire_api": config.MODEL_WIRE_API,
        "reasoning_effort": config.MODEL_REASONING_EFFORT,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_model": config.EMBEDDING_MODEL_NAME if config.EMBEDDING_PROVIDER == "openai"
        else config.MATH_EMBEDDING_MODEL_PATH,
        "embedding_endpoint": config.EMBEDDING_API_BASE or config.OPENAI_API_BASE,
    })


def _input_identity(row: dict[str, Any]) -> str:
    return _fingerprint({key: row.get(key, [] if key == "contexts" else "")
                         for key in ("question", "answer", "contexts", "ground_truth")})


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
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not 0 <= value <= 1):
            return False
    return not record.get("error")


def _build_ragas_runtime():
    _ensure_legacy_vertexai_shim()

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from agentic_rag.chains import _build_llm
    from config import (
        CRITIC_MODEL_NAME, MATH_EMBEDDING_MODEL_PATH, EMBEDDING_PROVIDER,
        EMBEDDING_API_BASE, EMBEDDING_MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY,
    )

    judge = _build_llm(CRITIC_MODEL_NAME, timeout=180, max_retries=4)
    llm = LangchainLLMWrapper(
        judge,
        bypass_n=True,
        bypass_temperature=True,
    )
    if EMBEDDING_PROVIDER == "openai":
        from langchain_core.embeddings import Embeddings
        from openai import OpenAI

        class CompatibleOpenAIEmbeddings(Embeddings):
            def __init__(self):
                self.client = OpenAI(api_key=OPENAI_API_KEY or "local-development-key",
                                     base_url=EMBEDDING_API_BASE or OPENAI_API_BASE)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                response = self.client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
                return [item.embedding for item in response.data]

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

        embedding_model = CompatibleOpenAIEmbeddings()
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=MATH_EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    embeddings = LangchainEmbeddingsWrapper(embedding_model)
    return llm, embeddings


def evaluate_ragas_records(
    rows: Iterable[dict[str, Any]],
    *,
    checkpoint_path: Path,
    batch_size: int = 2,
    max_workers: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resume valid metrics only for identical inputs and scoring configuration."""
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
    inputs = {str(row["case_id"]): _input_identity(row) for row in row_list}
    if len(inputs) != len(row_list):
        raise ValueError("Duplicate case IDs cannot be scored")
    evaluation_identity = _evaluation_identity()
    existing = _load_checkpoint(checkpoint_path)
    valid_existing = {
        case_id: record
        for case_id, record in existing.items()
        if _valid_metric_record(record) and case_id in inputs
        and record.get("input_sha256") == inputs[case_id]
        and record.get("evaluation_sha256") == evaluation_identity
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
                record.update(input_sha256=inputs[record["case_id"]],
                              evaluation_sha256=evaluation_identity)
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
    frame = pd.DataFrame(records, columns=["case_id", *METRIC_COLUMNS, "error",
                                          "input_sha256", "evaluation_sha256"])
    valid = frame.apply(lambda item: _valid_metric_record(item.to_dict()), axis=1).astype(bool)
    summary: dict[str, Any] = {
        "requested": len(row_list),
        "evaluated": len(frame),
        "valid_cases": int(valid.sum()),
        "invalid_cases": int((~valid).sum()),
        "metric_coverage": float(valid.mean()) if len(frame) else 0.0,
        "complete": bool(row_list and len(frame) == len(row_list) and valid.all()),
        "resumed_cases": len(valid_existing),
        "new_cases": len(pending),
        "max_workers": max(1, max_workers),
        "evaluation_sha256": evaluation_identity,
    }
    for column in METRIC_COLUMNS:
        values = pd.to_numeric(frame.loc[valid, column], errors="coerce")
        summary[column] = float(values.mean()) if len(values) else None
    summary["answer_relevance"] = summary.pop("answer_relevancy")
    return frame, summary
