"""Run BEIR and C-MTEB retrieval tasks with the production embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASKS = ("SciFact", "CovidRetrieval")
REPORT_DIR = Path(__file__).with_name("reports") / "mteb"
RETRIEVAL_METRICS = (
    "ndcg_at_10",
    "map_at_10",
    "recall_at_10",
    "precision_at_10",
)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _inject_local_snapshot(task: Any, data_root: Path) -> bool:
    task_name = task.metadata.name
    task_dir = data_root / task_name
    if not task_dir.is_dir():
        return False

    split = task.metadata.eval_splits[0]
    if task_name == "SciFact":
        corpus_rows = _jsonl_rows(task_dir / "corpus.jsonl")
        query_rows = _jsonl_rows(task_dir / "queries.jsonl")
        qrel_rows = _jsonl_rows(task_dir / "qrels.jsonl")
        corpus = {
            str(row["_id"]): (
                f"{row.get('title', '')} {row.get('text', '')}".strip()
            )
            for row in corpus_rows
        }
        queries = {
            str(row["_id"]): str(row["text"])
            for row in query_rows
        }
        relevant_docs: dict[str, dict[str, int]] = {}
        for row in qrel_rows:
            query_id = str(row.get("query-id", row.get("query_id", "")))
            corpus_id = str(row.get("corpus-id", row.get("corpus_id", "")))
            relevant_docs.setdefault(query_id, {})[corpus_id] = int(row["score"])
    elif task_name == "CovidRetrieval":
        corpus_rows = pd.read_parquet(task_dir / "corpus.parquet").to_dict("records")
        query_rows = pd.read_parquet(task_dir / "queries.parquet").to_dict("records")
        qrel_rows = pd.read_parquet(task_dir / "qrels.parquet").to_dict("records")
        corpus = {str(row["id"]): str(row["text"]) for row in corpus_rows}
        queries = {str(row["id"]): str(row["text"]) for row in query_rows}
        relevant_docs = {}
        for row in qrel_rows:
            relevant_docs.setdefault(str(row["qid"]), {})[str(row["pid"])] = int(
                row["score"]
            )
    else:
        raise ValueError(f"不支持本地快照任务: {task_name}")

    relevant_docs = {
        query_id: documents
        for query_id, documents in relevant_docs.items()
        if query_id in queries and documents
    }
    queries = {
        query_id: queries[query_id]
        for query_id in relevant_docs
    }

    task.corpus = {split: corpus}
    task.queries = {split: queries}
    task.relevant_docs = {split: relevant_docs}
    task.data_loaded = True
    return True


def _snapshot_provenance(task_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        }
        for path in sorted(task_dir.iterdir())
        if path.is_file()
    ]


def _capture_task_size(task: Any) -> dict[str, Any]:
    """Load a retrieval task once and record its complete judged split size."""
    if not getattr(task, "data_loaded", False):
        task.load_data()
    split = task.metadata.eval_splits[0]
    relevant_docs = task.relevant_docs[split]
    return {
        "split": split,
        "corpus": len(task.corpus[split]),
        "judged_queries": len(task.queries[split]),
        "qrels": sum(len(documents) for documents in relevant_docs.values()),
    }


def _sample_retrieval_task(
    task: Any,
    *,
    max_queries: int,
    max_corpus: int,
) -> dict[str, int]:
    split = task.metadata.eval_splits[0]
    queries = task.queries[split]
    corpus = task.corpus[split]
    relevant_docs = task.relevant_docs[split]
    eligible_query_ids = sorted(
        query_id
        for query_id in queries
        if relevant_docs.get(query_id)
    )
    selected_query_ids = eligible_query_ids[:max_queries]
    positive_doc_ids = {
        doc_id
        for query_id in selected_query_ids
        for doc_id in relevant_docs[query_id]
        if doc_id in corpus
    }
    if len(positive_doc_ids) > max_corpus:
        raise ValueError("max_corpus 小于所选查询的正例文档数")
    filler_ids = (
        doc_id
        for doc_id in sorted(corpus)
        if doc_id not in positive_doc_ids
    )
    selected_doc_ids = list(sorted(positive_doc_ids))
    selected_doc_ids.extend(
        list(filler_ids)[: max_corpus - len(selected_doc_ids)]
    )
    selected_doc_id_set = set(selected_doc_ids)
    task.queries[split] = {
        query_id: queries[query_id] for query_id in selected_query_ids
    }
    task.corpus[split] = {
        doc_id: corpus[doc_id] for doc_id in selected_doc_ids
    }
    task.relevant_docs[split] = {
        query_id: {
            doc_id: score
            for doc_id, score in relevant_docs[query_id].items()
            if doc_id in selected_doc_id_set
        }
        for query_id in selected_query_ids
    }
    return {
        "queries": len(task.queries[split]),
        "corpus": len(task.corpus[split]),
        "positive_documents": len(positive_doc_ids),
    }


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize_task_result(result: Any) -> dict[str, Any]:
    """Flatten a MTEB TaskResult without combining unrelated tasks."""
    scores = getattr(result, "scores", {}) or {}
    split_summaries: dict[str, dict[str, float]] = {}
    for split, rows in scores.items():
        split_summary = {}
        for metric in RETRIEVAL_METRICS:
            values = _numeric_values(rows, metric)
            if values:
                split_summary[metric] = mean(values)
        main_scores = _numeric_values(rows, "main_score")
        if main_scores:
            split_summary["main_score"] = mean(main_scores)
        split_summaries[str(split)] = split_summary
    return {
        "task": str(getattr(result, "task_name", "")),
        "dataset_revision": str(getattr(result, "dataset_revision", "")),
        "mteb_version": str(getattr(result, "mteb_version", "")),
        "evaluation_time_seconds": getattr(result, "evaluation_time", None),
        "splits": split_summaries,
    }


def run_mteb(
    *,
    task_names: list[str],
    model_name: str,
    output_dir: Path,
    batch_size: int,
    overwrite: bool,
    data_root: Path | None = None,
    max_seq_length: int = 512,
    max_queries: int = 0,
    max_corpus: int = 0,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict[str, Any]:
    try:
        import mteb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "MTEB 评测依赖未安装。请先使用 requirements-eval.txt 安装评测依赖。"
        ) from exc

    tasks = mteb.get_tasks(tasks=task_names)
    resolved_names = [task.metadata.name for task in tasks]
    missing = sorted(set(task_names).difference(resolved_names))
    if missing:
        raise ValueError(f"MTEB 中不存在任务: {missing}")
    local_tasks = []
    sample_sizes = {}
    task_sizes = {}
    snapshot_provenance = {}
    if data_root is not None:
        for task in tasks:
            if _inject_local_snapshot(task, data_root):
                local_tasks.append(task.metadata.name)
                snapshot_provenance[task.metadata.name] = _snapshot_provenance(
                    data_root / task.metadata.name
                )

    for task in tasks:
        task_sizes[task.metadata.name] = _capture_task_size(task)
        if max_queries > 0 and max_corpus > 0:
            sample_sizes[task.metadata.name] = _sample_retrieval_task(
                task,
                max_queries=max_queries,
                max_corpus=max_corpus,
            )

    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    model = SentenceTransformer(
        model_name,
        trust_remote_code=False,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    model.max_seq_length = max_seq_length
    evaluator = mteb.MTEB(tasks=tasks)
    results = evaluator.run(
        model,
        output_folder=str(output_dir),
        overwrite_results=overwrite,
        raise_error=True,
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        },
    )
    return {
        "model": model_name,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "device": resolved_device,
        "local_files_only": local_files_only,
        "local_snapshot_tasks": local_tasks,
        "task_sizes": task_sizes,
        "snapshot_provenance": snapshot_provenance,
        "evaluation_scope": (
            "deterministic_sampled_smoke" if sample_sizes else "full_task"
        ),
        "sample_sizes": sample_sizes,
        "tasks": [summarize_task_result(result) for result in results],
        "metric_scope": (
            "Embedding retrieval benchmark only. These scores are not "
            "C-Eval/GSM8K accuracy, RAGAS quality, or mistake-diagnosis quality."
        ),
    }


def main() -> None:
    from config import MATH_EMBEDDING_MODEL_PATH

    parser = argparse.ArgumentParser(
        description="Run BEIR SciFact and C-MTEB retrieval evaluation."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="MTEB task names; defaults to SciFact and CovidRetrieval.",
    )
    parser.add_argument("--model", default=MATH_EMBEDDING_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-corpus", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Embedding 设备；auto 优先使用 CUDA，否则使用 CPU。",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="仅从本地 Hugging Face 缓存加载 embedding 模型。",
    )
    parser.add_argument("--version", default="dev")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="可选本地固定快照目录，子目录名为 SciFact/CovidRetrieval。",
    )
    args = parser.parse_args()

    output_dir = REPORT_DIR / args.version / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_mteb(
        task_names=args.tasks,
        model_name=args.model,
        output_dir=output_dir,
        batch_size=max(1, args.batch_size),
        overwrite=args.overwrite,
        data_root=args.data_root,
        max_seq_length=max(64, args.max_seq_length),
        max_queries=max(0, args.max_queries),
        max_corpus=max(0, args.max_corpus),
        device=args.device,
        local_files_only=args.local_files_only,
    )
    report_path = REPORT_DIR / args.version / "summary.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**summary, "report": str(report_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
