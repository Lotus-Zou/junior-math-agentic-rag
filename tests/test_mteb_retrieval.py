import json
from types import SimpleNamespace

from evaluation.mteb_retrieval import (
    PROJECT_ROOT,
    _capture_task_size,
    _inject_local_snapshot,
    _sample_retrieval_task,
    summarize_task_result,
)


def test_mteb_runner_resolves_project_root():
    assert (PROJECT_ROOT / "config.py").is_file()


def test_scifact_local_snapshot_is_injected(tmp_path):
    root = tmp_path / "SciFact"
    root.mkdir()
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "T", "text": "Evidence"}) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"_id": "q1", "text": "Claim"}),
                json.dumps({"_id": "q-unjudged", "text": "Other split"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "qrels.jsonl").write_text(
        json.dumps({"query-id": "q1", "corpus-id": "d1", "score": 1}) + "\n",
        encoding="utf-8",
    )
    task = SimpleNamespace(
        metadata=SimpleNamespace(name="SciFact", eval_splits=["test"]),
        data_loaded=False,
    )

    assert _inject_local_snapshot(task, tmp_path) is True
    assert task.corpus["test"]["d1"] == "T Evidence"
    assert task.queries["test"]["q1"] == "Claim"
    assert "q-unjudged" not in task.queries["test"]
    assert task.relevant_docs["test"]["q1"]["d1"] == 1


def test_retrieval_sampling_keeps_all_positives():
    task = SimpleNamespace(
        metadata=SimpleNamespace(eval_splits=["test"]),
        corpus={
            "test": {
                "d1": "positive one",
                "d2": "positive two",
                "d3": "distractor",
                "d4": "distractor",
            }
        },
        queries={"test": {"q1": "one", "q2": "two"}},
        relevant_docs={"test": {"q1": {"d2": 1}, "q2": {"d1": 1}}},
    )

    sizes = _sample_retrieval_task(task, max_queries=1, max_corpus=2)

    assert sizes == {"queries": 1, "corpus": 2, "positive_documents": 1}
    assert task.queries["test"] == {"q1": "one"}
    assert "d2" in task.corpus["test"]
    assert task.relevant_docs["test"] == {"q1": {"d2": 1}}


def test_capture_task_size_loads_online_task_and_counts_qrels():
    task = SimpleNamespace(
        metadata=SimpleNamespace(name="OnlineTask", eval_splits=["dev"]),
        data_loaded=False,
    )

    def load_data():
        task.corpus = {"dev": {"d1": "one", "d2": "two"}}
        task.queries = {"dev": {"q1": "one", "q2": "two"}}
        task.relevant_docs = {
            "dev": {"q1": {"d1": 1}, "q2": {"d1": 1, "d2": 1}}
        }
        task.data_loaded = True

    task.load_data = load_data

    assert _capture_task_size(task) == {
        "split": "dev",
        "corpus": 2,
        "judged_queries": 2,
        "qrels": 3,
    }
    assert task.data_loaded is True


def test_mteb_summary_preserves_task_and_averages_subsets():
    result = SimpleNamespace(
        task_name="CovidRetrieval",
        dataset_revision="abc",
        mteb_version="1.39.7",
        evaluation_time=12.5,
        scores={
            "dev": [
                {
                    "hf_subset": "default",
                    "main_score": 0.4,
                    "ndcg_at_10": 0.4,
                    "map_at_10": 0.3,
                    "recall_at_10": 0.6,
                },
                {
                    "hf_subset": "other",
                    "main_score": 0.6,
                    "ndcg_at_10": 0.6,
                    "map_at_10": 0.5,
                    "recall_at_10": 0.8,
                },
            ]
        },
    )

    summary = summarize_task_result(result)

    assert summary["task"] == "CovidRetrieval"
    assert summary["splits"]["dev"]["ndcg_at_10"] == 0.5
    assert summary["splits"]["dev"]["map_at_10"] == 0.4
    assert summary["splits"]["dev"]["recall_at_10"] == 0.7
