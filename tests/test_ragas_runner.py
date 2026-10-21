import sys
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from evaluation.ragas_runner import _ensure_legacy_vertexai_shim
from evaluation import ragas_runner as runner


def test_legacy_vertexai_compatibility_is_installed_before_ragas_import():
    module_name = "langchain_community.chat_models.vertexai"

    _ensure_legacy_vertexai_shim()
    _ensure_legacy_vertexai_shim()

    assert module_name in sys.modules
    assert hasattr(sys.modules[module_name], "ChatVertexAI")


@pytest.fixture
def scoring(monkeypatch):
    _ensure_legacy_vertexai_shim()
    import ragas

    calls = []
    def evaluate(dataset, **kwargs):
        calls.append(dataset.to_list())
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame([
            dict.fromkeys(runner.METRIC_COLUMNS, .8) for _ in dataset]))

    monkeypatch.setattr(ragas, "evaluate", evaluate)
    monkeypatch.setattr(runner, "_build_ragas_runtime", lambda: (None, None))
    monkeypatch.setattr(runner, "_evaluation_identity", lambda: "judge-v1")
    return calls


def sample():
    return {"case_id": "case-1", "question": "2+2?", "answer": "4",
            "contexts": ["2+2=4"], "ground_truth": "4"}


def test_checkpoint_resumes_only_exact_input_and_judge(tmp_path, monkeypatch, scoring):
    checkpoint = tmp_path / "scores.jsonl"
    row = sample()
    runner.evaluate_ragas_records([row], checkpoint_path=checkpoint)
    _, resumed = runner.evaluate_ragas_records([row], checkpoint_path=checkpoint)
    assert resumed["resumed_cases"] == 1 and len(scoring) == 1
    changed = {**row, "answer": "5"}
    _, fresh = runner.evaluate_ragas_records([changed], checkpoint_path=checkpoint)
    assert fresh["resumed_cases"] == 0 and len(scoring) == 2
    monkeypatch.setattr(runner, "_evaluation_identity", lambda: "judge-v2")
    _, fresh = runner.evaluate_ragas_records([changed], checkpoint_path=checkpoint)
    assert fresh["resumed_cases"] == 0 and len(scoring) == 3
    records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(records) == 3
    assert records[0]["input_sha256"] != records[1]["input_sha256"]
    assert records[1]["evaluation_sha256"] != records[2]["evaluation_sha256"]


def test_legacy_checkpoint_is_preserved_but_not_reused(tmp_path, scoring):
    checkpoint = tmp_path / "legacy.jsonl"
    legacy = {"case_id": "case-1", **dict.fromkeys(runner.METRIC_COLUMNS, .99)}
    checkpoint.write_text(json.dumps(legacy) + "\n")
    _, result = runner.evaluate_ragas_records([sample()], checkpoint_path=checkpoint)
    assert result["resumed_cases"] == 0 and len(scoring) == 1
    assert json.loads(checkpoint.read_text().splitlines()[0]) == legacy


def test_empty_evaluation_is_not_complete(tmp_path, scoring):
    _, result = runner.evaluate_ragas_records([], checkpoint_path=tmp_path / "empty.jsonl")
    assert result["requested"] == 0 and not result["complete"]
    assert not scoring


def test_duplicate_case_ids_rejected(tmp_path, scoring):
    with pytest.raises(ValueError, match="Duplicate"):
        runner.evaluate_ragas_records([sample(), sample()], checkpoint_path=tmp_path / "dup.jsonl")
    assert not scoring
