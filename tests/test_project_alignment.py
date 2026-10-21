from copy import deepcopy
import csv
from pathlib import Path
import shlex
from types import SimpleNamespace

import pandas as pd
import pytest

import app as api
from agentic_rag import chains, skill_handlers
from agentic_rag.domain.schemas import AnswerEnvelope, AnswerRepairInput, RetrievalCandidate, RetrievalOutput, RenderInput
from agentic_rag.skill_runtime.pipeline import PipelineExecutor
from agentic_rag.response_contract import capture_skill_contract, consume_skill_contract
from evaluation.dataset_audit import audit_dataset
from evaluation.evaluation import coverage_gate, quality_gate
from evaluation.generate_dataset import build_rows
from evaluation.paired_evaluation import compare_runs, export_run, fingerprint, METRICS
from evaluation.ragas_runner import _valid_metric_record


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset_size,evaluated", [(5, 2), (2, 2), (5, True), ("5", 5), (5, 0)])
def test_partial_experiments_cannot_pass_full_quality_gate(dataset_size, evaluated):
    assert not coverage_gate({"dataset_size": dataset_size, "evaluated": evaluated}, 5)[0]


def test_coverage_gate_requires_actual_full_dataset_size():
    assert coverage_gate({"dataset_size": 5, "evaluated": 5}, 5) == (True, [])


def test_gateway_image_includes_all_production_go_sources():
    copied = set()
    for line in (ROOT / "gateway/Dockerfile").read_text().splitlines():
        tokens = shlex.split(line)
        if tokens and tokens[0] == "COPY" and "--from=" not in line:
            for pattern in tokens[1:-1]:
                copied.update(path.name for path in (ROOT / "gateway").glob(pattern))
    sources = {p.name for p in (ROOT / "gateway").glob("*.go") if not p.name.endswith("_test.go")}
    assert "routes.go" in sources
    assert sources <= copied


@pytest.mark.parametrize("citation", ["", " [1]", " [999]"])
def test_final_judge_cannot_claim_unseen_retrieval_sources(monkeypatch, citation):
    monkeypatch.setattr(chains, "get_final_answer_judge_chain", lambda: SimpleNamespace(
        invoke=lambda _: {"is_complete": True, "math_logic_valid": True,
                          "corrected_answer": "11+4=15，因此选择 B。" + citation + "\n答案：B"}))
    request = AnswerRepairInput(query="计算 11+4\nA. 14\nB. 15\nC. 16\nD. 17",
                                answer="rejected", intent="multiple_choice",
                                contexts=[RetrievalCandidate(content="Unrelated example", source="unseen.md")])
    context = SimpleNamespace(trace_id="source-test", feature_flags={"llm_critic": True})
    if citation:
        with pytest.raises(ValueError, match="evidence it did not receive"):
            skill_handlers.answer_final_judge(request, context)
    else:
        assert skill_handlers.answer_final_judge(request, context).sources == []


@pytest.mark.parametrize("language", ["zh", "en"])
def test_retrieval_excerpt_is_publishable_but_never_verified_or_cached(language):
    data = AnswerRepairInput(query="Why reverse the inequality?", answer="rejected",
                             intent="knowledge_query", language=language,
                             contexts=[RetrievalCandidate(content="Multiplication by a negative reverses order.", source="first.md"),
                                       RetrievalCandidate(content="Unused", source="unused.md")])
    with capture_skill_contract() as capture:
        result = skill_handlers.targeted_fallback(data, SimpleNamespace(trace_id="excerpt-test"))
        contract = consume_skill_contract(capture)
    public = api._public_response(result.model_dump(mode="json"), api.AskRequest(query=data.query), contract=contract)
    assert public["response_type"] == "general_answer"
    assert public["validation_passed"] is False
    assert [source["source"] for source in public["sources"]] == ["first.md"]
    assert not contract
    assert not api._cacheable_response(public, True)
    rendered = skill_handlers.response_render(RenderInput(query=data.query, **result.model_dump()), None)
    assert rendered.response_type == "general_answer"
    assert rendered.validation_passed is False
    assert rendered.answer == result.answer


@pytest.mark.parametrize("response_type,validated", [
    ("clarification_required", True), ("supported_refusal", True),
    ("general_answer", False), ("verified_answer", False), ("general_answer", None),
])
def test_incomplete_or_unvalidated_answers_do_not_count_as_business_success(response_type, validated):
    from tools.realistic_load_test import answer_complete

    assert not answer_complete({"answer": "nonempty", "response_type": response_type,
                                "validation_passed": validated})


def test_pipeline_preserves_final_judge_empty_sources():
    response = AnswerEnvelope(answer="11+4=15", response_type="verified_answer", validation_passed=True)
    state = {"rerank_filter": RetrievalOutput(candidates=[RetrievalCandidate(content="unseen", source="unseen.md")]),
             "answer_final_judge": response}
    projected = PipelineExecutor(api._skill_executor)._project_input(
        "math.response_render@1", state, {"query": "11+4?", "language": "zh"}, None)
    assert projected["sources"] == []
    assert skill_handlers.response_render(RenderInput.model_validate(projected), None).sources == []


def test_pipeline_preserves_only_quoted_excerpt_and_unverified_status():
    candidates = [RetrievalCandidate(content="textbook property", source="quoted.md"),
                  RetrievalCandidate(content="unused", source="unused.md")]
    response = skill_handlers.targeted_fallback(AnswerRepairInput(
        query="为什么移项要改变符号？", answer="rejected", intent="knowledge_query", contexts=candidates), None)
    state = {"rerank_filter": RetrievalOutput(candidates=candidates), "targeted_fallback": response}
    projected = PipelineExecutor(api._skill_executor)._project_input(
        "math.response_render@1", state, {"query": "为什么移项要改变符号？", "language": "zh"}, None)
    rendered = skill_handlers.response_render(RenderInput.model_validate(projected), None)
    assert rendered.response_type == "general_answer" and not rendered.validation_passed
    assert rendered.answer == response.answer
    assert [source.source for source in rendered.sources] == ["quoted.md"]


def test_citation_repair_receives_critic_issue_and_original_evidence(monkeypatch):
    calls = []
    def invoke(messages):
        calls.append(messages)
        return SimpleNamespace(content="一次函数 y=kx+b 的斜率是 k。[1] 本题斜率为 -2。")

    monkeypatch.setattr(chains, "generator_llm", SimpleNamespace(invoke=invoke))
    response = skill_handlers.answer_repair(AnswerRepairInput(
        query="y=-2x+3 的斜率是什么？", answer="斜率为 -2。", intent="knowledge_query",
        issues=["关键步骤没有教材引用"],
        contexts=[RetrievalCandidate(content="一次函数 y=kx+b 的斜率是 k。", source="slope.md")]),
        SimpleNamespace(feature_flags={"llm_generate": True}))
    import json
    payload = json.loads(calls[0][1].content)
    assert payload["critic_issues"] == ["关键步骤没有教材引用"]
    assert "[1] 一次函数 y=kx+b 的斜率是 k。" == payload["retrieved_evidence"]
    assert "Returning the same uncited draft" in calls[0][0].content
    assert "[1]" in response.answer


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, None, "0.99", -0.1, 1.1])
def test_invalid_quality_scores_cannot_pass(invalid):
    scores = {"context_precision": 1., "context_recall": 1., "faithfulness": 1.,
              "answer_relevance": invalid, "knowledge_point_accuracy": 1., "direct_answer_violation_rate": 0.}
    passed, errors = quality_gate(scores)
    assert not passed and any("answer_relevance" in error for error in errors)
    assert not _valid_metric_record({"context_precision": 1., "context_recall": 1.,
                                     "faithfulness": 1., "answer_relevancy": invalid})


def experiment(scores=(.5, .7)):
    return {"schema_version": 1, "metadata": {"dataset_sha256": "dataset", "evaluator_sha256": "evaluator",
            "judge_model": "judge", "judge_settings": {"provider": "test"},
            "embedding_model": "embedding", "embedding_settings": {"provider": "test"},
            "metric_definition": {"k": 5}},
            "cases": [{"case_id": str(i), "question_sha256": fingerprint(i),
                       "answer_sha256": fingerprint((i, score)), "completed": True,
                       "scores": dict.fromkeys(METRICS, score)} for i, score in enumerate(scores)]}


def test_paired_comparison_retains_failures_in_denominator():
    before, after = experiment(), experiment((.8, .9))
    after["cases"][1]["completed"] = False
    report = compare_runs(before, after, samples=50)
    assert report["metrics"]["answer_relevance"]["after"] == .4
    assert report["metrics"]["answer_relevance"]["delta"] == pytest.approx(-.2)
    assert report["failed_cases"]["after"] == 1
    assert report["historical_claim_verified"] is False


@pytest.mark.parametrize("field", ["dataset_sha256", "evaluator_sha256", "judge_model", "judge_settings",
                                  "embedding_model", "embedding_settings", "metric_definition"])
def test_paired_comparison_rejects_mismatched_identity(field):
    before, after = experiment(), experiment()
    after["metadata"][field] = "changed"
    with pytest.raises(ValueError, match="evaluation identity"):
        compare_runs(before, after)


@pytest.mark.parametrize("change", ["drop", "duplicate", "stem"])
def test_paired_comparison_rejects_unpaired_cases(change):
    before, after = experiment(), experiment()
    if change == "drop":
        after["cases"].pop()
    elif change == "duplicate":
        after["cases"].append(deepcopy(after["cases"][0]))
    else:
        after["cases"][0]["question_sha256"] = fingerprint("different question")
    with pytest.raises(ValueError):
        compare_runs(before, after)


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), True, 2.])
def test_paired_comparison_never_drops_failed_judge_scores(missing):
    before, after = experiment(), experiment()
    after["cases"][0]["scores"]["answer_relevance"] = missing
    result = compare_runs(before, after)
    metric = result["metrics"]["answer_relevance"]
    assert result["complete"] is False
    assert metric["comparable"] is False and metric["delta"] is None
    assert metric["planned_cases"] == 2


def test_e2e_export_preserves_unscored_cases_and_refuses_overwrite(tmp_path):
    rows = build_rows()[:2]
    report = pd.DataFrame([{"question_id": row["question_id"], "answer": "result",
                            "response_type": "verified_answer", "validation_passed": True,
                            "source_recall_at_k": 1.} for row in rows])
    output = tmp_path / "run.json"
    export_run(frame=pd.DataFrame(rows), report=report, summary={"version": "test"}, output=output, k=5)
    import json
    run = json.loads(output.read_text())
    result = compare_runs(run, run)
    assert result["metrics"]["source_recall_at_k"]["delta"] == 0
    assert result["metrics"]["answer_relevance"]["comparable"] is False
    with pytest.raises(FileExistsError):
        export_run(frame=pd.DataFrame(rows), report=report, summary={"version": "test"}, output=output, k=5)


def test_dataset_provenance_does_not_claim_human_labels():
    result = audit_dataset()
    assert result["rows"] == result["unique_normalized_stems"] == 1000
    assert result["annotation_method"] == "deterministic_programmatic"
    assert result["human_reviewed_cases"] == 0
    assert len(result["source_files"]) == 10


@pytest.mark.parametrize("change", ["label", "stem", "duplicate", "drop"])
def test_dataset_audit_rejects_tampering(tmp_path, change):
    rows = build_rows()
    if change == "label":
        rows[0]["reference_answer"] = "tampered"
    elif change == "stem":
        rows[0]["problem_text"] = "tampered"
    elif change == "duplicate":
        rows[1] = deepcopy(rows[0])
    else:
        rows.pop()
    path = tmp_path / "dataset.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        audit_dataset(path)
