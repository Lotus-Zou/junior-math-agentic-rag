from types import SimpleNamespace

import pytest

from agentic_rag.domain.schemas import TurnRouteOutput
from agentic_rag.skill_runtime.pipeline import PipelineExecutor
from tools.evening_capacity_test import acceptance
from tools.realistic_load_test import summarize


@pytest.mark.parametrize("ref", [
    "math.answer_generate@1", "math.answer_critic@1",
    "math.answer_repair@1", "math.answer_final_judge@1",
    "math.targeted_fallback@1",
])
def test_answer_lifecycle_preserves_specific_router_intent(ref):
    manifest = SimpleNamespace(input_model=SimpleNamespace(model_fields={"query": None, "intent": None}))
    runner = PipelineExecutor(SimpleNamespace(registry=SimpleNamespace(resolve=lambda _: manifest)))
    turn = TurnRouteOutput(intent="multiple_choice", route="rag", routed_query="Compute 2+2; A. 3 B. 4", reason="complete choice problem")
    state = {"turn_router": turn, "question_parse": {"intent": "solve"}}
    result = runner._project_input(ref, state, {}, None)
    assert result["intent"] == "multiple_choice"


def test_rejections_cannot_hide_slow_successful_answers():
    rows = [{"status": 429, "latency_ms": 1, "business_success": False} for _ in range(99)]
    rows.append({"status": 200, "latency_ms": 30000, "business_success": True})
    summary = summarize(rows)
    assert summary["latency_ms"]["p95"] == 1
    assert summary["successful_answer_latency_ms"]["p95"] == 30000
    result = acceptance({"summary": summary, "completed_answers_per_second": 1}, 1, 20000, 1)
    assert not result["checks"]["p95_latency"]
    assert not result["checks"]["business_success_rate"]
    assert not result["passed"]


def test_public_corpus_excludes_heldout_stems_and_permuted_choices(tmp_path):
    import json
    import sqlite3
    from tools.prepare_public_corpus import build_corpus
    from agentic_rag.public_case_retriever import retrieve_public_cases

    def case(key, stem, choices):
        return {"queId": key, "problem": stem, "dataset_name": "mid_math_competition",
                "answer_option_list": [[{"aoVal": label, "content": text}] for label, text in zip("AB", choices)],
                "answer_value": "B", "answer_analysis": ["independent publisher reasoning"]}

    source = tmp_path / "ch_single_choice_constructed_5K"
    source.mkdir()
    train = [case("train", "方程 2x=8 求解", ["3", "4"]),
             case("duplicate", "方程 2x=8 求解", ["4", "3"]),
             case("leak", "方程 3x=9 求解", ["3", "9"])]
    test = [case("test", "方程 3x=9 求解", ["1", "3"])]
    for split, size, rows in (("train", 3, train), ("test", 2, test)):
        (source / f"ch_single_choice_{split}_{size}K.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    database = tmp_path / "corpus.sqlite3"
    result = build_corpus(tmp_path, database)
    assert result["unique_train_questions"] == 1
    assert result["train_test_overlaps_excluded"] == 1
    assert result["duplicate_train_rows_excluded"] == 1
    matches = retrieve_public_cases(["方程 2x=8"], str(database), 3)
    assert len(matches) == 1
    assert matches[0].metadata["source_id"] == "train"
    assert matches[0].metadata["split"] == "train"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1


def test_clarification_is_not_a_completed_answer():
    import io
    from tools.realistic_load_test import answer_complete, read_sse

    assert not answer_complete({'answer': 'Please provide a figure', 'response_type': 'clarification_required'})
    assert not answer_complete({'answer': 'x=4'}, require_verified=True)
    assert answer_complete({'answer': 'x=4', 'response_type': 'verified_answer', 'validation_passed': True}, True)
    stream = io.BytesIO(b'event: result\ndata: {"answer":"provide a figure","response_type":"clarification_required"}\n\nevent: done\ndata: {}\n\n')
    assert not read_sse(stream, 0, require_verified=True)['business_success']


def test_missing_project_evidence_cannot_pass(tmp_path):
    from tools.audit_project_evidence import audit
    result = audit(tmp_path, tmp_path/'manifest.json')
    assert result['artifact_count'] == 0
    assert not result['structurally_complete']
    assert not any(claim['verified'] for claim in result['claims'].values())


def test_third_party_annotation_does_not_prove_project_human_labels(tmp_path):
    import hashlib
    import json
    from tools.audit_project_evidence import audit
    body = json.dumps({'problem_sha256': 'a'*64, 'annotator_id': 'publisher',
                       'answer_label': 'B', 'error_type': 'arithmetic',
                       'method': 'human', 'annotated_at': '2026-01-01T00:00:00Z'})
    (tmp_path/'public.jsonl').write_text(body)
    (tmp_path/'manifest.json').write_text(json.dumps({'artifacts': [{
        'kind': 'annotations', 'path': 'public.jsonl', 'sha256': hashlib.sha256(body.encode()).hexdigest(),
        'origin': 'third_party_public', 'recorded_at': '2026-01-01T00:00:00Z'}]}))
    result = audit(tmp_path, tmp_path/'manifest.json')
    assert result['artifact_count'] == 1
    assert result['claims']['human_annotations']['unique_questions'] == 0


def test_evidence_hash_tampering_is_rejected(tmp_path):
    import json
    from tools.audit_project_evidence import audit
    (tmp_path/'proof.json').write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'artifacts': [{
        'kind': 'school_authorization', 'path': 'proof.json', 'sha256': '0'*64,
        'origin': 'project_primary', 'recorded_at': '2026-01-01T00:00:00Z'}]}))
    result = audit(tmp_path, tmp_path/'manifest.json')
    assert result['artifact_count'] == 0
    assert result['errors'][0]['error'] == 'SHA256 mismatch'


def test_competition_failure_bound_and_cache_invalidation():
    from tools.evaluate_competition_target import summarize as competition_summary
    rows = [{'exact_match': index < 9, 'prediction': 'A', 'status': 200, 'response': {}}
            for index in range(68)]
    result = competition_summary(rows, 512, 500)
    assert result['point_target_unreachable']
    assert result['best_possible_final_accuracy'] == 453/512
    assert not result['complete']
    assert not result['point_target_met']
    cached = [{'exact_match': True, 'prediction': 'A', 'status': 200, 'response': {'cached': True}}]
    assert not competition_summary(cached, 1, 1)['point_target_met']


def test_optional_public_corpus_failure_preserves_primary_retrieval(monkeypatch):
    import sqlite3
    from langchain_core.documents import Document
    from agentic_rag.domain.schemas import RetrievalInput
    from agentic_rag.math_retriever import math_retriever
    from agentic_rag import public_case_retriever, skill_handlers

    document = Document(page_content='An independent textbook fact', metadata={'chunk_id': 'textbook-1'})
    monkeypatch.setattr(math_retriever, 'retrieve_bm25_channel', lambda *args: ([document], []))
    def broken(*args):
        raise sqlite3.DatabaseError('corrupted optional corpus')
    monkeypatch.setattr(public_case_retriever, 'retrieve_public_cases', broken)
    output = skill_handlers.retrieve_bm25(RetrievalInput(query='Solve 2x=8', top_k=3), None)
    assert len(output.candidates) == 1
    assert output.trace[-1]['degraded'] is True
    assert output.trace[-1]['test_rows_allowed'] is False
