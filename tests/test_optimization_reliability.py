import io
import json
from types import SimpleNamespace

import pytest

import app as api
from agentic_rag import chains, skill_handlers
from agentic_rag.domain.schemas import AnswerCriticInput, CriticOutput, RetrievalOutput
from tools import realistic_load_test as load


@pytest.mark.parametrize("factual,logical,valid", [(False, True, False), (True, False, False), (True, True, False)])
def test_rejected_critic_always_gives_repair_action(monkeypatch, factual, logical, valid):
    monkeypatch.setattr(chains, "get_answer_validation_chain", lambda: SimpleNamespace(invoke=lambda _: {
        "is_valid": valid, "factual_faithfulness": factual, "math_logic_valid": logical, "issues": [],
    }))
    data = AnswerCriticInput(query="为什么移项要变号？", answer="移项本质上是两边加相反数。[1]", contexts=["等式两边同时加减同一个数，等式仍成立。"])
    result = skill_handlers.answer_critic(data, SimpleNamespace(feature_flags={"llm_critic": True}))
    assert not result.passed and result.issues


def test_critic_defect_report_reaches_repair_even_when_issue_list_is_empty(monkeypatch):
    monkeypatch.setattr(chains, "get_answer_validation_chain", lambda: SimpleNamespace(invoke=lambda _: {
        "is_valid": False, "factual_faithfulness": True, "math_logic_valid": False,
        "issues": [], "defect_report": "The example subtracts 2x on only one side.",
    }))
    result = skill_handlers.answer_critic(AnswerCriticInput(query="Why move a term?", answer="rule [1]", contexts=["rule"]),
        SimpleNamespace(feature_flags={"llm_critic": True}))
    assert "The example subtracts 2x on only one side." in result.issues


def test_self_contained_answer_cannot_bypass_nonexistent_citation(monkeypatch):
    monkeypatch.setattr(chains, "get_answer_validation_chain", lambda: SimpleNamespace(invoke=lambda _: {
        "is_valid": True, "factual_faithfulness": True, "math_logic_valid": True, "issues": [],
    }))
    data = AnswerCriticInput(query="计算 11+4\nA. 14\nB. 15\nC. 16\nD. 17", answer="11+4=15。[999]\n答案：B", contexts=["addition"])
    result = skill_handlers.answer_critic(data, SimpleNamespace(feature_flags={"llm_critic": True}))
    assert not result.passed
    assert any("引用编号不存在" in issue for issue in result.issues)


def test_pipeline_keeps_each_critic_diagnostic_private(monkeypatch):
    from agentic_rag.domain.schemas import AnswerEnvelope

    first = CriticOutput(passed=False, factual_faithfulness=False, math_logic_valid=True, issues=["missing support"], critic={"validation_mode": "llm"})
    second = first.model_copy(update={"passed": True, "factual_faithfulness": True, "issues": []})
    response = AnswerEnvelope(answer="unverified excerpt", response_type="general_answer", validation_passed=False)
    state = {"answer_critic": first, "answer_repair_critic": second, "response_render": response,
             "rerank_filter": RetrievalOutput(candidates=[])}
    monkeypatch.setattr(api, "get_graph", lambda: SimpleNamespace(invoke=lambda *_: {"skill_pipeline": state}))
    # No trusted publication proof is minted for an unverified excerpt.
    state["answer_repair_critic"] = first
    result = api._run_curriculum_skill(api.AskRequest(query="why?"))
    assert [row["node"] for row in result.critic_checks] == ["answer_critic", "answer_repair_critic"]
    assert result.critic_checks[0]["issues"] == ["missing support"]
    assert "critic_checks" not in result.response


def test_load_summary_never_counts_cached_or_replayed_model_work_twice():
    rows = [{"status": 200, "latency_ms": 10, "business_success": True,
             "model_attempts": 4, **flags} for flags in ({}, {"cached": True}, {"replay": True})]
    summary = load.summarize(rows)
    assert summary["uncached_model_attempts"] == 4
    assert summary["answer_cache_hits"] == summary["replay_requests"] == 1
    assert summary["requests"] == 3 and summary["business_success_rate"] == 1


def test_stream_final_exposes_same_execution_metadata_as_http():
    payload = {"answer": "x=4", "response_type": "verified_answer", "validation_passed": True,
               "cached": True, "metrics": {"model_attempts": 4, "execution_path": "agentic_rag"}}
    stream = io.BytesIO(("event: result\ndata: " + json.dumps(payload) + '\n\nevent: done\ndata: {}\n\n').encode())
    result = load.read_sse(stream, 0, require_verified=True)
    assert result["business_success"] and result["cached"]
    assert result["model_attempts"] == 4 and result["execution_path"] == "agentic_rag"


def test_load_run_id_isolates_authenticated_users_and_sessions(monkeypatch):
    sent = []

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        sent.append(json.loads(request.data))
        return Response(json.dumps({"answer": "x=4", "response_type": "verified_answer", "validation_passed": True}).encode())

    monkeypatch.setattr(load.urllib.request, "urlopen", urlopen)
    for namespace in ("first", "second"):
        assert load.request_one("http://localhost/ask", "test-secret", "steady", 0, "2x=8", False, "key", 1,
                                identity_namespace=namespace)["business_success"]
    assert sent[0]["user_id"] != sent[1]["user_id"]
    assert sent[0]["session_id"] != sent[1]["session_id"]


def test_capacity_command_refuses_to_overwrite_old_evidence(monkeypatch, tmp_path):
    import sys
    from tools import evening_capacity_test as capacity

    output = tmp_path / "prior.json"
    output.write_text('{"passed":false}')
    monkeypatch.setattr(sys, "argv", ["capacity", "--url", "http://localhost/ask", "--min-steady-rps", "1", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        capacity.main()
    assert error.value.code == 2
    assert output.read_text() == '{"passed":false}'
