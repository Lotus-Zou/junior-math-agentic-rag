import pytest
from fastapi.testclient import TestClient

from app import app


PUBLIC_RESPONSE_TYPES = {
    "verified_answer",
    "guided_exercise",
    "clarification_required",
    "supported_refusal",
}
INTERNAL_FIELDS = {
    "critic_report",
    "validation_evidence",
    "exercise_answer_hidden",
    "hidden_answer",
    "model_metadata",
    "retrieval_trace",
}


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("query", "language", "response_type"),
    [
        ("几何", "zh", "guided_exercise"),
        ("代数", "zh", "guided_exercise"),
        ("一次函数", "zh", "guided_exercise"),
        ("难一点", "zh", "clarification_required"),
        ("换个问题", "zh", "clarification_required"),
        ("reset", "en", "clarification_required"),
    ],
)
def test_short_commands_return_only_public_api_contract(client, query, language, response_type):
    response = client.post("/ask", json={"query": query, "language": language})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == response_type
    assert payload["response_type"] in PUBLIC_RESPONSE_TYPES
    assert payload["metrics"]["tool_calls"] == 0
    assert not (INTERNAL_FIELDS & payload.keys())


def test_geometry_never_reaches_complex_reasoning(client):
    response = client.post("/ask", json={"query": "几何", "language": "zh"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "guided_exercise"
    assert payload["metrics"]["tool_calls"] == 0
    assert "复杂推理服务" not in payload["answer"]
    assert not (INTERNAL_FIELDS & payload.keys())


def test_new_question_clears_previous_state(client):
    response = client.post(
        "/ask",
        json={
            "query": "换个问题",
            "language": "zh",
            "conversation_summary": "旧摘要",
            "conversation_history": [{"role": "student", "content": "解方程 2x+3=11"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification_required"
    assert payload["conversation_summary"] == ""
    assert "2x+3=11" not in str(payload["conversation_history"])
