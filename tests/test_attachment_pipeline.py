from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app as api
from agentic_rag.domain.schemas import (
    AttachmentStructureInput,
    AttachmentUploadInput,
)
from agentic_rag.skill_handlers import attachment_extract, attachment_structure
from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.errors import SkillRuntimeError


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _context(**flags):
    return SkillContext(
        request_id="attachment-test",
        trace_id="attachment-test",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=20),
        feature_flags=flags,
    )


def test_attachment_extract_validates_real_image_signature():
    output = attachment_extract(
        AttachmentUploadInput(
            filename="wrong-problem.png",
            media_type="image/png",
            content_base64=base64.b64encode(PNG_1X1).decode("ascii"),
        ),
        _context(),
    )

    assert output.media_type == "image/png"
    assert output.byte_size == len(PNG_1X1)
    assert output.image_width == output.image_height == 1
    assert len(output.sha256) == 64


def test_attachment_extract_rejects_spoofed_media_type():
    try:
        attachment_extract(
            AttachmentUploadInput(
                filename="fake.jpg",
                media_type="image/jpeg",
                content_base64=base64.b64encode(PNG_1X1).decode("ascii"),
            ),
            _context(),
        )
    except SkillRuntimeError as exc:
        assert "文件类型" in exc.safe_message
    else:
        raise AssertionError("spoofed media type was accepted")


def test_attachment_structure_uses_editable_pdf_text_fallback():
    output = attachment_structure(
        AttachmentStructureInput(
            filename="mistake.pdf",
            media_type="application/pdf",
            content_base64=base64.b64encode(b"%PDF-placeholder").decode("ascii"),
            sha256="0" * 64,
            byte_size=16,
            extracted_text="解方程 2x+3=11\n学生错误作答：2x=11+3",
        ),
        _context(attachment_agent=False),
    )

    assert output.status == "needs_confirmation"
    assert output.problem_text == "解方程 2x+3=11"
    assert output.student_answer == "2x=11+3"
    assert output.parser == "pdf_text"


def test_attachment_structure_accepts_fenced_agent_json(monkeypatch):
    def fake_invoke(_self, _messages):
        return AIMessage(
            content=(
                '```json\n{"problem_text":"求 x：x+1=2",'
                '"student_answer":"x=3","formulas":["x+1=2"],'
                '"confidence":0.91,"warnings":[]}\n```'
            )
        )

    from agentic_rag import chains

    monkeypatch.setattr(type(chains.attachment_llm), "invoke", fake_invoke)
    output = attachment_structure(
        AttachmentStructureInput(
            filename="problem.png",
            media_type="image/png",
            content_base64=base64.b64encode(PNG_1X1).decode("ascii"),
            sha256="0" * 64,
            byte_size=len(PNG_1X1),
        ),
        _context(),
    )

    assert output.status == "ready"
    assert output.problem_text == "求 x：x+1=2"
    assert output.formulas == ["x+1=2"]


def test_attachment_api_runs_vision_skill_and_returns_editable_fields(monkeypatch):
    def fake_invoke(_self, _messages):
        return AIMessage(
            content=(
                '{"problem_text":"解方程 2x + 3 = 11",'
                '"student_answer":"2x = 11 + 3",'
                '"formulas":["2x + 3 = 11"],'
                '"confidence":0.96,"warnings":[]}'
            )
        )

    from agentic_rag import chains

    monkeypatch.setattr(type(chains.attachment_llm), "invoke", fake_invoke)
    api._attachment_graph = None

    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/attachments/parse",
            files={"file": ("wrong-problem.png", PNG_1X1, "image/png")},
            data={"language": "zh"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["problem_text"] == "解方程 2x + 3 = 11"
    assert payload["student_answer"] == "2x = 11 + 3"
    assert payload["parser"] == "vision_agent"
    assert payload["trace_id"]
    assert "content_base64" not in payload
    assert "sha256" not in payload


def test_attachment_api_rejects_unsupported_file_type():
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/attachments/parse",
            files={"file": ("notes.txt", b"x = 4", "text/plain")},
            data={"language": "zh"},
        )

    assert response.status_code == 415
    assert "JPG" in response.json()["detail"]
