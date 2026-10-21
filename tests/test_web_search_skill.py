from datetime import datetime, timedelta, timezone

import httpx

import config
from agentic_rag.domain.schemas import RetrievalInput, RetrievalOutput
from agentic_rag.skill_handlers import tavily_search
from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.pipeline import PipelineExecutor
from agentic_rag.skill_runtime.registry import get_default_registry


def _context(**flags):
    return SkillContext(
        request_id="web-search",
        trace_id="web-search",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=15),
        feature_flags={"web_search": True, **flags},
    )


def test_tavily_search_is_disabled_without_key(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")

    result = tavily_search(RetrievalInput(query="初中数学冷门定理"), _context())

    assert result.candidates == []
    assert result.trace[0]["mode"] == "disabled"


def test_tavily_search_returns_bounded_untrusted_citations(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(config, "TAVILY_MAX_RESULTS", 2)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "等式性质",
                        "url": "https://example.edu/equality",
                        "content": "等式两边同时加上同一个数，等式仍成立。",
                        "score": 0.9,
                    },
                    {
                        "title": "无效地址",
                        "url": "javascript:alert(1)",
                        "content": "不应采用",
                        "score": 1,
                    },
                    {
                        "title": "一次函数",
                        "url": "https://example.edu/function",
                        "content": "一次函数 y=kx+b 的图像是一条直线。",
                        "score": 0.8,
                    },
                ]
            }

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _url, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer test-key"
            assert "api_key" not in kwargs["json"]
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = tavily_search(
        RetrievalInput(query="等式性质", top_k=5),
        _context(),
    )

    assert len(result.candidates) == 2
    assert all(item.metadata["untrusted_external_content"] for item in result.candidates)
    assert all(item.source.startswith("https://") for item in result.candidates)
    assert "test-key" not in result.model_dump_json()


def test_retrieval_output_controls_evidence_branch():
    assert PipelineExecutor.branch_key(RetrievalOutput(candidates=[])) == "fail"
    assert (
        PipelineExecutor.branch_key(
            RetrievalOutput(
                candidates=[
                    {
                        "chunk_id": "web-1",
                        "content": "可核验内容",
                        "source": "https://example.edu/source",
                    }
                ]
            )
        )
        == "pass"
    )


def test_tavily_skill_is_read_only_open_world_mcp_tool():
    manifest = get_default_registry().resolve("web.tavily_search@1")

    assert manifest.expose.mcp is True
    assert manifest.idempotent is True
    assert manifest.side_effects == []
    assert "network" in manifest.required_capabilities
