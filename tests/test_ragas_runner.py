import sys

from evaluation.ragas_runner import _ensure_legacy_vertexai_shim


def test_legacy_vertexai_compatibility_is_installed_before_ragas_import():
    module_name = "langchain_community.chat_models.vertexai"

    _ensure_legacy_vertexai_shim()
    _ensure_legacy_vertexai_shim()

    assert module_name in sys.modules
    assert hasattr(sys.modules[module_name], "ChatVertexAI")
