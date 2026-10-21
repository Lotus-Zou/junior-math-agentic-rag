from agentic_rag.skill_runtime.adapters.langchain_tools import LangChainToolAdapter
from agentic_rag.skill_runtime.adapters.mcp import MCPRegistryAdapter
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import get_default_registry


def test_langchain_and_mcp_share_wrapped_input_schema():
    registry = get_default_registry()
    executor = SkillExecutor(registry)
    langchain = LangChainToolAdapter(registry, executor)
    mcp_schemas = MCPRegistryAdapter(registry, executor).schemas()
    for manifest in registry.list(mcp=True):
        tool = langchain.build(manifest.ref)
        assert tool.args_schema.model_json_schema() == mcp_schemas[tool.name]
