"""Generate LangChain StructuredTools without duplicated business signatures."""

from langchain_core.tools import StructuredTool

from agentic_rag.skill_runtime.adapters.common import adapter_context, wrapped_input_model


class LangChainToolAdapter:
    def __init__(self, registry, executor):
        self.registry, self.executor = registry, executor

    def build(self, ref: str) -> StructuredTool:
        manifest = self.registry.resolve(ref)

        def invoke(payload):
            result = self.executor.execute(manifest.ref, payload, adapter_context(timeout_ms=manifest.timeout_ms))
            return result.model_dump(mode="json")

        return StructuredTool.from_function(
            func=invoke, name=manifest.id.replace(".", "_"), description=manifest.description,
            args_schema=wrapped_input_model(manifest),
        )

    def all_tools(self):
        return [self.build(item.ref) for item in self.registry.list() if item.expose.langchain]
