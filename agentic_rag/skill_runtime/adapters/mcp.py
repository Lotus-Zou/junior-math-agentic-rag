"""Generate FastMCP tools from Registry manifests and shared Pydantic schemas."""

from mcp.types import ToolAnnotations

from agentic_rag.skill_runtime.adapters.common import adapter_context, wrapped_input_model
from agentic_rag.skill_runtime.contracts import SkillResult


class MCPRegistryAdapter:
    def __init__(self, registry, executor):
        self.registry, self.executor = registry, executor

    def register(self, server):
        for manifest in self.registry.list(mcp=True):
            invoke = self._tool_callable(manifest)

            invoke.__name__ = manifest.id.replace(".", "_")
            invoke.__doc__ = manifest.description
            invoke.__annotations__ = {
                "payload": manifest.input_model,
                "return": SkillResult[manifest.output_model],
            }
            options = {
                "name": manifest.id.replace(".", "_"),
                "description": manifest.description,
                "annotations": ToolAnnotations(
                    readOnlyHint=not bool(manifest.side_effects), destructiveHint=False,
                    idempotentHint=manifest.idempotent,
                    openWorldHint="network" in manifest.required_capabilities,
                ),
                "structured_output": True,
            }
            if hasattr(server, "add_tool"):
                server.add_tool(invoke, **options)
            else:
                server.tool(**options)(invoke)
        return server

    def _tool_callable(self, manifest):
        def invoke(payload):
            return self.executor.execute(
                manifest.ref, payload, adapter_context(timeout_ms=manifest.timeout_ms)
            )

        return invoke

    def schemas(self) -> dict[str, dict]:
        return {item.id.replace(".", "_"): wrapped_input_model(item).model_json_schema() for item in self.registry.list(mcp=True)}
