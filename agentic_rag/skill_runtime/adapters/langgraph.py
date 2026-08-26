"""Compile a validated Pipeline into a LangGraph execution node."""

from langgraph.graph import END, StateGraph

from agentic_rag.skill_runtime.pipeline import PipelineExecutor


class LangGraphPipelineAdapter:
    def __init__(self, executor):
        self.executor = executor

    def compile(self, pipeline, context_factory):
        runner = PipelineExecutor(self.executor)
        graph = StateGraph(dict)

        def execute(state: dict) -> dict:
            return {**state, "skill_pipeline": runner.run(pipeline, state, context_factory(state))}

        graph.add_node("skill_pipeline", execute)
        graph.set_entry_point("skill_pipeline")
        graph.add_edge("skill_pipeline", END)
        return graph.compile()
