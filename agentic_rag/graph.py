# -*- coding: utf-8 -*-
"""Workflow + bounded ReAct hybrid for junior-high mathematics correction."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from langgraph.graph import END, StateGraph

from agentic_rag.nodes import (
    classify_knowledge_node,
    clarification_response_node,
    consolidate_memory_node,
    generate_response_node,
    no_evidence_response_node,
    parse_question_node,
    prepare_retry_node,
    react_agent_node,
    rerank_documents_node,
    retrieve_documents_node,
    retrieve_memory_node,
    rewrite_query_node,
    validate_answer_node,
    validation_failure_response_node,
)
from agentic_rag.state import AgentState
from config import MAX_CORRECTION_ATTEMPTS


def build_skill_pipeline_graph():
    """Build the versioned YAML pipeline during strangler migration."""
    from agentic_rag.skill_runtime.adapters.langgraph import LangGraphPipelineAdapter
    from agentic_rag.skill_runtime.contracts import SkillContext
    from agentic_rag.skill_runtime.executor import SkillExecutor
    from agentic_rag.skill_runtime.pipeline import PipelineLoader
    from agentic_rag.skill_runtime.registry import get_default_registry

    registry = get_default_registry()
    manifest = PipelineLoader(registry).load(
        Path(__file__).resolve().parent / "pipelines" / "correction.yaml"
    )

    def context_factory(state: dict) -> SkillContext:
        deadline = float(state.get("deadline_at", 0))
        if deadline:
            deadline_at = datetime.fromtimestamp(deadline, tz=timezone.utc)
        else:
            deadline_at = datetime.now(timezone.utc) + timedelta(seconds=8)
        return SkillContext(
            request_id=str(state.get("trace_id", "request")),
            trace_id=str(state.get("trace_id", "trace")),
            session_id=str(state.get("session_id", "anonymous")),
            language=str(state.get("response_language", "zh")),
            deadline_at=deadline_at,
        )

    return LangGraphPipelineAdapter(SkillExecutor(registry)).compile(manifest, context_factory)


def build_graph():
    workflow = StateGraph(AgentState)
    for name, node in {
        "retrieve_memory": retrieve_memory_node,
        "parse_question": parse_question_node,
        "rewrite_query": rewrite_query_node,
        "classify_knowledge": classify_knowledge_node,
        "react_agent": react_agent_node,
        "retrieve_documents": retrieve_documents_node,
        "llm_rerank": rerank_documents_node,
        "generate_response": generate_response_node,
        "validate_answer": validate_answer_node,
        "prepare_retry": prepare_retry_node,
        "clarify": clarification_response_node,
        "no_evidence": no_evidence_response_node,
        "validation_failure": validation_failure_response_node,
        "finalize": consolidate_memory_node,
    }.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "parse_question")
    workflow.add_edge("parse_question", "rewrite_query")
    workflow.add_conditional_edges(
        "rewrite_query",
        lambda state: "clarify" if state.get("needs_clarification") else "classify",
        {"clarify": "clarify", "classify": "classify_knowledge"},
    )
    workflow.add_conditional_edges(
        "classify_knowledge",
        lambda state: "react" if state.get("intent") == "knowledge_query" else "workflow",
        {"react": "react_agent", "workflow": "retrieve_documents"},
    )
    workflow.add_edge("react_agent", "retrieve_documents")
    workflow.add_conditional_edges(
        "retrieve_documents",
        lambda state: "rerank" if state.get("retrieval_candidates") else "no_evidence",
        {"rerank": "llm_rerank", "no_evidence": "no_evidence"},
    )
    workflow.add_conditional_edges(
        "llm_rerank",
        lambda state: "validate" if state.get("documents") and state.get("draft_response") else "generate" if state.get("documents") else "no_evidence",
        {"validate": "validate_answer", "generate": "generate_response", "no_evidence": "no_evidence"},
    )
    workflow.add_edge("generate_response", "validate_answer")

    def after_validation(state: AgentState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        if state.get("validation_passed"):
            return "accepted"
        if state.get("correction_attempts", 0) < MAX_CORRECTION_ATTEMPTS:
            return "retry"
        return "rejected"

    workflow.add_conditional_edges(
        "validate_answer",
        after_validation,
        {"accepted": "finalize", "retry": "prepare_retry", "clarify": "clarify", "rejected": "validation_failure"},
    )
    workflow.add_edge("prepare_retry", "rewrite_query")
    workflow.add_edge("clarify", "finalize")
    workflow.add_edge("no_evidence", "finalize")
    workflow.add_edge("validation_failure", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()
