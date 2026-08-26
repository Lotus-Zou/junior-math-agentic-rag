# -*- coding: utf-8 -*-
"""Explicit LangGraph state for the controlled mathematics Agent runtime."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    response_language: str
    stem: str
    student_answer: str
    intent: str
    error_clues: List[str]
    fast_path: bool
    updated_query: str
    sub_queries: List[str]
    query_facts: List[str]
    missing_conditions: List[str]
    chapter: str
    grade: str
    knowledge_points: List[str]
    question_type: str

    retrieval_candidates: List[Any]
    documents: List[Any]
    retrieval_trace: List[Dict[str, Any]]
    rerank_report: Dict[str, Any]
    graph_context: str

    draft_response: str
    response: str
    validation_passed: bool
    validation_issues: List[str]
    critic_report: Dict[str, Any]
    needs_clarification: bool
    follow_up_question: str
    correction_attempts: int

    working_memory: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    conversation_summary: str
    retrieved_memories: str

    trace_id: str
    trace_events: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    started_at: float
    deadline_at: float
    step_count: int
    tool_calls: List[Dict[str, Any]]
    error: Optional[str]
