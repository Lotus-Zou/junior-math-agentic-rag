# -*- coding: utf-8 -*-
"""Controlled LangGraph nodes for parsing, retrieval, rerank, generation, and Critic validation."""

from __future__ import annotations

import json
import time
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from agentic_rag import memory
from agentic_rag.completeness import analyze_completeness
from agentic_rag.chains import (
    generator_llm,
    get_answer_validation_chain,
    get_math_classifier_chain,
    get_memory_consolidation_chain,
    get_query_rewriter_chain,
    get_question_parser_chain,
    get_rerank_chain,
    message_text,
)
from agentic_rag.guardrails import guided_answer_violations, scope_violation
from agentic_rag.knowledge_graph import math_knowledge_graph
from agentic_rag.math_retriever import math_retriever
from agentic_rag.math_taxonomy import classify_math_text
from agentic_rag.math_validation import deterministic_equation_answer, deterministic_math_checks
from agentic_rag.memory_manager import compress_history, history_text, working_memory
from agentic_rag.response_contract import response_validation_digest
from agentic_rag.reliability import FailureKind, resolve_failure
from agentic_rag.state import AgentState
from agentic_rag.tracing import check_budget, event, new_trace, persist_trace
from config import RERANK_TOP_K


def _context_text(documents: Iterable[Document]) -> str:
    return "\n\n".join(
        f"[{index}] chunk_id={doc.metadata.get('chunk_id')}; 来源={doc.metadata.get('source', 'unknown')}; "
        f"知识点={doc.metadata.get('knowledge_points', doc.metadata.get('chapter', ''))}; "
        f"公式={doc.metadata.get('formula_ids', '')}\n{doc.page_content}"
        for index, doc in enumerate(documents or [], start=1)
    )


def _node_start(state: AgentState, node: str) -> tuple[float, dict]:
    return time.time(), check_budget(state, node)


def _node_result(state: AgentState, node: str, started: float, budget: dict, updates: dict, trace_payload=None) -> dict:
    traced_state = {**state, **budget, **updates}
    return {**updates, **budget, **event(traced_state, node, "completed", trace_payload if trace_payload is not None else updates, started)}


def _actionable_missing_conditions(intent: str, missing: list[str]) -> list[str]:
    if intent != "knowledge_query":
        return missing
    referential_gaps = ("指代", "所指", "哪个定理", "哪一个定理", "哪个概念", "哪一个概念", "问题主体", "具体知识点")
    return [item for item in missing if any(term in item for term in referential_gaps)]


def _usable_react_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    incomplete_markers = ("need more steps", "more steps to process", "需要更多步骤", "还需要更多步骤")
    return len(normalized) >= 80 and not any(marker in normalized for marker in incomplete_markers)


def _student_attempt_from_query(query: str) -> str:
    for marker in ("学生错误作答：", "学生错误作答:", "错误步骤是", "我写成", "我移项后写成"):
        if marker in query:
            return query.split(marker, 1)[1].strip()
    return query if any(marker in query for marker in ("错在哪里", "第一处错误", "为什么错")) else ""


def retrieve_memory_node(state: AgentState) -> dict:
    runtime = new_trace(state["query"]) if not state.get("trace_id") else {}
    base = {**state, **runtime}
    started, budget = _node_start(base, "retrieve_memory")
    summary, retained_history = compress_history(list(base.get("conversation_history", [])), base.get("conversation_summary", ""))
    try:
        recalled = memory.retrieve_memories(base["query"])
        recalled_text = "\n".join(item["text"] for item in recalled) or "无相关长期薄弱知识记录。"
    except Exception as exc:
        recalled_text = f"长期记忆不可用: {exc}"
    updates = {
        **runtime,
        "conversation_summary": summary,
        "conversation_history": retained_history,
        "retrieved_memories": recalled_text,
        "correction_attempts": base.get("correction_attempts", 0),
        "validation_issues": list(base.get("validation_issues", [])),
    }
    return _node_result(base, "retrieve_memory", started, budget, updates, {"long_term_memory": bool(recalled_text), "history_retained": len(retained_history)})


def parse_question_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "parse_question")
    query = state["query"]
    fast_path = state.get("correction_attempts", 0) == 0 and bool(deterministic_equation_answer(query))
    if fast_path:
        student_answer = _student_attempt_from_query(query)
        result = {
            "stem": query,
            "student_answer": student_answer,
            "intent": "error_analysis" if student_answer else "solve",
            "error_clues": ["一元一次方程变形或计算错误"] if student_answer else [],
        }
    else:
        try:
            result = get_question_parser_chain().invoke({"query": query, "conversation_summary": state.get("conversation_summary", "无")})
        except Exception:
            intent = "error_analysis" if any(word in query for word in ("错因", "为什么错", "哪里错")) else "knowledge_query" if any(word in query for word in ("什么是", "定义", "知识点")) else "solve"
            result = {"stem": query, "student_answer": "", "intent": intent, "error_clues": []}
    result["fast_path"] = fast_path
    return _node_result(state, "parse_question", started, budget, result, {"intent": result["intent"], "has_student_answer": bool(result.get("student_answer")), "fast_path": fast_path})


def analyze_completeness_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "analyze_completeness")
    result = analyze_completeness(
        state.get("stem") or state["query"],
        state.get("response_language", "zh"),
        has_image=bool(state.get("has_image", False)),
    )
    needs_clarification = result.status != "complete"
    updates = {
        "completeness_status": result.status,
        "missing_conditions": list(result.missing),
        "needs_clarification": needs_clarification,
        "follow_up_question": result.follow_up,
    }
    return _node_result(
        state,
        "analyze_completeness",
        started,
        budget,
        updates,
        {"status": result.status, "missing": result.missing},
    )


def rewrite_query_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "rewrite_query")
    if state.get("fast_path") and state.get("correction_attempts", 0) == 0:
        query = state.get("stem") or state["query"]
        result = {
            "rewritten_query": query,
            "sub_queries": ["一元一次方程 等式基本性质 移项变号", "一元一次方程 分步求解 代入验算"],
            "known_conditions": ["单变量一次方程"],
            "missing_conditions": [],
        }
    else:
        try:
            result = get_query_rewriter_chain().invoke({
                "query": state.get("stem") or state["query"],
                "student_answer": state.get("student_answer", ""),
                "history": history_text(state.get("conversation_history", []), limit=8),
                "validation_issues": "；".join(state.get("validation_issues", [])) or "无",
            })
        except Exception:
            query = state.get("stem") or state["query"]
            result = {"rewritten_query": query, "sub_queries": [query], "known_conditions": [], "missing_conditions": []}
    missing = _actionable_missing_conditions(state.get("intent", "solve"), result.get("missing_conditions", []))
    sub_queries = list(dict.fromkeys([result["rewritten_query"], *result.get("sub_queries", [])]))[:4]
    updates = {
        "updated_query": result["rewritten_query"],
        "sub_queries": sub_queries,
        "query_facts": result.get("known_conditions", []),
        "missing_conditions": missing,
        "needs_clarification": bool(missing),
        "follow_up_question": f"请补充以下题设信息：{'；'.join(missing)}" if missing else "",
    }
    return _node_result(state, "rewrite_query", started, budget, updates, {"sub_queries": sub_queries, "missing": missing})


def classify_knowledge_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "classify_knowledge")
    query = state.get("updated_query") or state["query"]
    if state.get("fast_path") and state.get("correction_attempts", 0) == 0:
        result = {
            "chapter": "代数",
            "grade": "七年级",
            "knowledge_points": ["一元一次方程", "等式的基本性质", "移项与验算"],
            "question_type": "计算题",
        }
    else:
        try:
            result = get_math_classifier_chain().invoke({"query": query})
        except Exception:
            fallback = classify_math_text(query)
            result = {"chapter": fallback.chapter, "grade": fallback.grade, "knowledge_points": fallback.knowledge_points, "question_type": fallback.question_type}
    result["graph_context"] = math_knowledge_graph.context(result.get("knowledge_points", []))
    return _node_result(state, "classify_knowledge", started, budget, result, {"chapter": result["chapter"], "knowledge_points": result["knowledge_points"]})


def react_agent_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "react_agent")
    try:
        from agentic_rag.react_agent import run_react_agent
        result = run_react_agent(
            state.get("updated_query") or state["query"],
            state.get("conversation_summary", ""),
            state.get("response_language", "zh"),
        )
        answer = result["answer"] if _usable_react_answer(result["answer"]) else ""
        updates = {
            "draft_response": answer,
            "tool_calls": [*state.get("tool_calls", []), *result.get("tool_calls", [])],
            "metrics": {
                **state.get("metrics", {}),
                "tool_calls": len(result.get("tool_calls", [])),
                **({"generation_mode": "react_llm"} if answer else {}),
            },
        }
    except Exception as exc:
        updates = {"draft_response": "", "error": f"ReAct 降级为固定 Workflow: {exc}"}
    return _node_result(state, "react_agent", started, budget, updates, {"tool_calls": len(updates.get("tool_calls", [])), "fallback": not bool(updates.get("draft_response"))})


def retrieve_documents_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "retrieve_documents")
    try:
        candidates, trace = math_retriever.retrieve_candidates(
            state.get("sub_queries") or [state.get("updated_query") or state["query"]],
            state.get("chapter", "综合"),
            state.get("knowledge_points", []),
        )
        updates = {"retrieval_candidates": candidates, "retrieval_trace": trace, "error": state.get("error")}
    except Exception:
        updates = {
            "retrieval_candidates": [],
            "retrieval_trace": [],
            "error": "retrieval_failed",
            "internal_failure_kind": "runtime_error",
        }
    return _node_result(state, "retrieve_documents", started, budget, updates, updates.get("retrieval_trace"))


def rerank_documents_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "llm_rerank")
    candidates = state.get("retrieval_candidates", [])
    if not candidates:
        return _node_result(state, "llm_rerank", started, budget, {"documents": []}, {"returned": 0})
    candidate_text = "\n\n".join(
        f"chunk_id={doc.metadata.get('chunk_id')} | RRF={doc.metadata.get('rrf_score')} | 元数据={json.dumps(doc.metadata, ensure_ascii=False)}\n{doc.page_content}"
        for doc in candidates
    )
    try:
        result = get_rerank_chain().invoke({
            "query": state.get("updated_query") or state["query"],
            "sub_queries": state.get("sub_queries", []),
            "knowledge_points": state.get("knowledge_points", []),
            "graph_context": state.get("graph_context", ""),
            "candidates": candidate_text,
        })
        by_id = {doc.metadata.get("chunk_id"): doc for doc in candidates}
        ranked_ids = [item for item in result.get("ranked_chunk_ids", []) if item in by_id]
        ranked_ids.extend(item for item in by_id if item not in ranked_ids and item not in result.get("rejected_chunk_ids", []))
        documents = [by_id[item] for item in ranked_ids[:RERANK_TOP_K]]
        report = result
        rerank_mode = "llm"
    except Exception as exc:
        documents = candidates[:RERANK_TOP_K]
        report = {"ranked_chunk_ids": [doc.metadata.get("chunk_id") for doc in documents], "rejected_chunk_ids": [], "reason": f"LLM-Rerank 降级为 RRF: {exc}"}
        rerank_mode = "rrf_fallback"
    for rank, document in enumerate(documents, start=1):
        document.metadata.update({"rank": rank, "retrieval_score": document.metadata.get("rrf_score", 0.0)})
    trace = [*state.get("retrieval_trace", []), {"stage": "llm_rerank", "returned": len(documents), "reason": report.get("reason", "")}]
    updates = {
        "documents": documents,
        "rerank_report": report,
        "retrieval_trace": trace,
        "working_memory": working_memory({**state, "documents": documents}),
        "metrics": {**state.get("metrics", {}), "rerank_mode": rerank_mode},
    }
    return _node_result(state, "llm_rerank", started, budget, updates, trace[-1])


def generate_response_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "generate_response")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是初中数学引导式辅导 Agent。只能依据题设与编号教材片段；禁止只给标准答案。必须输出知识点定位、错因分析、解题思路、分步过程、自检五部分，每个关键公式/定理标注 [n]。对学生错误作答要指出第一处错误，并用一句提示引导订正。回答应紧凑，中文通常控制在 350 字以内，避免重复结论。数学表达使用可直接显示的 Unicode 与普通文本，例如 2x + 3 = 11、8/2、×、≠；禁止输出 LaTeX 定界符、\\frac、\\boxed、\\quad。不得越出初中大纲。必须严格使用 {response_language} 回答。"""),
        ("human", "意图: {intent}\n题目: {query}\n学生错误作答: {student_answer}\n教材片段:\n{context}\nGraphRAG:\n{graph_context}\n长期薄弱点:\n{memories}\n会话摘要:\n{summary}"),
    ])
    try:
        message = (prompt | generator_llm).invoke({
            "intent": state.get("intent", "solve"),
            "query": state.get("updated_query") or state["query"],
            "student_answer": state.get("student_answer", "无"),
            "context": _context_text(state.get("documents", [])),
            "graph_context": state.get("graph_context", ""),
            "memories": state.get("retrieved_memories", "无"),
            "summary": state.get("conversation_summary", "无"),
            "response_language": "简体中文" if state.get("response_language", "zh") == "zh" else "English",
        })
        usage = getattr(message, "usage_metadata", None) or message.response_metadata.get("token_usage", {}) if hasattr(message, "response_metadata") else {}
        metrics = {**state.get("metrics", {}), "generation_tokens": usage or {}, "generation_mode": "llm"}
        draft = message_text(message)
    except Exception:
        draft = deterministic_equation_answer(
            state.get("updated_query") or state["query"],
            state.get("student_answer", ""),
            len(state.get("documents", [])),
            state.get("response_language", "zh"),
        )
        if not draft:
            raise RuntimeError("generation_failed")
        usage = {}
        metrics = {**state.get("metrics", {}), "generation_tokens": {}, "generation_mode": "local_sympy"}
    updates = {"draft_response": draft, "metrics": metrics}
    return _node_result(state, "generate_response", started, budget, updates, {"draft_length": len(draft), "usage": usage})


def validate_answer_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "critic_validation")
    answer = state.get("draft_response", "")
    deterministic = deterministic_math_checks(state.get("updated_query") or state["query"], answer, len(state.get("documents", [])))
    guard_issues = guided_answer_violations(answer)
    scope_issue = scope_violation(state.get("chapter", "综合"), answer)
    if scope_issue:
        guard_issues.append(scope_issue)
    try:
        critic = get_answer_validation_chain().invoke({
            "query": state.get("updated_query") or state["query"],
            "student_answer": state.get("student_answer", "无"),
            "context": _context_text(state.get("documents", [])),
            "answer": answer,
            "deterministic_checks": json.dumps(deterministic, ensure_ascii=False),
        })
        critic["validation_mode"] = "llm"
    except Exception:
        local_passed = deterministic.get("passed", False) and bool(state.get("documents"))
        critic = {
            "is_valid": local_passed,
            "factual_faithfulness": bool(state.get("documents")),
            "math_logic_valid": deterministic.get("passed", False),
            "issues": [] if local_passed else ["独立校验未能完成"],
            "hallucination_detected": False,
            "defect_report": "已使用本地数学校验结果，独立语义校验未完成。",
            "validation_mode": "local_sympy",
        }
    issues = list(dict.fromkeys([*deterministic.get("issues", []), *guard_issues, *critic.get("issues", [])]))
    passed = bool(critic.get("is_valid")) and deterministic.get("passed", False) and not guard_issues
    metrics = dict(state.get("metrics", {}))
    metrics["critic_failures"] = metrics.get("critic_failures", 0) + int(not passed)
    metrics["hallucinations_detected"] = metrics.get("hallucinations_detected", 0) + int(bool(critic.get("hallucination_detected")))
    updates = {
        "response": answer if passed else state.get("response", ""),
        "response_type": "verified_answer" if passed else "",
        "validation_passed": passed,
        "validation_issues": issues,
        "critic_report": {
            **critic,
            "deterministic": deterministic,
            "guardrail_issues": guard_issues,
            "validated_response_sha256": response_validation_digest(answer),
        },
        "needs_clarification": bool(critic.get("needs_clarification")) and bool(state.get("missing_conditions")),
        "follow_up_question": critic.get("follow_up_question", ""),
        "metrics": metrics,
    }
    return _node_result(state, "critic_validation", started, budget, updates, {"passed": passed, "issues": issues, "hallucination": critic.get("hallucination_detected", False)})


def prepare_retry_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "prepare_retry")
    updates = {
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "draft_response": "",
        "response": "",
        "response_type": "",
        "validation_passed": False,
        "critic_report": {},
        "needs_clarification": False,
        "clarification": None,
        "retrieval_candidates": [],
        "documents": [],
    }
    return _node_result(state, "prepare_retry", started, budget, updates, {"attempt": updates["correction_attempts"]})


def _failure_updates(
    state: AgentState,
    failure_kind: FailureKind,
    issues: list[str],
) -> dict:
    resolved = resolve_failure(
        query=state["query"],
        language=state.get("response_language", "zh"),
        history=list(state.get("conversation_history", [])),
        summary=state.get("conversation_summary", ""),
        failure_kind=failure_kind,
        issues=issues,
        verified_partial=state.get("verified_partial") or None,
    )
    policy_metrics = resolved.get("metrics", {})
    return {
        "draft_response": "",
        "response": resolved["answer"],
        "response_type": resolved["response_type"],
        "conversation_history": resolved["conversation_history"],
        "conversation_summary": resolved["conversation_summary"],
        "validation_passed": resolved["validation_passed"],
        "critic_report": {},
        "needs_clarification": resolved["response_type"] == "clarification_required",
        "clarification": resolved["clarification"],
        "internal_failure_kind": failure_kind,
        "metrics": {**state.get("metrics", {}), **policy_metrics},
    }


def clarification_response_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "clarify")
    updates = _failure_updates(state, "incomplete_input", list(state.get("missing_conditions", [])))
    if state.get("follow_up_question"):
        message = state["follow_up_question"]
        updates["response"] = message
        updates["conversation_history"][-1] = {"role": "tutor", "content": message}
    return _node_result(state, "clarify", started, budget, updates, {"missing": updates["clarification"]["missing"]})


def no_evidence_response_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "no_evidence")
    updates = _failure_updates(state, "retrieval_empty", list(state.get("validation_issues", [])))
    return _node_result(state, "no_evidence", started, budget, updates, {"failure_kind": "retrieval_empty"})


def validation_failure_response_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "validation_failure")
    issues = list(state.get("validation_issues", []))
    updates = _failure_updates(state, "critic_rejected", issues)
    return _node_result(state, "validation_failure", started, budget, updates, {"failure_kind": "critic_rejected", "issue_count": len(issues)})


def consolidate_memory_node(state: AgentState) -> dict:
    started, budget = _node_start(state, "finalize")
    history = list(state.get("conversation_history", []))
    if state.get("response") and (not history or history[-1].get("content") != state["response"]):
        history.extend([{"role": "student", "content": state["query"]}, {"role": "tutor", "content": state["response"]}])
    if state.get("validation_passed") and not state.get("fast_path"):
        try:
            result = get_memory_consolidation_chain().invoke({"conversation_history": history_text(history, limit=12)})
            if result.get("text") and "No valuable information" not in result["text"]:
                memory.add_memory(result["text"], result.get("type", "knowledge_gap"), result.get("importance", 5))
        except Exception:
            pass
    final_state = {**state, **budget, "conversation_history": history}
    trace_update = event(final_state, "finalize", "completed", {"metrics": state.get("metrics", {}), "validation_passed": state.get("validation_passed")}, started)
    final_state.update(trace_update)
    persist_trace(final_state)
    return {**budget, "conversation_history": history, **trace_update}
