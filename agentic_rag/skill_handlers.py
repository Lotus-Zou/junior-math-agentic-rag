"""Pure business handlers used by repository-owned Skill packages."""

from __future__ import annotations

import re
from typing import Any

from agentic_rag.domain.schemas import (
    AnswerCheckInput, AnswerCriticInput, AnswerDraftOutput, AnswerEnvelope,
    AnswerGenerateInput, ClassificationOutput, CriticOutput, CurriculumSolveInput,
    CurriculumSolveOutput, FusionInput, GuardOutput, MemoryInput, MemoryOutput,
    QuestionParseOutput, QueryInput, QueryRewriteInput, QueryRewriteOutput,
    RenderInput, RerankInput, RetrievalCandidate, RetrievalInput, RetrievalOutput,
    SimilarExerciseInput, SimilarExerciseOutput,
)
from agentic_rag.fast_path import build_fast_response
from agentic_rag.guardrails import input_guardrail_violation
from agentic_rag.math_taxonomy import classify_math_text
from agentic_rag.math_validation import deterministic_math_checks


def _candidate(doc: Any) -> RetrievalCandidate:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return RetrievalCandidate(
        chunk_id=metadata.get("chunk_id"), content=getattr(doc, "page_content", str(doc)),
        metadata=metadata, score=float(metadata.get("rrf_score", metadata.get("retrieval_score", 0.0)) or 0.0),
        source=str(metadata.get("source", "")),
    )


def input_guard(data: QueryInput, _context) -> GuardOutput:
    reason = input_guardrail_violation(data.query)
    return GuardOutput(
        normalized_query=re.sub(r"\s+", " ", data.query).strip(),
        allowed=not bool(reason),
        reason=reason,
    )


def question_parse(data: QueryInput, _context) -> QuestionParseOutput:
    text, student = data.query.strip(), ""
    for marker in ("学生错误作答：", "学生错误作答:", "我写成", "错误步骤是", "I wrote"):
        if marker in text:
            text, student = text.split(marker, 1)
            break
    intent = "error_analysis" if student else "knowledge_query" if any(word in text for word in ("什么", "怎么画", "定义", "what", "how")) else "solve"
    return QuestionParseOutput(stem=text.strip(), student_answer=student.strip(), intent=intent)


def query_rewrite(data: QueryRewriteInput, _context) -> QueryRewriteOutput:
    query = (data.stem or data.query).strip()
    missing = ["完整题干"] if query in {"这题怎么做", "怎么做", "不会", "how to solve this"} else []
    normalized = re.sub(r"\s+", " ", query).replace("×", "*").replace("÷", "/")
    classification = classify_math_text(normalized)
    return QueryRewriteOutput(
        rewritten_query=normalized,
        sub_queries=list(dict.fromkeys([normalized, *classification.knowledge_points])),
        missing_conditions=missing,
    )


def knowledge_classify(data: QueryInput, _context) -> ClassificationOutput:
    item = classify_math_text(data.query)
    return ClassificationOutput(grade=item.grade, chapter=item.chapter, knowledge_points=item.knowledge_points, question_type=item.question_type)


def curriculum_solve(data: CurriculumSolveInput, _context) -> CurriculumSolveOutput:
    response = build_fast_response(
        data.query,
        data.conversation_history,
        data.conversation_summary,
        data.language,
        exercise_state=(
            data.exercise_state.model_dump(mode="json")
            if data.exercise_state is not None
            else None
        ),
    )
    return CurriculumSolveOutput(handled=response is not None, response=response)


def _retrieve(data: RetrievalInput, strategy: str) -> RetrievalOutput:
    from agentic_rag.math_retriever import math_retriever

    docs, trace = math_retriever.retrieve_candidates(
        data.sub_queries or [data.query], data.chapter, data.knowledge_points,
        candidate_k=data.top_k, strategy=strategy,
    )
    return RetrievalOutput(candidates=[_candidate(doc) for doc in docs[:data.top_k]], trace=trace)


def retrieve_dense(data: RetrievalInput, _context) -> RetrievalOutput:
    from agentic_rag.math_retriever import math_retriever

    docs, trace = math_retriever.retrieve_dense_channel(
        data.sub_queries or [data.query], data.chapter, data.top_k
    )
    return RetrievalOutput(candidates=[_candidate(doc) for doc in docs], trace=trace)


def retrieve_bm25(data: RetrievalInput, _context) -> RetrievalOutput:
    from agentic_rag.math_retriever import math_retriever

    docs, trace = math_retriever.retrieve_bm25_channel(
        data.sub_queries or [data.query], data.chapter, data.top_k
    )
    return RetrievalOutput(candidates=[_candidate(doc) for doc in docs], trace=trace)


def retrieve_graph(data: RetrievalInput, _context) -> RetrievalOutput:
    from agentic_rag.knowledge_graph import math_knowledge_graph
    from agentic_rag.math_retriever import math_retriever

    docs, trace = math_retriever.retrieve_graph_channel(
        data.knowledge_points, data.chapter, data.top_k
    )
    trace.append({"stage": "knowledge_graph_context", "context": math_knowledge_graph.context(data.knowledge_points)})
    return RetrievalOutput(candidates=[_candidate(doc) for doc in docs], trace=trace)


def rrf_fusion(data: FusionInput, _context) -> RetrievalOutput:
    from agentic_rag.math_retriever import fuse_rankings

    by_id: dict[str, RetrievalCandidate] = {}
    rankings: list[list[str]] = []
    for group in data.rankings:
        ids = []
        for item in group:
            key = item.chunk_id or f"content:{hash(item.content)}"
            by_id[key], ids = item, [*ids, key]
        rankings.append(ids)
    scores = fuse_rankings(*rankings)
    candidates = []
    for key in sorted(scores, key=scores.get, reverse=True)[:data.top_k]:
        item = by_id[key].model_copy(deep=True)
        item.score = scores[key]
        candidates.append(item)
    return RetrievalOutput(candidates=candidates, trace=[{"stage": "rrf_fusion", "returned": len(candidates)}])


def rerank_filter(data: RerankInput, _context) -> RetrievalOutput:
    terms = set(re.findall(r"[\w\u4e00-\u9fff]+", data.query.lower()))
    ranked = sorted(data.candidates, key=lambda item: (sum(term in item.content.lower() for term in terms), item.score), reverse=True)
    return RetrievalOutput(candidates=ranked[:data.top_k], trace=[{"stage": "deterministic_rerank", "returned": min(len(ranked), data.top_k)}])


def answer_generate(data: AnswerGenerateInput, _context) -> AnswerDraftOutput:
    local = build_fast_response(data.query, [], "", data.language)
    if local:
        return AnswerDraftOutput(answer=local["answer"], citations=[str(item.get("source", "")) for item in local.get("sources", [])])
    if not data.contexts:
        text = "请补充完整题目或相关教材内容。" if data.language == "zh" else "Please provide the complete problem or relevant source material."
        return AnswerDraftOutput(answer=text)
    excerpt = data.contexts[0].content[:800]
    prefix = "根据检索到的教材内容：" if data.language == "zh" else "Based on the retrieved source: "
    return AnswerDraftOutput(answer=prefix + excerpt, citations=[data.contexts[0].chunk_id or data.contexts[0].source])


def answer_critic(data: AnswerCriticInput, _context) -> CriticOutput:
    checks = deterministic_math_checks(data.query, data.answer, len(data.contexts))
    passed = bool(checks.get("passed"))
    return CriticOutput(
        passed=passed, factual_faithfulness=bool(data.contexts) or passed,
        math_logic_valid=passed, issues=list(checks.get("issues", [])), deterministic=checks,
    )


def similar_exercise(data: SimilarExerciseInput, _context) -> SimilarExerciseOutput:
    from agentic_rag.math_retriever import math_retriever

    classification = classify_math_text(data.query)
    docs, trace = math_retriever.search(
        f"典型例题 巩固练习 {data.query}", classification.chapter,
        data.knowledge_points or classification.knowledge_points, top_k=data.top_k,
    )
    return SimilarExerciseOutput(exercises=[_candidate(doc) for doc in docs], trace=trace)


def answer_check(data: AnswerCheckInput, context) -> CriticOutput:
    critic_input = AnswerCriticInput(
        query=data.query, answer=data.student_answer, contexts=data.contexts,
        student_answer=data.student_answer,
    )
    return answer_critic(critic_input, context)


def memory_recall(_data: MemoryInput, _context) -> MemoryOutput:
    return MemoryOutput(summary="")


def memory_commit(data: MemoryInput, _context) -> MemoryOutput:
    return MemoryOutput(events=[{"session_id": data.session_id, "knowledge_points": data.knowledge_points}])


def response_render(data: RenderInput, _context) -> AnswerEnvelope:
    payload = data.model_dump(exclude={"response_type", "language"})
    response_type = data.response_type
    if data.validation_passed is not True:
        response_type = "clarification_required"
        payload.update(
            answer=(
                "I could not verify that draft. Please share the full problem or the step you want checked, and I will help you work through it."
                if data.language == "en"
                else "这份草稿未通过核对。请补充完整题目或需要检查的步骤，我会帮你继续推导。"
            ),
            intent="clarification",
            knowledge_points=[],
            sources=[],
            conversation_history=[],
            conversation_summary="",
            exercise_state=None,
            cached=False,
            clarification={
                "missing": [
                    "the full problem or the step to check"
                    if data.language == "en"
                    else "完整题目或需要检查的步骤"
                ]
            },
        )
    return AnswerEnvelope(
        **payload,
        response_type=response_type,
        language=data.language,
    )
