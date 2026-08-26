# -*- coding: utf-8 -*-
"""Structured LLM chains for the mathematics Agent runtime."""

from typing import List

import torch
from chromadb.utils import embedding_functions
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from config import (
    CRITIC_MODEL_NAME,
    EMBEDDING_API_BASE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
    LLM_MODEL_NAME,
    LOCAL_EMBEDDING_MODEL_PATH,
    MODEL_REASONING_EFFORT,
    MODEL_WIRE_API,
    OPENAI_API_BASE,
    RERANK_MODEL_NAME,
    DISABLE_RESPONSE_STORAGE,
    LLM_CALL_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
)


def _build_llm(model: str):
    params = {"model": model, "timeout": LLM_CALL_TIMEOUT_SECONDS, "max_retries": LLM_MAX_RETRIES}
    if OPENAI_API_BASE:
        params["base_url"] = OPENAI_API_BASE
    if MODEL_WIRE_API == "responses":
        params["use_responses_api"] = True
        params["store"] = not DISABLE_RESPONSE_STORAGE
        if MODEL_REASONING_EFFORT:
            params["reasoning"] = {"effort": MODEL_REASONING_EFFORT}
    else:
        params["temperature"] = 0
    return ChatOpenAI(**params)


generator_llm = _build_llm(LLM_MODEL_NAME)
rerank_llm = _build_llm(RERANK_MODEL_NAME)
critic_llm = _build_llm(CRITIC_MODEL_NAME)
llm = generator_llm


def message_text(message) -> str:
    """Normalize Chat Completions and Responses API messages to plain text."""
    if isinstance(message, str):
        return message
    text = getattr(message, "text", None)
    if text is not None:
        return str(text)
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") in {None, "text", "output_text"}
        ).strip()
    return str(content or "")


def get_embedding_function():
    if EMBEDDING_PROVIDER == "openai":
        params = {"model": EMBEDDING_MODEL_NAME}
        api_base = EMBEDDING_API_BASE or OPENAI_API_BASE
        if api_base:
            params["base_url"] = api_base
        return OpenAIEmbeddings(**params)
    if EMBEDDING_PROVIDER == "local":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=LOCAL_EMBEDDING_MODEL_PATH, device=device)
    raise ValueError(f"未知的嵌入模型提供商: {EMBEDDING_PROVIDER}")


class QuestionParseResult(BaseModel):
    stem: str = Field(description="完整题干")
    student_answer: str = Field(default="", description="学生原错误作答，没有则为空")
    intent: str = Field(description="solve、error_analysis、knowledge_query 之一")
    error_clues: List[str] = Field(default_factory=list, description="显式可见的错因线索")


class RewriteResult(BaseModel):
    rewritten_query: str = Field(description="保留全部公式、数字、单位和求解目标的规范数学表述")
    sub_queries: List[str] = Field(description="2-4 条分别面向定义、公式、步骤或前置知识的检索 Query")
    known_conditions: List[str] = Field(default_factory=list)
    missing_conditions: List[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    chapter: str = Field(description="代数、几何、函数、统计与概率、综合之一")
    grade: str = Field(description="七年级、八年级、九年级或初中")
    knowledge_points: List[str] = Field(description="1-4 个具体知识点")
    question_type: str = Field(description="计算题、证明题、图像题、概念题或应用题")


class RerankResult(BaseModel):
    ranked_chunk_ids: List[str] = Field(description="按支撑力从高到低排列的 chunk_id，最多 6 个")
    rejected_chunk_ids: List[str] = Field(default_factory=list, description="噪声或不相关候选")
    reason: str = Field(description="简短说明重排依据，不输出隐式思维过程")


class ValidationResult(BaseModel):
    is_valid: bool
    factual_faithfulness: bool = Field(description="所有关键陈述均有教材片段支持")
    math_logic_valid: bool = Field(description="推导、公式、符号、单位和结论正确")
    issues: List[str] = Field(default_factory=list)
    hallucination_detected: bool = False
    needs_clarification: bool = False
    follow_up_question: str = ""
    defect_report: str = Field(default="", description="可复现的缺陷报告")


class MetadataExtractionResult(BaseModel):
    grade: str = "初中"
    chapter: str = "综合"
    knowledge_points: List[str] = Field(default_factory=list)
    question_type: str = "概念题"
    error_class: str = "计算或概念错误"
    formulas: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)


class MemoryToSave(BaseModel):
    text: str
    type: str = Field(description="mistake_pattern、knowledge_gap 或 preference")
    importance: int = Field(ge=1, le=10)


def _json_chain(schema, system: str, human: str, model=generator_llm):
    parser = JsonOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages([("system", system + "\n{format_instructions}"), ("human", human)]).partial(format_instructions=parser.get_format_instructions())
    return prompt | model | parser


def get_question_parser_chain():
    return _json_chain(
        QuestionParseResult,
        "你是初中数学错题解析 Agent。只提取明确给出的信息，不补造学生答案。intent 必须是 solve、error_analysis、knowledge_query 之一。",
        "用户输入:\n{query}\n\n会话摘要:\n{conversation_summary}",
    )


def get_query_rewriter_chain():
    return _json_chain(
        RewriteResult,
        """你是数学查询改写与分解 Agent。把口语问题改为规范数学问题，再生成 2-4 条互补子 Query，分别覆盖定义/定理、公式适用条件、求解步骤和前置知识。保留 LaTeX、数字、单位与图形关系，不得臆造条件。教材版本、年级册次、可由系统选择的示例以及应由知识库提供的定理依据都不是用户必须补充的条件；只有题目主体、代词指代、关键数值或图形条件确实缺失时才能填写 missing_conditions。""",
        "对话历史:\n{history}\n\n题干: {query}\n学生错误作答: {student_answer}\n上轮验证缺陷: {validation_issues}",
    )


def get_math_classifier_chain():
    return _json_chain(
        ClassificationResult,
        "chapter 只能是代数、几何、函数、统计与概率、综合；知识点应具体到一元二次方程、三角形全等等层级。",
        "规范题目: {query}",
    )


def get_rerank_chain():
    return _json_chain(
        RerankResult,
        """你是独立 LLM-Rerank 模型。根据题目、子 Query 和知识点，对候选教材片段按“能否直接支撑正确解题步骤”排序。公式条件不匹配、年级错误或仅词面相似的片段必须拒绝。只能返回真实 chunk_id。""",
        "题目: {query}\n子 Query: {sub_queries}\n知识点: {knowledge_points}\nGraphRAG 关系: {graph_context}\n\n候选片段:\n{candidates}",
        rerank_llm,
    )


def get_answer_validation_chain():
    return _json_chain(
        ValidationResult,
        """你是与生成 Agent 完全隔离的 Critic。分别执行：1) 事实忠实度校验，核对每个公式/定理与引用片段；2) 数学逻辑校验，核对推导、符号、单位、公式条件、跳步和最终结论。不能因文风流畅而通过。回答直接只给答案、越出初中大纲或引用不存在时必须失败，并输出可复现缺陷报告。""",
        "题目: {query}\n学生错误作答: {student_answer}\n教材依据:\n{context}\n\n待验证草稿:\n{answer}\n\n确定性校验结果:\n{deterministic_checks}",
        critic_llm,
    )


def get_metadata_extraction_chain():
    return _json_chain(
        MetadataExtractionResult,
        "抽取初中数学资料的年级、章节、知识点、题型、错误分类、完整公式和前置知识。不得改写公式。",
        "资料片段:\n{text}",
    )


def get_summarizer_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "压缩会话但保留当前题干、学生错误步骤、未解决问题、已确认知识点和验证缺陷。"),
        ("human", "历史:\n{conversation_history}"),
    ])
    return prompt | generator_llm


def get_memory_consolidation_chain():
    return _json_chain(
        MemoryToSave,
        "只提取对以后辅导有用的稳定错因、薄弱知识点或讲解偏好。没有则 text 输出 No valuable information to save。",
        "对话历史:\n{conversation_history}",
    )
