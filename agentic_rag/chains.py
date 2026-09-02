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
    ATTACHMENT_LLM_CALL_TIMEOUT_SECONDS,
    EXERCISE_LLM_CALL_TIMEOUT_SECONDS,
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


def _build_llm(
    model: str,
    *,
    timeout: float = LLM_CALL_TIMEOUT_SECONDS,
    max_retries: int = LLM_MAX_RETRIES,
):
    params = {"model": model, "timeout": timeout, "max_retries": max_retries}
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
attachment_llm = _build_llm(
    LLM_MODEL_NAME, timeout=ATTACHMENT_LLM_CALL_TIMEOUT_SECONDS
)
exercise_llm = _build_llm(
    LLM_MODEL_NAME, timeout=EXERCISE_LLM_CALL_TIMEOUT_SECONDS
)
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


class FirstErrorJudgeResult(BaseModel):
    first_error_step: int = Field(ge=1, description="最早实质数学错误的步骤编号")
    explanation: str = Field(
        min_length=1,
        description="简短、可核验的错误说明，不包含隐藏思维过程",
    )


class FinalAnswerJudgeResult(BaseModel):
    is_complete: bool = Field(
        description="题面是否包含独立求解所需的全部条件"
    )
    math_logic_valid: bool = Field(
        description="corrected_answer 的推导与最终结论是否正确"
    )
    corrected_answer: str = Field(
        default="",
        description="题面完整时给出独立重算答案；题面不完整时具体解释歧义并给出条件化推导",
    )
    issues: List[str] = Field(default_factory=list)


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
        """你是与生成 Agent 完全隔离的 Critic。分别执行：1) 事实忠实度校验，核对每个公式/定理与引用片段；2) 数学逻辑校验，核对推导、符号、单位、公式条件、跳步和最终结论。不能因文风流畅而通过。普通产品回答若直接只给答案、越出初中大纲或引用不存在时必须失败，并输出可复现缺陷报告。对于题干和 A-D 选项已经完整给出的自包含选择题，应逐项独立验算；知识库没有逐字覆盖不能单独作为数学逻辑失败的理由，但错误或无关引用仍应标记 factual_faithfulness=false。对于要求数值结果的完整课堂应用题，如果草稿明确声明了常见教材约定（例如按每月 4 周估算），并且在该约定下逐步计算正确，则应判定 math_logic_valid=true；可以在 issues 中保留现实口径差异，但不得仅因还存在另一种现实换算口径而要求补题或否定答案。未声明约定、约定与题意冲突或计算错误时仍必须失败。课堂借款题若只写借款 N 个月、利率 r%，且没有 annual、per year、monthly、per month 等周期限定，则按整个所述借款期一次性简单利率 r% 计算；草稿明确这一约定且计算正确时不得自行改成年利率后拒绝。题面明确年利率或月利率时必须按明确周期换算。严格区分“达到阈值”和“首次超过阈值”：累计收益刚好等于成本只是回本，不是开始盈利；当周期数的商恰为整数且题目问何时开始盈利、超过成本或严格大于阈值时，答案必须进入下一个完整周期。若题目要求定位 First error，它是封闭题内的数学审查：不得仅因知识点超出初中大纲或没有外部检索证据而拒绝；必须从第一步开始依次复算，只接受最早一个会改变推导有效性或结果的实质数学错误。不影响数学含义和结果的术语不精确、文风、冗余说明以及仍在误差容限内的中间近似不能作为首错。若某一步明确使用约等号，且按该步已经显示的舍入数值计算一致，不得用更高精度重算后把舍入差当作该步错误；普通等号则声称精确相等，若舍入小数并不精确满足等式，该步就是错误。若原题要求精确值，后续把近似小数作为没有约等号的最终精确答案时，首错才落在该最终答案步骤。""",
        "题目: {query}\n学生错误作答: {student_answer}\n教材依据:\n{context}\n\n待验证草稿:\n{answer}\n\n确定性校验结果:\n{deterministic_checks}",
        critic_llm,
    )


def get_first_error_judge_chain():
    return _json_chain(
        FirstErrorJudgeResult,
        """你是生成 Agent 之外的最终数学裁决 Critic。输入包含完整题目、编号步骤、候选草稿和前两轮 Critic 缺陷。你必须独立从 Step 1 开始逐步复算，不得沿用候选结论；返回最早一个会改变推导有效性或结果的实质错误。忽略无害术语、文风和冗余说明。明确使用约等号且按已显示舍入值计算一致的中间近似不是错误；普通等号声称精确相等，舍入小数不精确满足等式时就是错误。原题要求精确值时，把近似小数作为无约等号的最终答案也是错误。不得因知识点超纲或没有外部检索证据拒绝封闭题内审查。""",
        "完整审查请求:\n{query}\n\n候选草稿:\n{candidate}\n\n已有缺陷:\n{issues}",
        critic_llm,
    )


def get_final_answer_judge_chain():
    return _json_chain(
        FinalAnswerJudgeResult,
        """你是与生成 Agent、修复 Agent 和前两轮 Critic 隔离的最终数学裁决 Agent。你必须忽略候选草稿的结论，从原题条件独立重算，再给出可公开的、步骤可核验的 corrected_answer。不要因没有外部检索片段而拒绝自包含计算题或选择题。严格检查运算、单位、方程、除零、比例与阈值语义；达到成本只表示回本，题目问开始盈利或首次超过时必须使用严格大于。若题面缺少唯一求解所需条件，is_complete=false，但 corrected_answer 不能留空：必须点名缺少的具体条件，证明为什么结果不唯一，并尽可能给出不同假设下的条件化结果；这种回答不要伪造唯一的最终数值行。若能够独立确认，is_complete=true、math_logic_valid=true，并保留用户要求的最终答案格式（例如最后一行 Answer: value 或答案：A/B/C/D）。不得为了匹配候选答案而改变题意。""",
        "完整题目:\n{query}\n\n二次修复后的候选草稿:\n{candidate}\n\n前两轮 Critic 缺陷:\n{issues}",
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
