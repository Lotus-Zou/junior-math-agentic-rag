# -*- coding: utf-8 -*-
"""Bounded ReAct runtime with Registry-discovered retrieval tools."""

from __future__ import annotations

from agentic_rag.chains import generator_llm, message_text
from agentic_rag.skill_runtime.adapters.langchain_tools import LangChainToolAdapter
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import get_default_registry
from config import MAX_REACT_STEPS

_registry = get_default_registry()
_tool_adapter = LangChainToolAdapter(_registry, SkillExecutor(_registry))
REACT_SKILLS = [
    _tool_adapter.build(manifest.ref)
    for manifest in _registry.list(capability="retrieval")
    if manifest.expose.langchain
]


def run_react_agent(query: str, conversation_summary: str = "", response_language: str = "zh") -> dict:
    from langgraph.prebuilt import create_react_agent

    answer_language = "简体中文" if response_language == "zh" else "English"
    prompt = (
        "你是初中数学辅导 Tool-Calling Agent。只使用 Registry 授予的 retrieval 工具，最多调用两个工具；"
        "工具只接受一个 payload 对象，字段必须符合其 Schema。得到依据后立即作答，不重复检索。"
        "不得直接给无步骤标准答案。最终回答包含知识点定位、解题思路、分步过程、自检四部分，"
        "并使用 [1] 标注检索来源。使用 Unicode 与普通文本表达数学内容，不输出 LaTeX 定界符。"
        f"仅处理初中大纲，不展示隐藏思维过程，必须使用{answer_language}回答。"
    )
    agent = create_react_agent(generator_llm, REACT_SKILLS, prompt=prompt)
    result = agent.invoke(
        {"messages": [("user", f"会话摘要:\n{conversation_summary or '无'}\n\n问题:\n{query}")]},
        config={"recursion_limit": MAX_REACT_STEPS},
    )
    messages = result.get("messages", [])
    final = message_text(messages[-1]) if messages else ""
    tool_calls = []
    for message in messages:
        if getattr(message, "tool_calls", None):
            tool_calls.extend(message.tool_calls)
    return {"answer": final, "tool_calls": tool_calls}
