# -*- coding: utf-8 -*-
"""CLI entry point for the junior-high mathematics mistake tutor."""

import sys

from agentic_rag import memory
from agentic_rag.graph import build_graph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def handle_memory_commands(query: str) -> bool:
    if query.strip() == "!show_memories":
        memories = memory.view_memories(limit=10)
        if not memories:
            print("暂无学习记录。")
        for item in memories:
            print(f"[ID {item['id']}] {item['type']}: {item['text']}")
        return True
    if query.startswith("!forget"):
        topic = query.removeprefix("!forget").strip()
        if not topic:
            print("用法: !forget 要删除的知识点或错因")
            return True
        candidates = memory.retrieve_memories(topic, top_k=5)
        for item in candidates:
            memory.delete_memory(item["id"])
        print(f"已删除 {len(candidates)} 条相关学习记录。")
        return True
    return False


def main():
    memory.initialize_memory_db()
    graph = build_graph()
    conversation_history = []
    print("初中数学错题智能问答系统")
    print("输入完整题目开始订正；输入 !new 开始新对话，输入 exit 退出。")
    while True:
        query = input("\n题目或追问: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        if query == "!new":
            conversation_history = []
            print("已开始新对话。")
            continue
        if handle_memory_commands(query):
            continue
        final_state = graph.invoke({"query": query, "conversation_history": conversation_history, "correction_attempts": 0, "validation_issues": []}, config={"recursion_limit": 40})
        conversation_history = final_state.get("conversation_history", conversation_history)
        print("\n" + final_state.get("response", "系统未生成答案。"))
        if final_state.get("chapter"):
            print(f"\n[知识点] {final_state['chapter']} / {'、'.join(final_state.get('knowledge_points', []))}")
        if final_state.get("retrieval_trace"):
            print(f"[检索] {final_state['retrieval_trace']}")
        if final_state.get("validation_passed"):
            print("[验证] 通过")


if __name__ == "__main__":
    main()