# -*- coding: utf-8 -*-
"""Runtime configuration for the mathematics Agentic RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


PROJECT_ROOT = Path(__file__).resolve().parent


def _knowledge_root() -> Path:
    configured = os.getenv("KNOWLEDGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (PROJECT_ROOT, *PROJECT_ROOT.parents)
    for candidate in candidates:
        if (candidate / "data" / "初中数学核心知识.md").is_file():
            return candidate
    return PROJECT_ROOT


KNOWLEDGE_ROOT = _knowledge_root()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_WIRE_API = os.getenv("MODEL_WIRE_API", "chat_completions")
MODEL_REASONING_EFFORT = os.getenv("MODEL_REASONING_EFFORT", "low")
DISABLE_RESPONSE_STORAGE = os.getenv("DISABLE_RESPONSE_STORAGE", "false").lower() in {"1", "true", "yes"}
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-plus")
CRITIC_MODEL_NAME = os.getenv("CRITIC_MODEL_NAME", LLM_MODEL_NAME)
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", LLM_MODEL_NAME)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or None

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE") or None
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
MATH_EMBEDDING_MODEL_PATH = os.getenv("MATH_EMBEDDING_MODEL_PATH", "BAAI/bge-m3")
LOCAL_EMBEDDING_MODEL_PATH = MATH_EMBEDDING_MODEL_PATH
ENABLE_DENSE_RETRIEVAL = os.getenv("ENABLE_DENSE_RETRIEVAL", "false").lower() in {"1", "true", "yes"}
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_BASE = os.getenv("TAVILY_API_BASE", "https://api.tavily.com/search")
TAVILY_MAX_RESULTS = max(1, min(5, int(os.getenv("TAVILY_MAX_RESULTS", "5"))))
TAVILY_TIMEOUT_SECONDS = max(
    2.0, min(20.0, float(os.getenv("TAVILY_TIMEOUT_SECONDS", "10")))
)
TAVILY_INCLUDE_DOMAINS = tuple(
    item.strip()
    for item in os.getenv("TAVILY_INCLUDE_DOMAINS", "").split(",")
    if item.strip()
)

CHROMA_PATH = str(Path(os.getenv("CHROMA_PATH", KNOWLEDGE_ROOT / "chroma_db")).resolve())
KNOWLEDGE_DATA_PATH = str(Path(os.getenv("KNOWLEDGE_DATA_PATH", KNOWLEDGE_ROOT / "data")).resolve())
KNOWLEDGE_SOURCE_PATH = str(Path(KNOWLEDGE_DATA_PATH) / "初中数学核心知识.md")
KNOWLEDGE_GRAPH_PATH = str(Path(os.getenv("KNOWLEDGE_GRAPH_PATH", Path(KNOWLEDGE_DATA_PATH) / "knowledge_graph.json")).resolve())
TRACE_PATH = os.getenv("TRACE_PATH", "traces")
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "24"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", str(RETRIEVAL_TOP_K)))
MAX_CORRECTION_ATTEMPTS = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "1"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "24"))
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "8"))
RUN_TIMEOUT_SECONDS = max(240.0, float(os.getenv("RUN_TIMEOUT_SECONDS", "240")))
LLM_CALL_TIMEOUT_SECONDS = max(
    45.0, float(os.getenv("LLM_CALL_TIMEOUT_SECONDS", "45"))
)
EXERCISE_LLM_CALL_TIMEOUT_SECONDS = max(
    45.0, float(os.getenv("EXERCISE_LLM_CALL_TIMEOUT_SECONDS", "45"))
)
ATTACHMENT_LLM_CALL_TIMEOUT_SECONDS = max(
    45.0, float(os.getenv("ATTACHMENT_LLM_CALL_TIMEOUT_SECONDS", "45"))
)
LLM_MAX_RETRIES = max(0, min(3, int(os.getenv("LLM_MAX_RETRIES", "2"))))
ENABLE_EXERCISE_AGENT = os.getenv("ENABLE_EXERCISE_AGENT", "true").lower() in {
    "1", "true", "yes"
}
ENABLE_TUTOR_AGENT = os.getenv("ENABLE_TUTOR_AGENT", "true").lower() in {
    "1", "true", "yes"
}
FORCE_LLM_EVERY_TURN = os.getenv("FORCE_LLM_EVERY_TURN", "true").lower() in {
    "1", "true", "yes"
}
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "")
OPERATIONS_METRICS_TOKEN = os.getenv("OPERATIONS_METRICS_TOKEN", "")

QUALITY_THRESHOLDS = {
    "context_precision": float(os.getenv("MIN_CONTEXT_PRECISION", "0.65")),
    "context_recall": float(os.getenv("MIN_CONTEXT_RECALL", "0.70")),
    "faithfulness": float(os.getenv("MIN_FAITHFULNESS", "0.75")),
    "answer_relevance": float(os.getenv("MIN_ANSWER_RELEVANCE", "0.70")),
    "knowledge_point_accuracy": float(os.getenv("MIN_KNOWLEDGE_POINT_ACCURACY", "0.80")),
    "direct_answer_violation_rate": float(os.getenv("MAX_DIRECT_ANSWER_VIOLATION_RATE", "0.05")),
}

TOOL_WHITELIST = (
    "question_parse_skill",
    "math_retrieval_skill",
    "similar_exercise_skill",
    "answer_verify_skill",
    "web_tavily_search",
)
EXCEL_METADATA_COLUMNS = ["年级", "章节", "知识点", "题型", "错误分类", "前置知识点", "来源"]
