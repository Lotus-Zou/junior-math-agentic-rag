# -*- coding: utf-8 -*-
"""Formula-aware ingestion for textbook, knowledge-point, mistake, and exercise data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import chromadb
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from agentic_rag.chains import get_embedding_function
from agentic_rag.knowledge_graph import math_knowledge_graph
from agentic_rag.math_retriever import CHUNK_COLLECTION_NAME
from agentic_rag.math_taxonomy import adaptive_chunk_size, classify_math_text, extract_formulas, infer_prerequisites
from config import CHROMA_PATH, EXCEL_METADATA_COLUMNS, KNOWLEDGE_DATA_PATH

DATA_PATH = KNOWLEDGE_DATA_PATH
SUMMARY_COLLECTION_NAME = "math_summaries"
BATCH_SIZE = 512


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_math_document(document: Document) -> bool:
    source = str(document.metadata.get("source", ""))
    if "数学" in source or any(document.metadata.get(key) for key in ("章节", "知识点", "年级")):
        return True
    markers = (
        "一元一次方程", "一元二次方程", "不等式", "因式分解", "分式方程",
        "一次函数", "二次函数", "反比例函数", "全等三角形", "相似三角形",
        "勾股定理", "圆周角", "等可能试验", "概率公式",
    )
    return sum(marker in document.page_content for marker in markers) >= 2


def load_documents_from_directory(directory_path: str) -> list[Document]:
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }
    documents: list[Document] = []
    paths = [path for path in Path(directory_path).rglob("*") if path.is_file() and path.name != "knowledge_graph.json"]
    for path in tqdm(paths, desc="加载数学资料"):
        extension = path.suffix.lower()
        try:
            if extension in (".xlsx", ".xls"):
                frame = pd.read_excel(path)
                for row_index, row in frame.iterrows():
                    metadata = {"source": str(path), "row_index": int(row_index)}
                    parts = []
                    for column in frame.columns:
                        value = "" if pd.isna(row[column]) else str(row[column])
                        parts.append(f"{column}: {value}")
                        if column in EXCEL_METADATA_COLUMNS:
                            metadata[column] = value
                    documents.append(Document(page_content=_clean_text("\n".join(parts)), metadata=metadata))
            elif extension in loader_map:
                loader = loader_map[extension](str(path), encoding="utf-8") if extension in (".txt", ".md") else loader_map[extension](str(path))
                for document in loader.load():
                    document.page_content = _clean_text(document.page_content)
                    documents.append(document)
        except Exception as exc:
            print(f"跳过无法加载的文件 {path}: {exc}")
    return [document for document in documents if document.page_content and _is_math_document(document)]


def _protect_formulas(text: str) -> tuple[str, dict[str, str]]:
    protected, replacements = text, {}
    for index, formula in enumerate(extract_formulas(text)):
        placeholder = f"FORMULA_TOKEN_{index}_END"
        protected = protected.replace(formula, placeholder)
        replacements[placeholder] = formula
    return protected, replacements


def _error_class(text: str) -> str:
    if any(word in text for word in ("符号", "变号", "正负")):
        return "符号错误"
    if any(word in text for word in ("公式", "定理", "条件")):
        return "公式条件误用"
    if any(word in text for word in ("漏", "跳步", "对应")):
        return "步骤遗漏"
    return "计算或概念错误"


def _llm_metadata(section: str) -> dict:
    from agentic_rag.chains import get_metadata_extraction_chain

    try:
        return get_metadata_extraction_chain().invoke({"text": section})
    except Exception as exc:
        print(f"LLM 元数据抽取降级为规则: {exc}")
        return {}


def split_math_document(document: Document, use_llm_metadata: bool = False) -> list[Document]:
    """Split headings, protect formulas, and attach complete retrieval metadata."""
    sections = [section.strip() for section in re.split(r"(?m)(?=^##\s+)", document.page_content) if section.strip()]
    chunks: list[Document] = []
    for section_index, section in enumerate(sections):
        classification = classify_math_text(f"{document.metadata.get('source', '')}\n{section}")
        if classification.chapter == "综合" and len(section) < 100:
            continue
        formulas = extract_formulas(section)
        prerequisites = infer_prerequisites(classification.knowledge_points)
        enhanced = _llm_metadata(section) if use_llm_metadata else {}
        formulas = list(dict.fromkeys([*formulas, *enhanced.get("formulas", [])]))
        prerequisites = list(dict.fromkeys([*prerequisites, *enhanced.get("prerequisites", [])]))
        chunk_size = adaptive_chunk_size(section)
        protected, replacements = _protect_formulas(section)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=max(40, chunk_size // 6),
            separators=["\n### ", "\n\n", "\n", "。", "；", "，", " "],
        )
        metadata = dict(document.metadata)
        metadata.update({
            "grade": enhanced.get("grade") or metadata.get("年级") or classification.grade,
            "chapter": enhanced.get("chapter") or metadata.get("章节") or classification.chapter,
            "knowledge_points": "、".join(enhanced.get("knowledge_points") or classification.knowledge_points),
            "question_type": enhanced.get("question_type") or metadata.get("题型") or classification.question_type,
            "error_class": enhanced.get("error_class") or metadata.get("错误分类") or _error_class(section),
            "formula_ids": "、".join(f"F-{_stable_id(formula)[:10]}" for formula in formulas),
            "formulas": json.dumps(formulas, ensure_ascii=False),
            "prerequisites": "、".join(prerequisites),
            "chunk_tokens": chunk_size,
            "section_index": section_index,
        })
        protected_chunks = splitter.split_documents([Document(page_content=protected, metadata=metadata)])
        for chunk in protected_chunks:
            for placeholder, formula in replacements.items():
                chunk.page_content = chunk.page_content.replace(placeholder, formula)
            chunks.append(chunk)
        missing_formulas = [formula for formula in formulas if formula in section and not any(formula in chunk.page_content for chunk in protected_chunks)]
        if missing_formulas:
            raise ValueError(f"公式在分块过程中丢失: {missing_formulas}")
        for point in (enhanced.get("knowledge_points") or classification.knowledge_points):
            math_knowledge_graph.add(point, prerequisites)
    return chunks


def _recreate_collection(client, name: str, embedding_function):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, embedding_function=embedding_function)


def _add_batches(collection, ids, documents, metadatas):
    for start in tqdm(range(0, len(ids), BATCH_SIZE), desc=f"写入 {collection.name}"):
        end = start + BATCH_SIZE
        collection.add(ids=ids[start:end], documents=documents[start:end], metadatas=metadatas[start:end])


def main():
    parser = argparse.ArgumentParser(description="构建初中数学多索引知识库")
    parser.add_argument("--llm-metadata", action="store_true", help="调用 LLM 增强公式、知识点和依赖元数据")
    args = parser.parse_args()
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"数据目录不存在: {DATA_PATH}")
    source_documents = load_documents_from_directory(DATA_PATH)
    if not source_documents:
        raise RuntimeError("未找到初中数学资料；请将教材、错题集或题解放入 data/。")
    chunks = [chunk for document in tqdm(source_documents, desc="公式感知切分") for chunk in split_math_document(document, args.llm_metadata)]
    embedding_function = get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chunk_collection = _recreate_collection(client, CHUNK_COLLECTION_NAME, embedding_function)
    summary_collection = _recreate_collection(client, SUMMARY_COLLECTION_NAME, embedding_function)

    chunk_ids = [_stable_id(chunk.metadata.get("source"), chunk.metadata.get("page", 0), index, chunk.page_content[:80]) for index, chunk in enumerate(chunks)]
    _add_batches(chunk_collection, chunk_ids, [chunk.page_content for chunk in chunks], [chunk.metadata for chunk in chunks])

    summaries, summary_metadata, summary_ids = [], [], []
    for index, document in enumerate(source_documents):
        classification = classify_math_text(document.page_content)
        summary_ids.append(_stable_id(document.metadata.get("source", "unknown"), index))
        summaries.append(document.page_content[:800])
        summary_metadata.append({"source": document.metadata.get("source", "unknown"), "chapter": classification.chapter, "grade": classification.grade, "knowledge_points": "、".join(classification.knowledge_points)})
    _add_batches(summary_collection, summary_ids, summaries, summary_metadata)
    math_knowledge_graph.save()
    print(f"知识库完成：{len(source_documents)} 份资料，{len(chunks)} 个公式感知 Chunk，GraphRAG {len(math_knowledge_graph.as_dict()['prerequisites'])} 个节点。")


if __name__ == "__main__":
    main()
